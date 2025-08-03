# --- Model Architecture ---
import torch
import torch.nn as nn
import timm
from transformers import BertModel, BertTokenizer, BertConfig
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import os


class ResizeAndPad:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        w, h = image.size
        new_w, new_h = (self.output_size, int(h * self.output_size / w)) if w > h else (int(w * self.output_size / h), self.output_size)
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_image = Image.new("RGB", (self.output_size, self.output_size))
        new_image.paste(resized_image, ((self.output_size - new_w) // 2, (self.output_size - new_h) // 2))
        return new_image


class IU_XRay_Dataset(Dataset):
    def __init__(self, df, tokenizer, transforms, image_path, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.image_path = image_path
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_tensor = self.transforms(Image.open(os.path.join(self.image_path, row['image_file'])).convert("RGB"))
        text_encoding = self.tokenizer(
            row['full_report'], max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors="pt"
        )
        return {
            "image": image_tensor,
            "input_ids": text_encoding['input_ids'].squeeze(),
            "attention_mask": text_encoding['attention_mask'].squeeze(),
            "labels": torch.FloatTensor(row['labels'])
        }

class VisionEncoder(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.feature_dim = self.model.head.in_features
        self.model.head = nn.Identity()

    def forward(self, image):
        
        return self.model(image)


class TextDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained(cfg.tokenizer_name)
        vocab_size = tokenizer.vocab_size

        decoder_config = BertConfig.from_pretrained(cfg.decoder_model_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.num_hidden_layers = cfg.num_decoder_layers
        decoder_config.num_attention_heads = cfg.nhead

        self.decoder = BertModel(config=decoder_config)

        pretrained_bert = BertModel.from_pretrained(cfg.decoder_model_name)
        self.decoder.load_state_dict(pretrained_bert.state_dict(), strict=False)

        self.fc = nn.Linear(cfg.hidden_dim, vocab_size)

    def forward(self, text_input, attention_mask, encoder_output):
        # The BERT decoder can now use the attention mask to ignore padding
        decoder_output = self.decoder(
            input_ids=text_input,
            attention_mask=attention_mask, # Pass the padding mask here
            encoder_hidden_states=encoder_output
        ).last_hidden_state

        predictions = self.fc(decoder_output)
        return predictions


class ImageCaptioningModel(nn.Module):
    def __init__(self, cfg, num_classes=14):
        super().__init__()
        self.encoder = VisionEncoder(cfg.vision_model_name)
        self.decoder = TextDecoder(cfg) # Our BioBERT-initialized decoder

        # Bridge layer between vision and text dimensions
        self.projection = nn.Linear(self.encoder.feature_dim, cfg.hidden_dim)

        # --- Stage 1: Topic Prediction Head ---
        # This head attaches directly to the vision encoder's output
        self.topic_prediction_head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, num_classes)
        )

        # --- Stage 2: Conditioning Layer ---
        # This layer will combine the predicted topic vector with the text embedding
        # before feeding it to the decoder's self-attention layers.
        self.topic_gating_layer = nn.Linear(cfg.hidden_dim + num_classes, cfg.hidden_dim)

    def forward(self, image, text_input, attention_mask):
        # --- Get Raw Visual Features ---
        visual_features = self.encoder(image)
        # Reshape to [Batch, Patches, Dim]
        b, h, w, c = visual_features.shape
        visual_features = visual_features.view(b, h * w, c)

        # --- STAGE 1: Predict Clinical Topics from Image ---
        # We use the average of all patch features for topic prediction
        mean_visual_features = visual_features.mean(dim=1)
        topic_logits = self.topic_prediction_head(mean_visual_features)

        # Get the topic probabilities to condition the decoder
        topic_probs = torch.sigmoid(topic_logits)

        # --- STAGE 2: Generate Text Conditioned on Topics ---
        # Project visual features for the decoder's cross-attention
        projected_features = self.projection(visual_features)

        # Get the initial text embeddings from the decoder
        text_embeddings = self.decoder.decoder.embeddings(input_ids=text_input)

        # --- Conditioning Step ---
        # Expand the topic probabilities to match the sequence length
        topic_probs_expanded = topic_probs.unsqueeze(1).expand(-1, text_input.size(1), -1)

        # Concatenate text embeddings with topic probabilities
        conditioned_input = torch.cat([text_embeddings, topic_probs_expanded], dim=-1)

        # Pass the concatenated input through the gating layer
        gated_input = self.topic_gating_layer(conditioned_input)

        # --- Standard Decoding ---
        # The rest of the decoding process now uses this "topic-aware" input
        decoder_output = self.decoder.decoder(
            inputs_embeds=gated_input,
            attention_mask=attention_mask,
            encoder_hidden_states=projected_features
        ).last_hidden_state

        text_logits = self.decoder.fc(decoder_output)

        # Return both sets of logits for our two loss functions
        return text_logits, topic_logits