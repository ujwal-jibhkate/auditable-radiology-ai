from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from typing import List
import torch
from torchvision import transforms as T
from transformers import BertTokenizer
from PIL import Image
import io
from huggingface_hub import hf_hub_download

# Import our custom classes and functions
from model import ImageCaptioningModel, ResizeAndPad
from auditors import consistency_auditor, RuleBasedLabeler

# --- 1. Configuration and Setup ---
class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "../models/hierarchical_v4_best_model.pth" # Adjust path relative to backend folder
    tokenizer_name = "emilyalsentzer/Bio_ClinicalBERT"
    decoder_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    vision_model_name = 'swin_base_patch4_window7_224.ms_in22k'
    image_size = 224
    hidden_dim = 768
    num_decoder_layers = 6
    nhead = 12
    max_length = 128

# --- 2. Load Model and Assets (Done once on startup) ---
# Load model

print("Downloading model from Hugging Face Hub...")
model_path = hf_hub_download(
    repo_id="ujwal-jibhkate/auditable-radiology-ai-model", 
    filename="hierarchical_v4_best_model.pth",
    cache_dir="../models" # Download to the models folder
)
print("Model download complete.")

# Load model from the downloaded path
model = ImageCaptioningModel(CFG).to(CFG.device)
checkpoint = torch.load(model_path, map_location=CFG.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer and transforms
tokenizer = BertTokenizer.from_pretrained(CFG.tokenizer_name)
transforms = T.Compose([
    ResizeAndPad(CFG.image_size),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instantiate the labeler to get pathology names
labeler = RuleBasedLabeler()
pathologies = labeler.pathologies

print("Model and assets loaded successfully.")

# --- 3. Define API Data Models ---

class LabelPrediction(BaseModel):
    label: str
    probability: float = Field(..., ge=0, le=1)


class AuditResult(BaseModel):
    is_consistent: bool
    

class PredictionResponse(BaseModel):
    generated_report: str
    predicted_labels: List[LabelPrediction]
    audit: AuditResult

# --- 4. FastAPI App and Endpoints ---
app = FastAPI(title="Radiology Report AI API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Auditable Radiology AI API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(image_file: UploadFile = File(...)):
    # Read and process the uploaded image
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transforms(image).unsqueeze(0).to(CFG.device)

    # --- Run Inference ---
    # Generate report text
    generated_sequence = [tokenizer.cls_token_id]
    for _ in range(CFG.max_length - 1):
        input_ids = torch.LongTensor([generated_sequence]).to(CFG.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            text_logits, _ = model(image_tensor, input_ids, attention_mask)
        next_token_id = text_logits[:, -1, :].argmax(1).item()
        generated_sequence.append(next_token_id)
        if next_token_id == tokenizer.sep_token_id:
            break
    generated_report = tokenizer.decode(generated_sequence, skip_special_tokens=True)

    # Predict labels
    dummy_input = torch.LongTensor([[tokenizer.cls_token_id]]).to(CFG.device)
    dummy_mask = torch.ones_like(dummy_input)
    with torch.no_grad():
        _, topic_logits = model(image_tensor, dummy_input, dummy_mask)
    probabilities = torch.sigmoid(topic_logits).squeeze().cpu().numpy()
    
    predicted_labels = [
        LabelPrediction(label=pathologies[i], probability=float(probabilities[i]))
        for i in range(len(pathologies))
    ]

    # --- Run Audit ---
    inconsistent_count = consistency_auditor([generated_report])
    audit_result = AuditResult(is_consistent=(inconsistent_count == 0))

    return PredictionResponse(
        generated_report=generated_report,
        predicted_labels=predicted_labels,
        audit=audit_result
    )