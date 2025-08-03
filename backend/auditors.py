import re
import tqdm
import os
import pandas as pd


class RuleBasedLabeler:
    """An improved rule-based labeler using sentence-level analysis."""
    def __init__(self):
        self.pathologies = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
        ]
        self.keywords = {
            'Cardiomegaly': [r'cardiomegaly', r'cardiac silhouette is enlarged', r'enlarged heart'],
            'Lung Opacity': [r'opacity', r'opacities'],
            'Lung Lesion': [r'lesion', r'nodule', r'mass'],
            'Edema': [r'edema', r'pulmonary edema', r'heart failure', r'chf'],
            'Consolidation': [r'consolidation', r'consolidative'],
            'Pneumonia': [r'pneumonia', r'infection'],
            'Atelectasis': [r'atelectasis', r'atelectatic'],
            'Pneumothorax': [r'pneumothorax', r'pneumothoraces'],
            'Pleural Effusion': [r'pleural effusion', r'effusion'],
            'Fracture': [r'fracture', r'rib fracture'],
            'Support Devices': [r'tube', r'catheter', r'pacemaker', 'device', 'line', 'etracheal tube']
        }
        self.negation = [r'no evidence of', r'no sign of', r'not seen', r'are clear', r'is normal', r'is unremarkable', r'no ', r'negative for', r'without', r'clear of']

    def get_labels(self, report_text):
        labels = {p: 0 for p in self.pathologies}
        sentences = [sent.text.lower() for sent in nlp(report_text).sents]

        positive_found = False
        for pathology, phrases in self.keywords.items():
            for sentence in sentences:
                for phrase in phrases:
                    if re.search(r'\b' + phrase + r'\b', sentence): # Use word boundaries for more exact match
                        # Check for negation ONLY within the same sentence
                        is_negated = any(re.search(neg, sentence) for neg in self.negation)
                        if not is_negated:
                            labels[pathology] = 1
                            positive_found = True
                            break # Found in one sentence, move to next pathology
                if labels[pathology] == 1:
                    break

        if not positive_found:
            labels['No Finding'] = 1

        return [labels[p] for p in self.pathologies]


def create_final_labeled_data(cfg):
    """
    Final data prep function using our robust rule-based labeler.
    """
    if os.path.exists(cfg.final_df_path):
        print(f"Loading final labeled data from {cfg.final_df_path}")
        return pd.read_pickle(cfg.final_df_path)

    print(f"Loading intermediate data from {cfg.intermediate_df_path}")
    df = pd.read_pickle(cfg.intermediate_df_path)
    df.dropna(subset=['image_file', 'full_report'], inplace=True)
    df = df[df['full_report'] != ''].copy().reset_index(drop=True)

    print("Generating CheXpert labels with our custom Rule-Based Labeler...")

    labeler = RuleBasedLabeler()
    all_labels = []
    for report in tqdm(df["full_report"], desc="Labeling Reports"):
        all_labels.append(labeler.get_labels(report))

    pathologies = labeler.pathologies
    labels_df = pd.DataFrame(all_labels, columns=pathologies)

    final_df = pd.concat([df, labels_df], axis=1)
    final_df['labels'] = final_df[pathologies].values.tolist()

    print(f"Saving final labeled data to {cfg.final_df_path}")
    final_df.to_pickle(cfg.final_df_path)

    return final_df


def consistency_auditor(reports):
    """A more robust version of the consistency auditor."""
    contradiction_pairs = [
        ('pneumothorax', 'no pneumothorax'),
        ('effusion', 'no effusion'),
        ('consolidation', 'no consolidation'),
        ('edema', 'no edema'),
        ('fracture', 'no fracture'),
        ('opacity', 'clear lungs')
    ]
    inconsistent_count = 0

    for report in reports:
        for pos_term, neg_term in contradiction_pairs:
            # Use negative lookbehind `(?<!no )` to find the positive term only if it's NOT preceded by "no "
            # This prevents the positive term from matching inside the negative phrase.
            positive_match = re.search(r'(?<!no\s)\b' + pos_term + r'\b', report, re.IGNORECASE)
            negative_match = re.search(r'\b' + neg_term + r'\b', report, re.IGNORECASE)

            if positive_match and negative_match:
                inconsistent_count += 1
                break
    return inconsistent_count