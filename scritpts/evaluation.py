# evaluation.py

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
import numpy as np


class SimpleDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}

def evaluate(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer, 
        dataset, batch_size=8,
        average='binary'
        ):
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    simple_dataset = SimpleDataset(texts, labels)

    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        encoding = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        encoding['labels'] = labels
        return encoding

    dataloader = DataLoader(simple_dataset, batch_size=batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_batch_labels = batch['labels'].cpu().numpy()
            true_labels.extend(true_batch_labels)
            predicted_labels.extend(preds)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=average)
    acc = accuracy_score(true_labels, predicted_labels)

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    class_report = classification_report(true_labels, predicted_labels, target_names=["Class 0", "Class 1"])

    return metrics, confusion_mat, class_report




def calculate_metrics(confusion_matrix):
    TN, FP, FN, TP = confusion_matrix.ravel()

    # Calculating metrics for the positive class
    precision_pos = TP / (TP + FP) if TP + FP > 0 else 0
    recall_pos = TP / (TP + FN) if TP + FN > 0 else 0
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if precision_pos + recall_pos > 0 else 0

    # Calculating metrics for the negative class
    precision_neg = TN / (TN + FN) if TN + FN > 0 else 0
    recall_neg = TN / (TN + FP) if TN + FP > 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if precision_neg + recall_neg > 0 else 0

    # Calculating accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0

    # Calculating weighted averages
    total_instances = TP + TN + FP + FN
    weighted_precision = ((TP + FP) / total_instances * precision_pos) + ((TN + FN) / total_instances * precision_neg)
    weighted_recall = ((TP + FP) / total_instances * recall_pos) + ((TN + FN) / total_instances * recall_neg)
    weighted_f1 = ((TP + FP) / total_instances * f1_pos) + ((TN + FN) / total_instances * f1_neg)

    # Printing the metrics
    print(f'Precision (Positive Class): {precision_pos:.3f}')
    print(f'Recall (Positive Class): {recall_pos:.3f}')
    print(f'F1-Score (Positive Class): {f1_pos:.3f}\n')

    print(f'Precision (Negative Class): {precision_neg:.3f}')
    print(f'Recall (Negative Class): {recall_neg:.3f}')
    print(f'F1-Score (Negative Class): {f1_neg:.3f}\n')

    print(f'Accuracy: {accuracy:.3f}\n')

    print(f'Weighted Precision: {weighted_precision:.4f}')
    print(f'Weighted Recall: {weighted_recall:.4f}')
    print(f'Weighted F1-Score: {weighted_f1:.4f}')
    