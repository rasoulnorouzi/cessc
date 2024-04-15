# evaluation.py

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


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

    class_report = classification_report(true_labels, predicted_labels, target_names=["Class 0", "Class 1"])
    
    # beatiful print
    print("Classification Report:")
    print("=================================")
    print(class_report)
    print("=================================")

    return class_report




