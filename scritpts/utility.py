from torch.utils.data import DataLoader
import torch

def error_analysis(model, tokenizer, dataset, batch_size=8):
    class SimpleDataset:
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return {'text': self.texts[idx], 'label': self.labels[idx]}

    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    simple_dataset = SimpleDataset(texts, labels)

    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        encoding = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        encoding['labels'] = labels  # Include labels in the batch
        return encoding

    dataloader = DataLoader(simple_dataset, batch_size=batch_size, collate_fn=collate_fn)

    true_labels = []
    predicted_labels = []

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_batch_labels = batch['labels'].cpu().numpy()  # Now 'labels' is included in the batch
            true_labels.extend(true_batch_labels)
            predicted_labels.extend(preds)

    misclassified_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)) if true != pred]
    misclassified_samples = [
        {
            'text': dataset[i]['text'],
            'true_label': dataset[i]['label'],
            'predicted_label': predicted_labels[i]
        }
        for i in misclassified_indices
    ]

    return misclassified_samples