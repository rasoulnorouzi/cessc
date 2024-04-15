# evaluation.py

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


class SimpleDataset(Dataset):

    """
    Simple dataset class for text classification

    Args:
        texts (list): list of texts
        labels (list): list of labels

    Returns:
        Dataset: Dataset object
    """
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
        dataset, batch_size=8
        ):
    
    """
    Evaluate a model on a dataset

    Args:
        model (PreTrainedModel): model to evaluate
        tokenizer (PreTrainedTokenizer): tokenizer to use
        dataset (list): list of dictionaries with 'text' and 'label'
        batch_size (int): batch size

    Returns:
        str: classification report
    """
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    simple_dataset = SimpleDataset(texts, labels)

    # collate function for DataLoader to handle variable length texts and labels 
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        encoding = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        encoding['labels'] = labels
        return encoding

    # create DataLoader object for the dataset 
    dataloader = DataLoader(simple_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # move model to device and set to eval mode 
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

    # classification report 
    class_report = classification_report(true_labels, predicted_labels, target_names=["Class 0", "Class 1"])
    
    # beatiful print
    print("Classification Report:")
    print("=================================")
    print(class_report)
    print("=================================")

    return class_report




