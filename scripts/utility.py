from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import set_seed as hf_set_seed
from scripts.configurations import seed_value

seed_value = seed_value

def set_all_seeds(seed_value = seed_value):
    """
    Set seed for reproducibility across all used libraries and systems that support seed setting.
    
    Parameters:
        seed_value (int): The seed number to use for all random number generators.
    """
    # Set seed for PyTorch
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.cuda.manual_seed(seed_value)
    
    # Set seed for Numpy
    np.random.seed(seed_value)
    
    # Set seed for Python's random module
    random.seed(seed_value)
    
    # Set seed for any hashing-based operations
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    # Ensure reproducibility for PyTorch when using CUDA (if available)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set seed for Hugging Face transformers
    hf_set_seed(seed_value)


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
        encoding = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
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



import torch

@torch.no_grad()
def analyze_predictions(data, model, device, tokenizer, batch_size=32):
    """
    Analyzes model predictions on a given dataset.

    Parameters:
    data (list): List of input data samples to analyze.
    model (torch.nn.Module): The model to use for predictions.
    device (torch.device): The device (CPU or GPU) to perform the computations on.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer to preprocess the input data.
    batch_size (int, optional): The number of samples to process in each batch. Default is 32.

    Returns:
    tuple: A tuple containing:
        - predictions (list): The predicted labels for the input data.
        - logits (list): The raw output logits from the model for each input sample.
    """

    predictions = []  # List to store the predicted labels
    logits = []       # List to store the raw output logits

    # Move the model to the specified device and set it to evaluation mode
    model.to(device)
    model.eval()

    # Process the data in batches
    for i in range(0, len(data), batch_size):
        # Tokenize the current batch of data
        batch = tokenizer(data[i:i+batch_size], padding=True, truncation=True, return_tensors='pt')
        
        # Move the tokenized batch to the specified device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Get the model's output logits for the batch
        outputs = model(**batch)
        
        # Predict the label by taking the argmax of the logits along the last dimension
        predictions.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
        
        # Store the logits for the batch
        logits.extend(outputs.logits.cpu().numpy())

    # Return the predictions and logits
    return predictions, logits




def bootstrap_confidence_interval(true_labels, predicted_labels, metric_function=f1_score, n_iterations=10000, alpha=0.05, **metric_kwargs):
    """
    Computes the bootstrap confidence interval for a given metric.

    Parameters:
    true_labels (list or array-like): The true labels of the data.
    predicted_labels (list or array-like): The predicted labels of the data.
    metric_function (function, optional): The metric function to evaluate. Default is f1_score.
    n_iterations (int, optional): The number of bootstrap iterations. Default is 10,000.
    alpha (float, optional): Significance level for the confidence interval. Default is 0.05.
    **metric_kwargs: Additional keyword arguments to pass to the metric function.

    Returns:
    tuple: A tuple containing the mean metric score, lower bound of the confidence interval, and upper bound of the confidence interval.
    """

    # Set default for 'average' to 'macro' if using f1_score and not explicitly provided
    if metric_function == f1_score and 'average' not in metric_kwargs:
        metric_kwargs['average'] = 'macro'
    
    metric_scores = []  # List to store metric scores for each bootstrap iteration
    n_samples = len(true_labels)  # Number of samples in the data
    
    for _ in range(n_iterations):
        # Resample indices with replacement
        resample_indices = np.random.randint(0, n_samples, n_samples)
        
        # Create resampled true and predicted labels based on the resample indices
        true_labels_resampled = [true_labels[i] for i in resample_indices]
        predicted_labels_resampled = [predicted_labels[i] for i in resample_indices]
        
        # Calculate the metric for the resampled data
        metric_value = metric_function(true_labels_resampled, predicted_labels_resampled, **metric_kwargs)
        metric_scores.append(metric_value)

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound, upper_bound = np.percentile(metric_scores, [alpha/2 * 100, (1 - alpha/2) * 100])
    
    # Return the mean metric score and the confidence interval bounds
    return np.mean(metric_scores), lower_bound, upper_bound


def bootstrap_hypothesis_test(metric_sample1, metric_sample2, ci_sample1, ci_sample2, n_iterations=1000):
    """
    Performs a bootstrap hypothesis test to compare two metric samples.

    Parameters:
    metric_sample1 (float): The metric score for sample 1.
    metric_sample2 (float): The metric score for sample 2.
    ci_sample1 (tuple): Confidence interval (lower, upper) for sample 1.
    ci_sample2 (tuple): Confidence interval (lower, upper) for sample 2.
    n_iterations (int, optional): The number of bootstrap iterations. Default is 1,000.

    Returns:
    tuple: A tuple containing the p-value and the effect size (Cohen's d).
    """

    # Estimate standard errors from confidence intervals for both samples
    se_sample1 = (ci_sample1[1] - ci_sample1[0]) / (2 * 1.96)
    se_sample2 = (ci_sample2[1] - ci_sample2[0]) / (2 * 1.96)
    
    # Generate bootstrap samples assuming normal distribution with calculated standard errors
    sample1_bootstrap = np.random.normal(metric_sample1, se_sample1, n_iterations)
    sample2_bootstrap = np.random.normal(metric_sample2, se_sample2, n_iterations)
    
    # Calculate the differences between the bootstrap samples
    metric_differences = sample1_bootstrap - sample2_bootstrap
    
    # Calculate the p-value as the proportion of differences less than or equal to zero
    p_value = np.mean(metric_differences <= 0)
    
    # Calculate the effect size (Cohen's d) using pooled standard deviation
    pooled_standard_deviation = np.sqrt((se_sample1**2 + se_sample2**2) / 2)
    effect_size = (metric_sample1 - metric_sample2) / pooled_standard_deviation
    
    # Return the p-value and effect size
    return p_value, effect_size