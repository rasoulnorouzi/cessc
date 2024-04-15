from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support
    )

from transformers import Trainer,  DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class FineTuner:

    """
    FineTuner class for training a model

    Args:
        model (PreTrainedModel): model to train
        tokenizer (PreTrainedTokenizer): tokenizer to use
        training_args (TrainingArguments): training arguments
        
    Returns:
        FineTuner: FineTuner object for training a model 
    """
    def __init__(self, model, tokenizer, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.trainer = None

    # tokenize the dataset 
    def tokenize_dataset(self, dataset):
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, max_length=512)
        return dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # train the model 
    def train(self, train_dataset, val_dataset):
        """
        Train the model

        Args:
            train_dataset (Dataset): training dataset
            val_dataset (Dataset): validation dataset

        Returns:
            a trained model

        """
        tokenized_train_dataset = self.tokenize_dataset(train_dataset)
        tokenized_val_dataset = self.tokenize_dataset(val_dataset)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        def compute_metrics(p):
            labels = p.label_ids
            preds = p.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        self.trainer.train()