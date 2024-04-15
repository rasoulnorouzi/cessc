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
    def __init__(self, model, tokenizer, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.trainer = None

    def tokenize_dataset(self, dataset):
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, max_length=512)
        return dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    def train(self, train_dataset, val_dataset):
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