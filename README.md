# causal-extraction-social-science 

## Datasets

# causal-extraction-social-science

## Datasets

The datasets provided in this repository are structured to support causal extraction for social science research as well as general-purpose machine learning tasks. The datasets are divided into three categories:

### Social Science Datasets (`ssc_`)

- `ssc_train.csv`: This is the training set specifically curated for social science applications. It contains examples that are relevant to social science topics, annotated with causal relations.

- `ssc_val.csv`: The validation set used for tuning the models on social science data. It includes a variety of social science examples for validating the performance of the causal extraction models.

- `ssc_test.csv`: The test set comprises unseen social science data used for evaluating the model's performance in identifying causal relationships within the domain of social science.

### General-Purpose Datasets (`general_`)

- `general_train.csv`: This is the training set derived from a combination of established causal datasets: AltLex, BECAUSE 2.0, CausalTimeBank (CTB), EventStoryLine (ESL), and SemEval 2010 Task 8. It is designed to train models on a diverse array of causal relations within general contexts.

- `general_val.csv`: The validation set consists of carefully selected examples from the aforementioned datasets to provide a balanced and diverse array of topics for model validation. This ensures that the model's performance is representative of its generalization capabilities.

- `general_test.csv`: The test set is composed of examples from the combined datasets, providing a comprehensive challenge to assess the model's ability to generalize causal extraction across varied general-domain texts. It serves as a benchmark for model performance against a diverse set of established sources.

These datasets collectively form a robust resource for training, validating, and testing causal extraction models, aiming to perform well on a broad spectrum of general-domain examples.

### Merged Datasets (`all_`)

- `all_train.csv`: A merged training set that combines both social science and general-purpose examples. This dataset is ideal for training models that require a comprehensive understanding of causal relationships across diverse subjects.

- `all_val.csv`: The merged validation set includes a mix of examples from both the social science and general domains. It is used for fine-tuning the models to perform well on a wide range of topics.

- `all_test.csv`: The merged test set provides a comprehensive evaluation across all included topics, testing the model's robustness in causal extraction across both specialized and generalized contexts.

Each CSV file is formatted with consistent column structures to facilitate seamless model training, validation, and testing processes across different dataset categories.

