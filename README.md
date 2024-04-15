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

Each CSV file is formatted with consistent column structures to facilitate seamless model training, validation, and testing processes across different dataset categories.

## Scripts

This directory contains the Python scripts that are essential for the machine learning workflow in this project, including configurations, model evaluation, fine-tuning, and utility functions.

### `configurations.py`

This script holds all the configuration parameters for the project. It is designed to centralize and manage the settings that control various aspects of the machine learning process, such as model parameters, training options, and dataset paths.

### `evaluation.py`

Contains the code for evaluating the models. This script is used to apply the trained model on the test datasets and to calculate performance metrics that gauge the effectiveness of the model in causal extraction.

### `finetuner.py`

This script is used for fine-tuning the models on the specified training data. It includes functions and classes that handle the training process, including data loading, model updates, and logging training progress.

### `model_factory.py`

Acts as a factory for creating different machine learning models, allowing for easy instantiation and management of various model architectures for the project. It supports the following models:

- `BERT`: The Bidirectional Encoder Representations from Transformers model, known for its effectiveness in natural language processing tasks.
- `SciBERT`: A variant of BERT pretrained on scientific text, which is especially suitable for tasks in the scientific domain.
- `Roberta`: A robustly optimized BERT approach, which has been shown to outperform BERT on several benchmarking tasks.
- `LLAMA2-7b`: A large language model adapted for a range of tasks, providing powerful text understanding and generation capabilities.
- `Mistral-7b`: Another large-scale language model designed for high performance in a variety of complex language tasks.

This script is responsible for handling model selection based on the project configuration, initializing them with appropriate weights and settings, and preparing them for integration into the training or evaluation workflows.

### `requirements.txt`

Specifies all the Python dependencies required by the scripts. This file ensures consistent environments across different setups, making it easier to replicate the project's results. The following packages and versions are specified:
- `numpy==1.25.2`: A package for scientific computing with Python.
- `torch`: An open-source machine learning library.
- `pandas==2.0.3`: A library providing high-performance, easy-to-use data structures and data analysis tools.
- `datasets==2.18.0`: A library for easily accessing and sharing datasets.
- `transformers==4.39.3`: A library providing thousands of pre-trained models to perform tasks on texts.
- `evaluate==0.4.1`: A library for evaluating models (duplicate entry removed).
- `bitsandbytes==0.43.1`: A library for optimizing CUDA operations.
- `accelerate==0.29.2`: A library for accelerating training on CPUs and GPUs.
- `peft==0.10.0`: A performance estimator for transformers.
- `sentencepiece==0.2.0`: A library for unsupervised text tokenization and detokenization.
- `loralib`: A library or module required for the project (details not specified).

### `utility.py`

Provides utility functions that are used across the project. These functions might include data preprocessing, result visualization, or any other helper functions that are not part of the core machine learning process but support it.

## Reproducibility
```
git clone
cd cessc
```
The `reproducibility.ipynb` notebook is a key resource in this project for ensuring that our results can be consistently replicated. It provides a step-by-step example of how to reproduce the training and testing process specifically for the BERT model across different datasets. The notebook is structured as follows:

- **Environment Setup**: Instructions on setting up the computational environment, including the installation of all necessary dependencies as detailed in `requirements.txt`.
- **Kernel Restart**: Guidelines on restarting the Jupyter kernel before commencing the fine-tuning of any model, to ensure a clean state.
- **Model Training**:
  - Training the BERT model on the general-purpose dataset (`general_train.csv`).
  - Training on the social science dataset (`ssc_train.csv`).
  - Training on the merged dataset (`all_train.csv`).
- **Model Testing**:
  - Testing the BERT model on the general-purpose test set (`general_test.csv`).
  - Testing on the social science test set (`ssc_test.csv`).

For fine-tuning additional models such as SciBERT, Roberta, LLAMA2-7b, and Mistral-7b, one can follow the same process as detailed in the notebook for the BERT model. Each step is annotated with clear instructions and code snippets to guide the user through the process.

For exact reproducibility, it is essential to run these processes on a GPU A100. This specification ensures that the computational power and architecture are consistent with the original environment where the models were trained and tested.

The reproducibility guide underscores the commitment to scientific rigor and transparency in this project, providing a clear path for others to validate and build upon our work.



