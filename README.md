# Causal Sentence Extractor Project

Welcome to the Causal Sentence Extractor repository, designed to facilitate the extraction of causal sentences from social science texts using transformers models.

## Table of Contents
- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Datasets](#datasets)
- [Usage](#usage)
  - [Getting Started](#getting-started)
  - [Fine-Tuning Models](#fine-tuning-models)
  - [Reproducibility](#reproducibility)
- [Inference with Huggingface API](#Inference-with-Huggingface-API) 
- [Project Structure](#project-structure)
- [Citing This Work](#citing-this-work)
- [License](#license)

## Introduction
This repository contains the resources needed to run, modify, and evaluate the text mining models developed for extracting causal sentences from social science context.

## Installation and Setup
To set up your environment for full functionality of the tools, please see `research_environment.md`. For optimal performance and reproducibility, using a NVIDIA A100 GPU is essential.

## Datasets

The datasets directory contains multiple subsets crucial for training and evaluating the models:

- **ssc_:** ssc_train.csv, ssc_val.csv, ssc_test.csv – Our custom curated datasets.
- **general_:** Datasets compiled from various sources like AltLex, BECAUSE 2.0, CausalTimeBank (CTB), EventStoryLine (ESL), and SemEval 2010 Task 8. These datasets undergo a deduplication process, balancing using undersampling, and are split into general_train.csv, general_val.csv, with general_test.csv remaining unbalanced.
- **all_:** A merged set of the above two categories for extended training and validation (all_train.csv, all_val.csv).

## Usage

### Getting Started
To use this project, first clone the repository and navigate to the project directory:
```bash
git clone https://github.com/rasoulnorouzi/cessc.git
cd cessc
```
Then open the `tutorial_reproducibility.ipynb` to see an example of how to run the code:
````bas
jupyter notebook tutorial_reproducibility.ipynb
````
### Fine-Tuning Models
To fine-tune a model:
1.  Ensure your dataset is in CSV format with `text` and `label` columns.
2.  In the script, specify your dataset's path and name.
3.  Choose the model for fine-tuning from our available models.
### Reproducibility
The provided Jupyter notebook (`tutorial_reproducibility.ipynb`) guides you through the model training and evaluation process. Remember to restart the notebook kernel after each training session to maintain consistency.

## Inference with Huggingface API
To utilize our best model for causal and non-causal classification tailored to the social science context, you can use the Huggingface `pipeline` for easy inference or load the model directly for more customized use. Below are examples demonstrating both approaches, and you can also [check the model directly on Huggingface](https://huggingface.co/rasoultilburg/ssc_bert?text=In+the+beginning%2C+Sonca+seemed+to+have+intensified+rapidly+since+its+formation+%2C+however%2C+soon+the+storm+weakened+back+to+a+minimal+tropical+storm+because+of+dry+air+entering+the+LLCC+that+caused+it+to+elongate+and+weaken.) for quick experiments.

#### Using Pipeline
Run the following example to classify text using our fine-tuned BERT model:

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="rasoultilburg/ssc_bert")

result = pipe("Our findings thus far show that the sanction reduced the number of chips that participants allocated to themselves and that it only increased the number of chips allocated to the yellow pool when there were two options.")
print(result)
# [{'label': 'LABEL_1', 'score': 0.939}]
```
#### Loading the Model Directly
For more advanced use cases, such as processing multiple sentences or integrating into a larger Python project, you can load the model and tokenizer directly:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("rasoultilburg/ssc_bert")
model = AutoModelForSequenceClassification.from_pretrained("rasoultilburg/ssc_bert")

sequences = [
    "Our findings thus far show that the sanction reduced the number of chips that participants allocated to themselves and that it only increased the number of chips allocated to the yellow pool when there were two options.",
    "First, we can assess the correlation between beliefs and contributions, which we expect to differ between types of players and which helps us to check on the player type as elicited in the P-experiment."
]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
print(output)
```

## Project Structure
````
cessc/
├── datasets/
│   ├── ssc_train.csv
│   ├── ssc_val.csv
│   ├── ssc_test.csv
│   ├── general_train.csv
│   ├── general_val.csv
│   ├── general_test.csv
│   └── all_train.csv
├── paper_files/
│   └── <list of LaTeX files and related documents>
├── scripts/
│   ├── configurations.py
│   ├── evaluation.py
│   ├── finetuner.py
│   ├── model_factory.py
│   ├── utility.py
│   └── requirements.txt
├── README.md
├── research_environment.md
└── tutorial_reproducibility.ipynb
````
## Citing This Work

If this project aids in your research, please cite it using the following BibTeX entry:
````
@article{Norouzi2024,
  author = {Norouzi, R. and Kleinberg, B. and Vermunt, J. and Van Lissa, C. J.},
  title = {Capturing Causal Claims: A Fine-Tuned Text Mining Model for Extracting Causal Sentences from Social Science Papers},
  year = {2024},
  doi = {10.31234/osf.io/kwtpm}
}
````
## License

This project is licensed under the GNU GPLv3, allowing for free use and modification with appropriate attribution.
