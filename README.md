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
