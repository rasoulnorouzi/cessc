# Academic Research Environment Documentation

This document outlines the technical environment and packages used for conducting this repo research, specifically focusing on converting PDF files to text and training models for causal sentence extraction, along with details on project management, writing, coding, and referencing tools.

## PDF to Text Conversion

### System Configuration
- **Operating System:** Windows
- **CPU:** Intel Core i5
- **RAM:** 8 GB

### Tools and Libraries
- **GROBID:** A machine learning library for extracting, parsing, and re-structuring raw data from scholarly documents into structured XML/TEI encoded documents with a particular focus on technical and scientific publications. For the conversion process, the full GROBID model, which is based on deep learning, was utilized.
- **Docker:** Used to run the GROBID model, ensuring an efficient and isolated environment for the process.

## Model Training for Causal Sentence Extraction

### System Configuration for Training
- **Platform:** Google Colab Pro
- **GPU:** NVIDIA A100
- **Graphic RAM:** 40 GB
- **System RAM:** 83.5 GB
- **Disk Space:** 201.2 GB

### Programming Environment
- **Python Version:** 3.11.5
### Key Python Packages
- `pandas`: 1.5.3
- `transformers`: 4.37.2
- `numpy`: 1.25.2
- `torch`: 2.1.0+cu121
- `sklearn`: 1.2.2
- `datasets`: 2.17.1
- `sentencepiece`: 0.1.99
- `peft`: 0.9.0
- `accelerate`: 0.27.2
- `evaluate`: 0.4.1
- `bitsandbytes`: 0.42.0

## Project Management and Collaboration Tools

### Communication and Management
- **Microsoft Teams:** Our team utilized Microsoft Teams for project management and communication, ensuring seamless collaboration and efficient management of the project's workflow.

### Writing and Documentation
- **Google Docs:** For drafting and sharing written content among team members.
- **Obsidian:** Used for organizing and linking our research notes and ideas in a rich knowledge base.
- **Zotero:** A referencing manager tool, which was crucial for managing bibliographies and references in our research project.

### Coding and Version Control
- **Hugging Face Hub:** Leveraged for accessing, storing, and collaborating on machine learning models.
- **GitHub:** Used for version control and code management, facilitating collaborative coding and project tracking.
### Objective
The objective of the model training phase was to fine-tune models for the extraction of causal sentences from the text. This phase utilized the computational resources and packages listed above to optimize performance and accuracy in sentence extraction tasks.

---

This documentation aims to provide a clear overview of the setup and tools used in the research process. It serves as a guide for replicating the environment or for future reference in similar research endeavors.
