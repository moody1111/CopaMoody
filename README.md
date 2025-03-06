# Copamoody Model

This repository contains the main code for the Copamoody model, a Transformer-based model for natural language processing tasks.

Authors ::Moody - Moody.eg@msn.com
## Description

The Copamoody model utilizes TensorFlow and Hugging Face's Transformers library to tokenize datasets, train a Transformer model, and save the trained model weights. The datasets are loaded from various directories, combined, and split into training, validation, and test sets.

## Abilities

- **Load Datasets**: Load text files from specified directories and combine them into a single dataset.
- **Tokenize Data**: Use the BERT tokenizer to tokenize the text data, preparing it for model training.
- **Dataset Splitting**: Split the combined dataset into training, validation, and test sets.
- **Transformer Model**: Define and train a Transformer model using TensorFlow with specified hyperparameters.
- **Save Model**: Save the trained model weights to disk.
- **Tokenized Dataset Saving**: Save tokenized datasets to disk for future use.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- TensorFlow
- Transformers
- Datasets

### Installation
1. Clone this repository:
```bash
git clone https://github.com/Moody1111/copamoody.git

#Usage

cache_dir = "CACHE_DIRECTORY_PLACEHOLDER"
data_dir_e = "DATA_DIRECTORY_PLACEHOLDER"
new_data_dir = "NEW_DATA_DIRECTORY_PLACEHOLDER"
dataset_dir_d = "DATASET_DIR_D_PLACEHOLDER"
chatbot_dataset_path = "CHATBOT_DATASET_PATH_PLACEHOLDER"
Dependencies
TensorFlow: For defining and training the Transformer model.
Transformers: For using the BERT tokenizer.
Datasets: For handling and processing datasets.

#License
This project is licensed under the Free License for Non-Commercial Use. You are free to use, modify, and distribute this code for personal and educational purposes. Commercial use is strictly prohibited without prior permission.
