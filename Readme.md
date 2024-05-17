# Sentiment Analysis with BERT

## Overview
This project aims to perform sentiment analysis on text data using the BERT (Bidirectional Encoder Representations from Transformers) model. Sentiment analysis involves determining the sentiment expressed in a piece of text, whether it's positive, negative, or neutral. The BERT model, pre-trained on a large corpus of text, is fine-tuned on a specific sentiment analysis dataset to perform this task.

## Requirements
- Python 3
- Google Colab (or a similar platform with GPU support)
- PyTorch
- Transformers library by Hugging Face
- Other necessary libraries as listed in the provided code

## Dataset
The sentiment analysis dataset used in this project is located at `/content/sentiment_beto_withlabels.csv`. This dataset contains text data along with their corresponding sentiment labels (positive, negative, neutral).

## Model Training
The model training process involves the following steps:
1. Data Preprocessing: The text data is preprocessed, including cleaning, tokenization, and encoding.
2. Model Configuration: The pre-trained BERT model is loaded and configured for sequence classification.
3. Fine-tuning: The model is fine-tuned on the sentiment analysis dataset using techniques like AdamW optimizer and linear scheduler.
4. Evaluation: The trained model is evaluated on a validation dataset to assess its performance.

## Usage
To train and evaluate the sentiment analysis model, follow these steps:
1. Open the provided Google Colab notebook.
2. Execute the code cells in sequence to preprocess data, train the model, and evaluate its performance.
3. Optionally, you can make modifications to the code to experiment with different hyperparameters, architectures, or datasets.

## Saving the Model
Once the model is trained and evaluated satisfactorily, you can save it for future use. The model and tokenizer are saved using the `save_pretrained` method from the Transformers library.

## Downloading the Model
To download the saved model, follow these steps:
1. Navigate to the directory where the model is saved.
2. Zip the model directory.
3. Download the zip file by clicking on the provided link.

## Author
Dhaksin.S

## License
This project is licensed under the [MIT License]([https://github.com/Dhaksin53/-IREX-El-Salvador-Sentiment_analysis/blob/main/MIT%20License)]).
