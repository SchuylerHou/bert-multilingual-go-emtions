
# Multilingual (English and Chinese) GoEmotions Classification Model

This repository hosts a fine-tuned BERT model for cross-language emotion classification on the GoEmotions dataset. This model is unique as it has been trained on a multilingual dataset comprising of English and Chinese texts. It is capable of classifying text into one of 28 different emotion categories.

The 28 emotion categories, according to the GoEmotions taxonomy, are: 'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', and 'neutral'.

# Model Performance
The model demonstrates high performance on the validation set, with the following scores:

Accuracy: 85.95%
Precision: 91.99%
Recall: 89.56%
F1 Score: 90.17%
These results indicate the model's high accuracy and precision in predicting the correct emotion category for a given input text, regardless of the language (English or Chinese).

## Training data

The dataset used for training the model is a combined dataset of the original English GoEmotions dataset and a machine translated Chinese version of the GoEmotions dataset.

The dataset is split into two parts:

- **Labeled data**: Used for initial training. It includes both English and machine translated Chinese samples. This labeled data is further split into a training set (80%) and a validation set (20%).
- **Unlabeled data**: Used for making predictions and adding confidently predicted samples to the training data. It includes both English and machine translated Chinese samples.

## Training

The model is trained in two stages:

1. Initial training on the labeled data.
2. Predictions are made on the unlabeled data, and the most confidently predicted samples are added to the training data. The model is then retrained on this updated labeled data.

The model is trained for a total of 20 epochs (10 epochs for each stage). Precision, recall, and F1 score are logged during training.

## Usage

Here is a code snippet showing how to use this model:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("SchuylerH/bert-multilingual-go-emtions")
model = AutoModelForSequenceClassification.from_pretrained("SchuylerH/bert-multilingual-go-emtions")
text = "I love you."
nlp = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer)
result = nlp(text)
print(result)
