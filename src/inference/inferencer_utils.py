import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


def load_tokenizer(model_name: str = None):
    if not model_name:
        model_name = config.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


def load_model(model_name: str = None):
    if not model_name:
        model_name = config.MODEL_NAME
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config.NUM_LABEL)
    return model


def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True, max_length=64, padding="max_length")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average='weighted'),
        "recall": recall_score(labels, predictions, average='weighted'),
        "f1": f1_score(labels, predictions, average='weighted')
    }


def predict_sentiment(texts, pipe):
    outputs = []
    results = pipe(texts, batch_size=8)

    for text, scores in zip(texts, results):
        predicted_entry = max(scores, key=lambda x: x["score"])
        predicted_label = predicted_entry["label"]

        bullish_score = next((entry["score"] for entry in scores if entry["label"].lower() == "bullish"), 0.0)
        bearish_score = next((entry["score"] for entry in scores if entry["label"].lower() == "bearish"), 0.0)

        sentiment_index = bullish_score - bearish_score

        outputs.append({
            "predicted_label": predicted_label,
            "sentiment_index": sentiment_index
        })
    return outputs

