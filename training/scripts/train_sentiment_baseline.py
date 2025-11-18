import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

def load_data():
    # Day 2 will fill this with a real finance dataset
    ds = load_dataset("financial_phrasebank", "sentences_75agree")
    return ds

def build_model(model_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    return tokenizer, model

def tokenize(batch, tokenizer):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

def main():
    tokenizer, model = build_model()
    ds = load_data()

    # Placeholder – tomorrow you’ll implement tokenization, training, metrics.
    print("Dataset loaded:", ds)
    print("Model loaded")

if __name__ == "__main__":
    main()
