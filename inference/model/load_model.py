from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)

def predict_sentiment(text: str):
    return sentiment_pipe(text)[0]
