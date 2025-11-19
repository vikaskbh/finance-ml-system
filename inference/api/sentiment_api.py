from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from inference.model.load_model import predict_sentiment
app = FastAPI()

origins = [
    "https://capricornai.dev",
    "https://www.capricornai.dev",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Finance ML Inference API"}

@app.post("/sentiment")
def sentiment(q: Query):
    result = predict_sentiment(q.text)
    return {
        "label": result['label'],
        "score": float(result['score'])
    }

#uvicorn inference.api.sentiment_api:app --host 127.0.0.1 --port 8001 --reload
