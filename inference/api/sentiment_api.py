from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Finance ML Inference API"}

@app.post("/sentiment")
def sentiment(q: Query):
    # Tomorrow/Day 3 will load a trained model.
    return {"label": "pending", "score": 0.0}
