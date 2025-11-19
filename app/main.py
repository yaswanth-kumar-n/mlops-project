from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# THIS is the ASGI app uvicorn expects
app = FastAPI(title="MLOps Survival API")

MODEL_PATH = Path("data/models/model.pkl")

# Load model at startup if it exists
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class PredictRequest(BaseModel):
    features: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"error": "Model not loaded"}
    df = pd.DataFrame([req.features])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}