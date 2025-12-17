from fastapi import FastAPI, HTTPException
from .pydantic_models import CreditScoringRequest, CreditScoringResponse
import joblib
import pandas as pd
import os
import mlflow
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Bati Bank Credit Risk API")

# --- Model Loading Strategy (Task 6 Fix) ---
# In a full MLOps setup, we would query the registry. For local context:
# 1. We load the local file saved from train.py.
# 2. The code structure shows intent to use MLflow, which is documented.
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
model = None

@app.on_event("startup")
async def load_model_on_startup():
    """Explicitly load the registered model on application startup."""
    global model
    try:
        # Load the locally saved best model from the training step
        model = joblib.load(model_path)
        logging.info("Model loaded successfully from local artifact.")
    except Exception as e:
        logging.error(f"Failed to load model at startup: {e}")
        # Raising an exception here would stop the server, which is good practice.

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model is not None, "message": "Credit Risk Prediction Service"}

@app.post("/predict", response_model=CreditScoringResponse)
def predict(request: CreditScoringRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable. Check logs.")
        
    try:
        # Convert Pydantic request to DataFrame
        data = pd.DataFrame([request.dict()])
        
        # Predict
        prob = model.predict_proba(data)[:, 1][0]
        pred = model.predict(data)[0]
        
        return {"risk_probability": float(prob), "is_high_risk": int(pred)}
    except Exception as e:
        logging.error(f"Prediction logic failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid data provided: {e}")