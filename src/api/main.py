from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import logging

# Initialize
app = FastAPI(title="Credit Risk API", description="Predicts likelihood of default based on RFM proxy.")
logging.basicConfig(level=logging.INFO)

# Pydantic Model
class CreditScoringRequest(BaseModel):
    TotalAmount: float
    AvgAmount: float
    StdAmount: float
    TxCount: int
    TotalValue: float
    AvgValue: float
    ProductCategory: int
    ChannelId: int
    PricingStrategy: int

class CreditScoringResponse(BaseModel):
    risk_probability: float
    is_high_risk: int

# Load Model Globally
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
try:
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model from {model_path}: {e}")
    model = None

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model is not None}

@app.post("/predict", response_model=CreditScoringResponse)
def predict(request: CreditScoringRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.dict()])
        
        # Predict
        prob = model.predict_proba(data)[:, 1][0]
        pred = model.predict(data)[0]
        
        return {
            "risk_probability": float(prob),
            "is_high_risk": int(pred)
        }
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))