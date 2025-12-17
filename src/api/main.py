from fastapi import FastAPI, HTTPException
from .pydantic_models import CreditScoringRequest, CreditScoringResponse
import joblib
import pandas as pd
import os
import mlflow
import logging

logging.basicConfig(level=logging.INFO)

# 1. Initialize FastAPI App (Only once)
app = FastAPI(title="Bati Bank Credit Risk API", 
              description="Predicts likelihood of default based on RFM proxy.")

# 2. Define Global Model Path
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
model = None

# 3. Model Loading on Startup (Task 6 Requirement)
@app.on_event("startup")
async def load_model_on_startup():
    """Explicitly loads the model artifact when the application starts."""
    global model
    try:
        # Load the locally saved best model from the training step
        model = joblib.load(model_path)
        logging.info("Model loaded successfully from local artifact.")
    except Exception as e:
        logging.error(f"Failed to load model at startup from {model_path}: {e}")
        # Server starts, but /predict will fail

# 4. Health Check Endpoint
@app.get("/")
def health_check():
    """Checks if the service is running and the model is loaded."""
    return {"status": "active", 
            "model_loaded": model is not None, 
            "message": "Credit Risk Prediction Service"}

# 5. Prediction Endpoint
@app.post("/predict", response_model=CreditScoringResponse)
def predict(request: CreditScoringRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable. Check server logs.")
        
    try:
        # Convert Pydantic request to DataFrame
        data = pd.DataFrame([request.dict()])
        
        # Predict Probability and Class
        prob = model.predict_proba(data)[:, 1][0]
        pred = model.predict(data)[0]
        
        return {"risk_probability": float(prob), 
                "is_high_risk": int(pred)}
    except Exception as e:
        logging.error(f"Prediction logic failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid data provided or prediction error: {e}")