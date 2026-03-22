from fastapi import FastAPI 
from pydantic import BaseModel
import joblib
import pandas as pd

#ham = 0 spam =1 

app = FastAPI()
import os
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "spam_model_nb_best.joblib")
model = joblib.load(model_path)

# 1. Define the data structure (Validation)
class EmailRequest(BaseModel):
    email: str

@app.post("/api/predict")
def predict(request: EmailRequest):
    # Get probabilities     
    prediction = model.predict(pd.Series([request.email]))[0] 
    proba = model.predict_proba(pd.Series([request.email]))[0]    
    
    return {
        "prediction": str(prediction),
        "confidence": round(float(max(proba)) * 100, 2)
    }

# Run with: uvicorn app:app --port 5001 --reload (locally)