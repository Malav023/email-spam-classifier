from fastapi import FastAPI 
from pydantic import BaseModel
import joblib
import pandas as pd

#ham = 0 spam =1 

app = FastAPI()
model = joblib.load(r"D:\Complete_proj\Email_spam\model\saved_models\spam_model_nb_best.joblib")

# 1. Define the data structure (Validation)
class EmailRequest(BaseModel):
    email: str

@app.post("/api/predict")
def predict(request: EmailRequest):
    # No need to manually extract from request.json; 
    # FastAPI gives you a clean object.
    # prediction = model.predict([request.email])[0] 
    # proba = model.predict_proba([request.email])[0]
    
    # # Get probabilities 
    
    prediction = model.predict(pd.Series([request.email]))[0] 
    proba = model.predict_proba(pd.Series([request.email]))[0]
    
    # HIGHLIGHT: FastAPI handles NumPy floats/arrays much better!
    return {
        "prediction": str(prediction),
        "confidence": round(float(max(proba)) * 100, 2)
    }

# Run with: uvicorn app:app --port 5001 --reload (locally)