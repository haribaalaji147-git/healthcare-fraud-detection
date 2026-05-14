from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="Healthcare Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("models/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

class Claim(BaseModel):
    patient_age: int
    hospital: str
    procedure: str
    diagnosis: str
    claim_amount: float
    days_in_hospital: int
    claims_in_month: int

@app.get("/")
def root():
    return {"message": "Healthcare Fraud Detection API is running"}

@app.post("/predict")
def predict(claim: Claim):
    try:
        hospital_enc = encoders["hospital"].transform([claim.hospital])[0]
        procedure_enc = encoders["procedure"].transform([claim.procedure])[0]
        diagnosis_enc = encoders["diagnosis"].transform([claim.diagnosis])[0]
    except:
        return {"error": "Unknown hospital, procedure, or diagnosis value"}

    features = np.array([[
        claim.patient_age,
        claim.claim_amount,
        claim.days_in_hospital,
        claim.claims_in_month,
        hospital_enc,
        procedure_enc,
        diagnosis_enc,
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "claim_amount": claim.claim_amount,
        "fraud_detected": bool(prediction),
        "fraud_probability": round(float(probability) * 100, 2),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW",
        "decision": "HOLD FOR REVIEW" if prediction else "AUTO APPROVE",
    }

@app.get("/claims/recent")
def recent_claims():
    df = pd.read_csv("data/claims.csv")
    sample = df.sample(10).to_dict(orient="records")
    return {"claims": sample}