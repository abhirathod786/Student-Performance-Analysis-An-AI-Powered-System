"""
FastAPI Backend for Student Performance System
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="Student Performance API")

# Load models
with open('../models/graduation_model.pkl','rb') as f: grad_model = pickle.load(f)
with open('../models/risk_model.pkl','rb') as f: risk_model = pickle.load(f)
with open('../models/feature_names.pkl','rb') as f: features = pickle.load(f)

class StudentData(BaseModel):
    overall_cgpa: float
    overall_attendance: float
    current_backlogs: int
    internships_completed: int
    coding_test_score: float

@app.get("/")
def read_root():
    return {"message": "Student Performance API", "status": "active"}

@app.post("/predict")
def predict(data: StudentData):
    X = np.zeros(len(features))
    X[0] = data.overall_cgpa
    X[1] = data.overall_attendance
    X[2] = data.current_backlogs
    
    risk = risk_model.predict(X.reshape(1,-1))[0]
    
    return {
        "risk_score": float(risk),
        "status": "Critical" if risk>70 else "High" if risk>50 else "Medium" if risk>30 else "Low"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}
