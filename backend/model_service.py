import pandas as pd
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models")

models = {
    "diabetes": joblib.load(os.path.join(MODEL_PATH, "diabetes_model.pkl")),
    "heart": joblib.load(os.path.join(MODEL_PATH, "heart_model.pkl")),
    "kidney": joblib.load(os.path.join(MODEL_PATH, "kidney_model.pkl"))
}

def predict_disease(disease, data_dict):
    model = models.get(disease.lower())

    if model is None:
        return {"error": "Invalid disease"}

    df = pd.DataFrame([data_dict])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }