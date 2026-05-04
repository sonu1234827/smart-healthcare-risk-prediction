# src/models/predict.py

import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/diabetes_model.pkl")

def predict_diabetes(input_data):
    """
    input_data should be a list in this format:
    [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    """

    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data], columns=columns)

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability


if __name__ == "__main__":
    # Example input
    sample_input = [2, 120, 70, 20, 85, 28.5, 0.5, 35]

    pred, prob = predict_diabetes(sample_input)

    if pred == 1:
        print(f"Diabetes Risk: High ({prob*100:.2f}%)")
    else:
        print(f"Diabetes Risk: Low ({prob*100:.2f}%)")