# src/models/predict_heart.py

import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/heart_model.pkl")

def predict_heart(input_data):
    """
    input_data format (IMPORTANT - order must match dataset):

    [age, sex, cp, trestbps, chol, fbs, restecg,
     thalach, exang, oldpeak, slope, ca, thal]
    """

    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data], columns=columns)

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability


if __name__ == "__main__":
    # Sample input (you can change values)
    sample_input = [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]

    pred, prob = predict_heart(sample_input)

    if pred == 1:
        print(f"Heart Disease Risk: High ({prob*100:.2f}%)")
    else:
        print(f"Heart Disease Risk: Low ({prob*100:.2f}%)")