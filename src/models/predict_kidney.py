# src/models/predict_kidney.py

import joblib
import pandas as pd

model = joblib.load("models/kidney_model.pkl")

def predict_kidney(input_data, columns):
    input_df = pd.DataFrame([input_data], columns=columns)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    return pred, prob


if __name__ == "__main__":
    # ⚠️ Replace columns with your dataset columns (except target)
    columns = list(pd.read_csv("data/processed/kidney_clean.csv").drop("classification", axis=1).columns)

    sample_input = [0]*len(columns)  # dummy input

    pred, prob = predict_kidney(sample_input, columns)

    if pred == 1:
        print(f"Kidney Disease Risk: High ({prob*100:.2f}%)")
    else:
        print(f"Kidney Disease Risk: Low ({prob*100:.2f}%)")