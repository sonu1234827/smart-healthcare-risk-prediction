# src/data_preprocessing/preprocess_kidney.py

import pandas as pd

def preprocess_kidney():
    df = pd.read_csv("data/raw/kidney_disease.csv")

    print("Original Shape:", df.shape)

    # Drop unnecessary columns (like id if present)
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    # Replace common string values with numeric
    df.replace({
        "yes": 1, "no": 0,
        "present": 1, "notpresent": 0,
        "abnormal": 1, "normal": 0,
        "good": 1, "poor": 0,
        "ckd": 1, "notckd": 0
    }, inplace=True)

    # Convert all to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill missing values
    df.fillna(df.median(), inplace=True)

    print("\nRemaining Missing Values:\n", df.isnull().sum())

    # Save processed data
    df.to_csv("data/processed/kidney_clean.csv", index=False)

    print("\nKidney data processed successfully!")


if __name__ == "__main__":
    preprocess_kidney()