# src/data_preprocessing/preprocess_diabetes.py

import pandas as pd

def preprocess_diabetes():
    # Load raw data
    df = pd.read_csv("data/raw/diabetes.csv")

    print("Original Data Shape:", df.shape)

    # Replace 0 values with NaN for specific columns
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zero] = df[cols_with_zero].replace(0, pd.NA)

    # Convert all columns to numeric (important safety step)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill missing values with median
    df.fillna(df.median(), inplace=True)

    # Final check for missing values
    print("\nRemaining Missing Values:\n", df.isnull().sum())

    # Save processed data
    df.to_csv("data/processed/diabetes_clean.csv", index=False)

    print("\nProcessed data saved successfully!")


if __name__ == "__main__":
    preprocess_diabetes()