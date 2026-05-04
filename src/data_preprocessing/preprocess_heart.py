# src/data_preprocessing/preprocess_heart.py

import pandas as pd

def preprocess_heart():
    # Load raw data
    df = pd.read_csv("data/raw/heart.csv")

    print("Original Data Shape:", df.shape)

    # Convert all columns to numeric (safety)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill missing values with median
    df.fillna(df.median(), inplace=True)

    # Final check
    print("\nRemaining Missing Values:\n", df.isnull().sum())

    # Save processed data
    df.to_csv("data/processed/heart_clean.csv", index=False)

    print("\nProcessed heart data saved successfully!")


if __name__ == "__main__":
    preprocess_heart()