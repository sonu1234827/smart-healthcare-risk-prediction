# src/models/train_kidney.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_kidney_model():
    df = pd.read_csv("data/processed/kidney_clean.csv")

    # Target column (IMPORTANT)
    X = df.drop("classification", axis=1)
    y = df["classification"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train.to_csv("models/kidney_background.csv", index=False)

    print("Training kidney models...\n")

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    print(f"Logistic Regression: {lr_acc:.4f}")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest: {rf_acc:.4f}")

    # XGBoost
    xgb = XGBClassifier(eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    print(f"XGBoost: {xgb_acc:.4f}")

    # Select best
    models = {
        "LR": (lr, lr_acc),
        "RF": (rf, rf_acc),
        "XGB": (xgb, xgb_acc),
    }

    best_name = max(models, key=lambda k: models[k][1])
    best_model = models[best_name][0]

    print(f"\nBest Model: {best_name}")

    joblib.dump(best_model, "models/kidney_model.pkl")

    print("Kidney model saved!")


if __name__ == "__main__":
    train_kidney_model()