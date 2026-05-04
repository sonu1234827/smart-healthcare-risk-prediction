# src/models/train_diabetes.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_diabetes_model():
    # Load processed data
    df = pd.read_csv("data/processed/diabetes_clean.csv")

    # Split features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train.to_csv("models/diabetes_background.csv", index=False)
    print("Training models...\n")

    # ----------------------------
    # 1. Logistic Regression
    # ----------------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)

    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    # ----------------------------
    # 2. Random Forest
    # ----------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # ----------------------------
    # 3. XGBoost
    # ----------------------------
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)

    print(f"XGBoost Accuracy: {xgb_acc:.4f}")

    # ----------------------------
    # Select Best Model
    # ----------------------------
    models = {
        "Logistic Regression": (lr, lr_acc),
        "Random Forest": (rf, rf_acc),
        "XGBoost": (xgb, xgb_acc),
    }

    best_model_name = max(models, key=lambda k: models[k][1])
    best_model, best_acc = models[best_model_name]

    print(f"\nBest Model: {best_model_name} (Accuracy: {best_acc:.4f})")

    # Save best model
    joblib.dump(best_model, "models/diabetes_model.pkl")

    print("Model saved successfully!")


if __name__ == "__main__":
    train_diabetes_model()