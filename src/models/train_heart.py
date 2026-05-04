# src/models/train_heart.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


def train_heart_model():
    df = pd.read_csv("data/processed/heart_clean.csv")

    # Features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train.to_csv("models/heart_background.csv", index=False)
    print("Training heart models...\n")

    # ----------------------------
    # 1. Logistic Regression (FIXED)
    # ----------------------------
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])

    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    print(f"Logistic Regression: {lr_acc:.4f}")

    # ----------------------------
    # 2. Random Forest
    # ----------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest: {rf_acc:.4f}")

    # ----------------------------
    # 3. XGBoost (FIXED)
    # ----------------------------
    xgb = XGBClassifier(eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    print(f"XGBoost: {xgb_acc:.4f}")

    # ----------------------------
    # Select Best Model
    # ----------------------------
    models = {
        "LR": (lr, lr_acc),
        "RF": (rf, rf_acc),
        "XGB": (xgb, xgb_acc),
    }

    best_name = max(models, key=lambda k: models[k][1])
    best_model = models[best_name][0]
    best_acc = models[best_name][1]

    print(f"\nBest Model: {best_name} (Accuracy: {best_acc:.4f})")

    # Save model
    joblib.dump(best_model, "models/heart_model.pkl")

    print("Heart model saved successfully!")


if __name__ == "__main__":
    train_heart_model()