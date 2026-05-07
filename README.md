# 🩺 Smart Healthcare Risk Prediction System

An AI-powered web application that analyzes medical reports (PDF) and predicts disease risk using Machine Learning with explainability (SHAP).

---

## 🚀 Features

- 📄 Upload medical report (PDF)
- 🧠 Automatic extraction of:
  - Patient Name
  - Age
  - Gender
- 🏥 Disease detection:
  - Diabetes
  - Heart Disease
  - Kidney Disease
- 📊 Risk Prediction (Low / High)
- 🔍 Explainable AI (SHAP feature contribution graph)
- 📈 Interactive Dashboard
- 🕒 Prediction History
- 🔐 Login & Register system

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Machine Learning (Scikit-learn)
- SHAP (Explainable AI)
- SQLite (Database)
- Plotly (Dashboard visualization)

---

## 📂 Project Structure
Smart Healthcare Risk Prediction System/
│
├── app/
│   └── app.py
│
├── models/
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── kidney_model.pkl
│
├── src/
│   └── utils/
│       ├── database.py
│       ├── pdf_parser.py
│
├── requirements.txt
├── README.md
└── database.db
## Installation
pip install -r requirements.txt
## Run the Project
cd frontend
python app.py 
