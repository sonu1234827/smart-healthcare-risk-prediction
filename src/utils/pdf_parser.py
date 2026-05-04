# src/utils/pdf_parser.py

import re
import PyPDF2

# -------------------------
# READ PDF TEXT
# -------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.lower()


# -------------------------
# GENERIC VALUE EXTRACTOR
# -------------------------
def extract_value(text, keywords):
    for key in keywords:
        pattern = rf"{key}\s*[:\-]?\s*(\d+\.?\d*)"
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return 0


# -------------------------
# DIABETES VALUES
# -------------------------
def extract_diabetes_values(text):
    return {
        "Pregnancies": 0,
        "Glucose": extract_value(text, ["glucose", "blood sugar"]),
        "BloodPressure": extract_value(text, ["blood pressure"]),
        "SkinThickness": extract_value(text, ["skin thickness"]),
        "Insulin": extract_value(text, ["insulin"]),
        "BMI": extract_value(text, ["bmi"]),
        "DiabetesPedigreeFunction": 0.5,
        "Age": extract_value(text, ["age"])
    }


# -------------------------
# HEART VALUES
# -------------------------
def extract_heart_values(text):
    return {
        "age": extract_value(text, ["age"]),
        "sex": 1,
        "cp": 0,
        "trestbps": extract_value(text, ["blood pressure"]),
        "chol": extract_value(text, ["cholesterol"]),
        "fbs": 0,
        "restecg": 0,
        "thalach": extract_value(text, ["heart rate"]),
        "exang": 0,
        "oldpeak": 0,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }


# -------------------------
# KIDNEY VALUES
# -------------------------
def extract_kidney_values(text):
    return {
        "age": extract_value(text, ["age"]),
        "bp": extract_value(text, ["blood pressure"]),
        "sg": 1.02,
        "al": 1,
        "su": 0,
        "bgr": extract_value(text, ["glucose"]),
        "bu": extract_value(text, ["urea"]),
        "sc": extract_value(text, ["creatinine"]),
        "sod": 135,
        "pot": 4.5,
        "hemo": extract_value(text, ["hemoglobin"]),
        "pcv": 40,
        "wc": 8000,
        "rc": 5,
        "htn": 0,
        "dm": 0,
        "cad": 0,
        "appet": 1,
        "pe": 0,
        "ane": 0
    }