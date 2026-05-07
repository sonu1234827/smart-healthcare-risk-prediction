from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    send_file
)

import os
import sqlite3
import requests
import re
import fitz

# OCR IMPORTS
import pytesseract
import cv2
import numpy as np

from pdf2image import convert_from_path
from PIL import Image

from werkzeug.utils import secure_filename

# ==========================================
# TESSERACT PATH
# ==========================================

pytesseract.pytesseract.tesseract_cmd = r"D:\myproject\tesseract.exe"

# ==========================================
# HYBRID PDF + OCR EXTRACTION
# ==========================================

def extract_text_from_pdf(filepath):

    text = ""

    # ==================================
    # NORMAL PDF EXTRACTION
    # ==================================

    try:

        doc = fitz.open(filepath)

        for page in doc:

            text += page.get_text()

    except Exception as e:

        print("PDF Extraction Error:", e)

    # ==================================
    # OCR IF TEXT NOT FOUND
    # ==================================

    if len(text.strip()) < 50:

        print("\n⚠ OCR MODE ACTIVATED...\n")

        try:

            images = convert_from_path(filepath)

            for img in images:

                img_cv = np.array(img)

                gray = cv2.cvtColor(
                    img_cv,
                    cv2.COLOR_BGR2GRAY
                )

                h, w = gray.shape

                if w < 1200:

                    scale = 1200 / w

                    gray = cv2.resize(
                        gray,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC
                    )

                gray = cv2.GaussianBlur(
                    gray,
                    (3, 3),
                    0
                )

                _, thresh = cv2.threshold(
                    gray,
                    180,
                    255,
                    cv2.THRESH_BINARY
                )

                ocr_text = pytesseract.image_to_string(
                    thresh,
                    config='--oem 3 --psm 6'
                )

                text += ocr_text

        except Exception as e:

            print("OCR Error:", e)

    print("\n========== FINAL EXTRACTED TEXT ==========\n")
    print(text)

    return text


# ==========================================
# PATIENT DETAILS EXTRACTION
# ==========================================

def extract_patient_details(text):

    name_match = re.search(
        r"(?:Name|Patient Name)\s*[:\-]?\s*([A-Za-z ]+)",
        text,
        re.IGNORECASE
    )

    age_match = re.search(
        r"(?:Age)\s*[:\-]?\s*(\d+)",
        text,
        re.IGNORECASE
    )

    gender_match = re.search(
        r"\b(Male|Female)\b",
        text,
        re.IGNORECASE
    )

    patient = {

        "name":
        name_match.group(1).strip()
        if name_match else "Unknown",

        "age":
        age_match.group(1)
        if age_match else "Unknown",

        "gender":
        gender_match.group(1)
        if gender_match else "Unknown"
    }

    return patient


# ==========================================
# SMART FEATURE EXTRACTION
# ==========================================

def smart_extract(text, disease):

    extracted = {}

    # ==================================
    # DIABETES FEATURES
    # ==================================

    diabetes_patterns = {

        "Glucose":
        r"(?:Glucose|Blood Glucose|Fasting Glucose|Sugar|FBS).*?(\d+(?:\.\d+)?)",

        "BMI":
        r"(?:BMI|Body Mass Index).*?(\d+(?:\.\d+)?)",

        "BloodPressure":
        r"(?:Blood Pressure|BP).*?(\d+(?:\.\d+)?)",

        "Insulin":
        r"(?:Insulin).*?(\d+(?:\.\d+)?)",

        "HbA1c":
        r"(?:HbA1c|A1C).*?(\d+(?:\.\d+)?)"
    }

    # ==================================
    # HEART FEATURES
    # ==================================

    heart_patterns = {

        "Cholesterol":
        r"(?:Total\s*)?Cholesterol.*?(\d+(?:\.\d+)?)",

        "HeartRate":
        r"(?:Heart\s*Rate|Pulse\s*Rate|Pulse|MaxHR).*?(\d+(?:\.\d+)?)",

        "BloodPressure":
        r"(?:Blood\s*Pressure|BP|RestingBP|Systolic).*?(\d+(?:\.\d+)?)",

        "Triglycerides":
        r"(?:Triglycerides?|TG).*?(\d+(?:\.\d+)?)",

        "HDL":
        r"(?:HDL(?:\s*Cholesterol)?).*?(\d+(?:\.\d+)?)",

        "LDL":
        r"(?:LDL(?:\s*Cholesterol)?).*?(\d+(?:\.\d+)?)",

        "VLDL":
        r"(?:VLDL(?:\s*Cholesterol)?).*?(\d+(?:\.\d+)?)"
    }

    # ==================================
    # KIDNEY FEATURES
    # ==================================

    kidney_patterns = {

        "Creatinine":
        r"(?:Creatinine|Serum Creatinine).*?(\d+(?:\.\d+)?)",

        "Urea":
        r"(?:Urea|Blood Urea).*?(\d+(?:\.\d+)?)",

        "Hemoglobin":
        r"(?:Hemoglobin|Hb).*?(\d+(?:\.\d+)?)",

        "UricAcid":
        r"(?:Uric Acid|UricAcid).*?(\d+(?:\.\d+)?)",

        "Sodium":
        r"(?:Sodium).*?(\d+(?:\.\d+)?)",

        "Potassium":
        r"(?:Potassium).*?(\d+(?:\.\d+)?)",

        "Calcium":
        r"(?:Calcium).*?(\d+(?:\.\d+)?)",

        "Albumin":
        r"(?:Albumin).*?(\d+(?:\.\d+)?)"
    }

    # ==================================
    # SELECT PATTERNS
    # ==================================

    if disease == "diabetes":
        patterns = diabetes_patterns

    elif disease == "heart":
        patterns = heart_patterns

    elif disease == "kidney":
        patterns = kidney_patterns

    else:
        patterns = {}

    # ==================================
    # EXTRACT VALUES
    # ==================================

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:

            try:

                extracted[key] = float(
                    match.group(1)
                )

            except:

                extracted[key] = match.group(1)

    print("\n========== EXTRACTED ==========\n")
    print(extracted)

    return extracted


# ==========================================
# REPORT VALIDATION
# ==========================================

def validate_report(text, disease):

    text = text.lower()

    diabetes_keywords = [
        "glucose",
        "insulin",
        "hba1c",
        "blood sugar"
    ]

    heart_keywords = [
        "cholesterol",
        "hdl",
        "ldl",
        "triglycerides",
        "pulse",
        "maxhr",
        "restingbp"
    ]

    kidney_keywords = [
        "creatinine",
        "urea",
        "uric acid",
        "potassium",
        "sodium"
    ]

    if disease == "diabetes":

        return any(
            keyword in text
            for keyword in diabetes_keywords
        )

    elif disease == "heart":

        return any(
            keyword in text
            for keyword in heart_keywords
        )

    elif disease == "kidney":

        return any(
            keyword in text
            for keyword in kidney_keywords
        )

    return False


# ==========================================
# APP SETUP
# ==========================================

app = Flask(__name__)

app.secret_key = "healthcare_secret"

API_URL = "http://127.0.0.1:5000/predict"

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ==========================================
# DATABASE
# ==========================================

def init_db():

    conn = sqlite3.connect("users.db")

    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        disease TEXT,
        probability REAL,
        prediction INTEGER
    )
    """)

    conn.commit()
    conn.close()


init_db()


# ==========================================
# USER FUNCTIONS
# ==========================================

def register_user(username, password):

    conn = sqlite3.connect("users.db")

    c = conn.cursor()

    try:

        c.execute(
            "INSERT INTO users(username,password) VALUES(?,?)",
            (username, password)
        )

        conn.commit()

        return True

    except:
        return False

    finally:
        conn.close()


def login_user(username, password):

    conn = sqlite3.connect("users.db")

    c = conn.cursor()

    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = c.fetchone()

    conn.close()

    return user


# ==========================================
# LOGIN
# ==========================================

@app.route("/", methods=["GET", "POST"])
def login():

    error = None

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        user = login_user(username, password)

        if user:

            session["user"] = username

            return redirect("/dashboard")

        else:
            error = "Invalid username or password"

    return render_template(
        "login.html",
        error=error
    )


# ==========================================
# REGISTER
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    error = None
    success = None

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if register_user(username, password):

            success = "Registration successful! Please login."

        else:
            error = "User already exists"

    return render_template(
        "register.html",
        error=error,
        success=success
    )


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/")

    return render_template(
        "dashboard.html",
        username=session["user"]
    )


# ==========================================
# UPLOAD PAGE
# ==========================================

@app.route("/upload")
def upload():

    if "user" not in session:
        return redirect("/")

    return render_template("upload.html")


# ==========================================
# PREVIEW PAGE
# ==========================================

@app.route("/preview", methods=["POST"])
def preview():

    if "user" not in session:
        return redirect("/")

    file = request.files["file"]

    disease = request.form["disease"]

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(filepath)

    text = extract_text_from_pdf(filepath)

    valid_report = validate_report(
        text,
        disease
    )

    if not valid_report:

        return render_template(
            "upload.html",
            error=f"Wrong report uploaded for {disease.title()} disease."
        )

    patient = extract_patient_details(text)

    extracted = smart_extract(text, disease)

    if not extracted:

        extracted = {
            "Message": "No parameters detected"
        }

    return render_template(
        "preview.html",
        patient=patient,
        extracted=extracted,
        disease=disease
    )


# ==========================================
# ANALYZE
# ==========================================

@app.route("/analyze", methods=["POST"])
def analyze():

    if "user" not in session:
        return redirect("/")

    disease = request.form["disease"]

    extracted = {}

    for key in request.form:

        if key not in [
            "disease",
            "name",
            "age",
            "gender"
        ]:

            try:

                extracted[key] = float(
                    request.form[key]
                )

            except:

                extracted[key] = request.form[key]

    patient = {

        "name":
        request.form["name"],

        "age":
        request.form["age"],

        "gender":
        request.form["gender"]
    }

    try:

        response = requests.post(
            f"{API_URL}/{disease}",
            json=extracted
        )

        result = response.json()

    except Exception as e:

        result = {

            "prediction": 0,

            "probability": 0,

            "shap_values": {},

            "error": str(e)
        }

    # ==================================
    # GET SHAP VALUES
    # ==================================

    shap_values = result.get(
        "shap_values",
        {}
    )

    print("\n========== SHAP VALUES ==========\n")
    print(shap_values)

    # ==================================
    # SAVE HISTORY
    # ==================================

    conn = sqlite3.connect("users.db")

    c = conn.cursor()

    c.execute("""
    INSERT INTO history(
        username,
        disease,
        probability,
        prediction
    )
    VALUES(?,?,?,?)
    """, (
        session["user"],
        disease,
        result.get("probability", 0),
        result.get("prediction", 0)
    ))

    conn.commit()
    conn.close()

    # ==================================
    # FEATURE EXPLANATIONS
    # ==================================

    explanations = []

    for k, v in shap_values.items():

        try:

            val = float(v)

            if val > 0:

                explanations.append(
                    f"{k} increased the disease risk."
                )

            elif val < 0:

                explanations.append(
                    f"{k} reduced the disease risk."
                )

        except:
            pass

    # ==================================
    # SAVE REPORT SESSION
    # ==================================

    session["report_data"] = {

        "patient": patient,

        "disease": disease,

        "prediction":
        result.get("prediction", 0),

        "probability":
        result.get("probability", 0),

        "extracted": extracted,

        "shap_values": shap_values,

        "explanations": explanations
    }

    return render_template(

        "result.html",

        patient=patient,

        extracted=extracted,

        result=result,

        shap_values=shap_values,

        explanations=explanations
    )


# ==========================================
# HISTORY
# ==========================================

@app.route("/history")
def history():

    if "user" not in session:
        return redirect("/")

    conn = sqlite3.connect("users.db")

    c = conn.cursor()

    c.execute("""
    SELECT disease, probability, prediction
    FROM history
    WHERE username=?
    """, (session["user"],))

    rows = c.fetchall()

    conn.close()

    return render_template(
        "history.html",
        history=rows
    )


# ==========================================
# DOWNLOAD REPORT
# ==========================================

@app.route("/download")
def download():

    if "report_data" not in session:

        return redirect("/dashboard")

    report = session["report_data"]

    filepath = os.path.join(
        BASE_DIR,
        "report.txt"
    )

    with open(filepath, "w") as f:

        f.write("SMART HEALTHCARE REPORT\n\n")

        f.write(
            f"Patient Name : "
            f"{report['patient']['name']}\n"
        )

        f.write(
            f"Age : "
            f"{report['patient']['age']}\n"
        )

        f.write(
            f"Gender : "
            f"{report['patient']['gender']}\n\n"
        )

        f.write(
            f"Disease : "
            f"{report['disease']}\n"
        )

        risk = (
            "High Risk"
            if report["prediction"] == 1
            else "Low Risk"
        )

        f.write(
            f"Prediction : {risk}\n"
        )

        f.write(
            f"Probability : "
            f"{round(report['probability'] * 100, 2)}%\n\n"
        )

        f.write("FEATURES\n")

        for k, v in report["extracted"].items():

            f.write(f"{k} : {v}\n")

        f.write("\nSHAP FEATURE CONTRIBUTIONS\n")

        for k, v in report["shap_values"].items():

            f.write(f"{k} : {v}\n")

        f.write("\nAI EXPLANATIONS\n")

        for item in report["explanations"]:

            f.write(f"- {item}\n")

    return send_file(
        filepath,
        as_attachment=True
    )


# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=8000
    )