import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import sys, os, re
import plotly.express as px

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.utils.pdf_parser import *
from src.utils.database import *

st.set_page_config(page_title="Smart Healthcare AI", layout="wide")
init_db()

# =========================
# PATIENT INFO EXTRACTION
# =========================
def extract_name(text):
    match = re.search(r"(name|patient name)[:\- ]+([a-zA-Z ]+)", text, re.IGNORECASE)
    return match.group(2).strip().title() if match else "Unknown"

def extract_age(text):
    match = re.search(r"age[:\- ]+(\d+)", text, re.IGNORECASE)
    return match.group(1) if match else "Unknown"

def extract_gender(text):
    text = text.lower()
    if "female" in text:
        return "Female"
    elif "male" in text:
        return "Male"
    return "Unknown"

# =========================
# EXTRACTION
# =========================
def extract_value(text, keywords):
    for k in keywords:
        pattern = rf"{k}[^0-9]*(\d+\.?\d*)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

def extract_bp(text):
    match = re.search(r"(\d{2,3})/(\d{2,3})", text)
    if match:
        return int(match.group(1))
    return None

def smart_extract(text, disease, fallback_fn):
    text = text.lower()

    if disease == "Diabetes":
        vals = {
            "Glucose": extract_value(text, ["glucose", "blood sugar", "fasting sugar"]),
            "BMI": extract_value(text, ["bmi"]),
            "Insulin": extract_value(text, ["insulin"]),
            "BloodPressure": extract_bp(text),
            "SkinThickness": extract_value(text, ["skin thickness"]),
            "Age": extract_value(text, ["age"]),
        }

    elif disease == "Heart":
        vals = {
            "chol": extract_value(text, ["cholesterol", "total cholesterol", "chol"]),
            "thalach": extract_value(text, ["heart rate", "pulse", "pulse rate"]),
            "trestbps": extract_bp(text) or extract_value(text, ["bp", "blood pressure"]),
            "oldpeak": extract_value(text, ["oldpeak", "st depression"]),
        }

    elif disease == "Kidney":
        vals = {
            "sc": extract_value(text, ["creatinine", "serum creatinine"]),
            "bu": extract_value(text, ["urea", "blood urea", "bun"]),
            "hemo": extract_value(text, ["hemoglobin", "hb"]),
        }

    else:
        vals = {}

    if not any(vals.values()):
        vals = fallback_fn(text)

    return {k: v for k, v in vals.items() if v is not None}

# =========================
# REPORT DETECTION
# =========================
def detect_report_type(text):
    text = text.lower()

    scores = {
        "Heart Disease": len(re.findall(r"cholesterol|heart rate|pulse|ecg", text)),
        "Kidney Disease": len(re.findall(r"creatinine|urea|renal|egfr", text)),
        "Diabetes": len(re.findall(r"glucose|hba1c|insulin|sugar", text))
    }

    return max(scores, key=scores.get)

# =========================
# VALIDATION
# =========================
def validate_report(vals, disease):
    required = {
        "Diabetes": ["Glucose", "BMI"],
        "Heart Disease": ["chol", "thalach", "trestbps"],
        "Kidney Disease": ["sc", "bu", "hemo"]
    }

    present = set(vals.keys())
    match = len(present & set(required[disease]))

    return match >= 2

# =========================
# LOADERS
# =========================
@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(BASE_DIR, "models", name))

@st.cache_data
def load_background(name):
    return pd.read_csv(os.path.join(BASE_DIR, "models", name))

# =========================
# SHAP → TREEEXPLAINER
# =========================
def shap_plot(model, df, bg_df):

    if hasattr(model, "named_steps"):
        final_model = model.named_steps[list(model.named_steps.keys())[-1]]
    else:
        final_model = model

    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(df)

        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

    except:
        def predict_fn(x):
            return model.predict_proba(pd.DataFrame(x, columns=df.columns))

        bg = bg_df[df.columns].sample(min(len(bg_df), 20))
        explainer = shap.KernelExplainer(predict_fn, bg.values)
        shap_values = explainer.shap_values(df.values, nsamples=20)

        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

    vals = np.array(vals).reshape(-1)

    if len(vals) != len(df.columns):
        vals = vals[:len(df.columns)]

    feature_df = pd.DataFrame({
        "Feature": df.columns,
        "Impact": vals
    })

    feature_df["AbsImpact"] = feature_df["Impact"].abs()
    feature_df = feature_df.sort_values(by="AbsImpact", ascending=False)

    return feature_df

# =========================
# SESSION
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# =========================
# LOGIN
# =========================
if not st.session_state.logged_in:
    st.title("🩺 Smart Healthcare Risk Prediction")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(u, p):
            st.session_state.logged_in = True
            st.session_state.username = u
            st.rerun()

    st.stop()

# =========================
# MENU
# =========================
menu = st.sidebar.selectbox("Menu", ["Prediction", "History", "Dashboard", "Logout"])

if menu == "Logout":
    st.session_state.clear()
    st.rerun()

# =========================
# CONFIG
# =========================
CONFIG = {
    "Diabetes": {
        "model": "diabetes_model.pkl",
        "bg": "diabetes_background.csv",
        "extract": extract_diabetes_values
    },
    "Heart Disease": {
        "model": "heart_model.pkl",
        "bg": "heart_background.csv",
        "extract": extract_heart_values
    },
    "Kidney Disease": {
        "model": "kidney_model.pkl",
        "bg": "kidney_background.csv",
        "extract": extract_kidney_values
    }
}

# =========================
# PREDICTION
# =========================
if menu == "Prediction":

    st.title("🔬 Disease Prediction")

    disease = st.selectbox("Select Disease", list(CONFIG.keys()))
    cfg = CONFIG[disease]

    model = load_model(cfg["model"])
    bg_df = load_background(cfg["bg"])

    pdf = st.file_uploader("📄 Upload Report", type=["pdf"])

    if pdf:
        text = extract_text_from_pdf(pdf)

        st.subheader("🧪 Extracted Parameters")

        vals = smart_extract(text, disease.split()[0], cfg["extract"])
        st.write("🔍 Extracted Values:", vals)

        detected = detect_report_type(text)
        st.info(f"Detected Report Type: {detected}")

        if detected != disease:
            st.error(f"❌ This looks like a {detected} report, not {disease}")
            st.stop()

        if not validate_report(vals, disease):
            st.error("❌ Required parameters not found")
            st.stop()

        st.success("✔ Valid report detected")

        df = pd.DataFrame([vals])
        df = df.reindex(columns=bg_df.columns, fill_value=0)

        st.write(df)

        if st.button("Analyze"):

            pred = model.predict(df)[0]
            prob = model.predict_proba(df)[0][1]

            st.subheader("👤 Patient Details")
            st.write("Name:", extract_name(text))
            st.write("Age:", extract_age(text))
            st.write("Gender:", extract_gender(text))

            # =========================
            # 🔥 INTERACTIVE DASHBOARD
            # =========================
            st.subheader("📊 Feature Importance Dashboard")

            feature_df = shap_plot(model, df, bg_df)
            top_df = feature_df.head(10)

            fig = px.bar(
                top_df,
                x="Impact",
                y="Feature",
                orientation="h",
                color="Impact",
                color_continuous_scale=["green", "red"],
                title="Top Features Affecting Prediction"
            )

            fig.update_layout(yaxis=dict(autorange="reversed"))

            st.plotly_chart(fig, use_container_width=True)

            # =========================
            # 🧠 EXPLANATION
            # =========================
            st.subheader("🧠 Risk Explanation")

            for _, row in top_df.head(5).iterrows():
                if row["Impact"] > 0:
                    st.error(f"{row['Feature']} increases risk")
                else:
                    st.success(f"{row['Feature']} reduces risk")

            # =========================
            # 🏥 RECOMMENDATION
            # =========================
            st.subheader("🏥 Recommendation")

            if pred == 1:
                st.error("Consult doctor immediately")
            else:
                st.success("Maintain healthy lifestyle")

            st.warning("⚠️ AI prediction only. Not a medical diagnosis.")

            save_history(st.session_state.username, "User", disease, pred, prob)

# =========================
# HISTORY
# =========================
elif menu == "History":
    st.title("📋 History")
    data = get_history(st.session_state.username)
    df = pd.DataFrame(data, columns=["Name", "Disease", "Prediction", "Probability", "Time"])
    st.dataframe(df)

# =========================
# DASHBOARD
# =========================
elif menu == "Dashboard":
    st.title("📊 Dashboard")
    data = get_history(st.session_state.username)
    df = pd.DataFrame(data, columns=["Name", "Disease", "Prediction", "Probability", "Time"])

    if len(df) == 0:
        st.warning("No data")
    else:
        st.bar_chart(df["Disease"].value_counts())

        fig = px.line(df, x="Time", y="Probability",
                      hover_data=["Name", "Disease"])
        st.plotly_chart(fig)