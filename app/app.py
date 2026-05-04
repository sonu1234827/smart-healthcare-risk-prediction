import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np
import sys, os
import re
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.pdf_parser import *
from src.utils.database import *

st.set_page_config(page_title="Healthcare AI", layout="wide")

init_db()

# -------------------------
# SESSION
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# -------------------------
# SMART EXTRACTION FUNCTIONS (FIXED)
# -------------------------
def extract_name(text):
    patterns = [
        r"Patient Name[:\- ]+([A-Za-z ]+)",
        r"Name[:\- ]+([A-Za-z ]+)",
        r"Pt Name[:\- ]+([A-Za-z ]+)",
        r"Patient[:\- ]+([A-Za-z ]+)"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if len(name) > 2:
                return name
    return "Unknown"

def extract_age(text):
    patterns = [
        r"Age[:\- ]+(\d+)",
        r"(\d+)\s*years",
        r"(\d+)\s*yrs"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return "Unknown"

def extract_gender(text):
    text = text.lower()
    if "female" in text:
        return "Female"
    if "male" in text:
        return "Male"
    return "Unknown"

# -------------------------
# LOGIN PAGE (UNCHANGED)
# -------------------------
if not st.session_state.logged_in:

    st.markdown("""
    <style>
    .stApp {
        background: url("https://images.unsplash.com/photo-1588776814546-1ffcf47267a5") no-repeat center center fixed;
        background-size: cover;
    }
    .login-box {
        width: 400px;
        margin: auto;
        margin-top: 100px;
        padding: 30px;
        border-radius: 12px;
        background: #1e40af;
        border: 2px solid white;
        color: white;
    }
    label { color:white !important; font-weight:bold; }
    input { background:#1e293b !important; color:white !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;color:white;'>🩺 Smart Healthcare Risk Prediction System</h1>", unsafe_allow_html=True)

    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Register"):
            register_user(u, p)

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------------------------
# MAIN UI
# -------------------------
st.title("🩺 Smart Healthcare System")

menu = st.sidebar.selectbox("Menu", ["Prediction","History","Dashboard","Logout"])

if menu == "Logout":
    st.session_state.clear()
    st.rerun()

# -------------------------
# MODELS
# -------------------------
diabetes_model = joblib.load("models/diabetes_model.pkl")
heart_model = joblib.load("models/heart_model.pkl")
kidney_model = joblib.load("models/kidney_model.pkl")

def align(df, path):
    bg = pd.read_csv(path)
    for col in bg.columns:
        if col not in df.columns:
            df[col] = 0
    return df[bg.columns]

# -------------------------
# SHAP FUNCTION (FIXED)
# -------------------------
def shap_plot(model, df, path):
    bg = pd.read_csv(path).sample(10)

    def f(x):
        return model.predict_proba(pd.DataFrame(x, columns=df.columns))

    explainer = shap.KernelExplainer(f, bg.values)
    shap_values = explainer.shap_values(df.values, nsamples=20)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    shap_vals = np.array(shap_vals)

    if shap_vals.ndim > 1:
        shap_vals = shap_vals[0]

    shap_vals = shap_vals.flatten()

    min_len = min(len(df.columns), len(shap_vals))

    features = df.columns[:min_len]
    shap_vals = shap_vals[:min_len]

    colors = ["red" if v > 0 else "green" for v in shap_vals]

    fig, ax = plt.subplots()
    ax.barh(features, shap_vals, color=colors)
    ax.axvline(0)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='red', label='Increase Risk'),
        Patch(color='green', label='Reduce Risk')
    ])

    st.pyplot(fig)

# -------------------------
# PREDICTION
# -------------------------
if menu == "Prediction":

    pdf = st.file_uploader("Upload PDF")

    if pdf and st.button("Analyze"):

        text = extract_text_from_pdf(pdf)

        # -------------------------
        # FIXED EXTRACTION
        # -------------------------
        name = extract_name(text)
        age = extract_age(text)
        gender = extract_gender(text)

        st.markdown(f"""
        <div style="background:#1e293b;padding:15px;border-radius:10px;">
        <h4>👤 Patient Details</h4>
        Name: {name} <br>
        Age: {age} <br>
        Gender: {gender}
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # MODEL SELECTION
        # -------------------------
        if "creatinine" in text.lower():
            vals = extract_kidney_values(text)
            model = kidney_model
            path = "models/kidney_background.csv"
            disease = "Kidney"

        elif "cholesterol" in text.lower():
            vals = extract_heart_values(text)
            model = heart_model
            path = "models/heart_background.csv"
            disease = "Heart"

        else:
            vals = extract_diabetes_values(text)
            model = diabetes_model
            path = "models/diabetes_background.csv"
            disease = "Diabetes"

        df = pd.DataFrame([vals])
        df = align(df, path)

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        if pred == 1:
            st.error(f"🔴 HIGH RISK ({prob*100:.2f}%)")
        else:
            st.success(f"🟢 LOW RISK ({prob*100:.2f}%)")

        shap_plot(model, df, path)

        st.markdown("### 🧠 Risk Explanation")
        st.write(f"Model predicts risk based on health parameters for {disease}.")

        st.markdown("### 🎯 Suggestions")
        st.write("• Exercise regularly")
        st.write("• Maintain diet")
        st.write("• Monitor health")

        st.markdown("### 🏥 Recommendations")
        if pred == 1:
            st.write("• Consult doctor")
            st.write("• Take tests")
        else:
            st.write("• Continue healthy lifestyle")

        save_history(st.session_state.username, name, disease, pred, prob)

# -------------------------
# HISTORY
# -------------------------
elif menu == "History":
    data = get_history(st.session_state.username)
    st.dataframe(pd.DataFrame(data, columns=["Name","Disease","Prediction","Probability","Time"]))

# -------------------------
# DASHBOARD
# -------------------------
elif menu == "Dashboard":

    data = get_history(st.session_state.username)

    if data:
        df = pd.DataFrame(data, columns=["Name","Disease","Prediction","Probability","Time"])

        st.bar_chart(df["Disease"].value_counts())

        fig = px.line(
            df,
            x="Time",
            y="Probability",
            markers=True,
            hover_data=["Name","Disease"]
        )

        st.plotly_chart(fig)

    else:
        st.warning("No data available")