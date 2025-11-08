import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np

# ============================
# Load Model and Encoders
# ============================
@st.cache_resource
def load_model():
    try:
        data = pickle.load(open('heart_disease_model.pkl', 'rb'))
    except:
        data = joblib.load('heart_disease_model.joblib')
    return data['model'], data['encoders']

model, encoders = load_model()

# Extract categorical mappings
category_mappings = {
    col: dict(zip(le.classes_, le.transform(le.classes_)))
    for col, le in encoders.items()
}

# ============================
# Streamlit Page Config
# ============================
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
            color: white;
        }
        div[data-testid="stSidebar"] {
            background-color: #141422;
        }
        h1, h2, h3 {
            color: #ff4b4b;
            text-align: center;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            width: 100%;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #ff6666;
            transform: scale(1.02);
        }
        .card {
            background-color: #1f1f2e;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 0px 10px rgba(255,255,255,0.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# Title Section
# ============================
st.markdown("<h1>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#ccc;'>Enter your health details to check your heart disease risk.</p>",
    unsafe_allow_html=True,
)

# ============================
# Collect User Inputs
# ============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.header("🧍‍♂️ Personal & Health Information")

user_input = {}

# Get column list from model training data
sample_df = pd.DataFrame(columns=model.feature_names_in_)

for col in model.feature_names_in_:
    if col in encoders:  # categorical column
        options = list(category_mappings[col].keys())
        selected = st.selectbox(f"{col}:", options)
        user_input[col] = category_mappings[col][selected]
    else:
        val = st.number_input(f"{col}:", step=0.01)
        user_input[col] = val

st.markdown("</div>", unsafe_allow_html=True)

# ============================
# Prediction Button
# ============================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔍 Predict Risk"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    proba = (
        model.predict_proba(input_df)[0][1]
        if hasattr(model, "predict_proba")
        else None
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(
            "<h2 style='color:#ff4b4b;'>🔴 High Risk of Heart Disease</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h2 style='color:#00ff99;'>🟢 Low Risk of Heart Disease</h2>",
            unsafe_allow_html=True,
        )

    if proba is not None:
        st.markdown(
            f"<p style='text-align:center; color:#aaa;'>Confidence Score: <b>{proba * 100:.2f}%</b></p>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><p style='text-align:center; color:#888;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
