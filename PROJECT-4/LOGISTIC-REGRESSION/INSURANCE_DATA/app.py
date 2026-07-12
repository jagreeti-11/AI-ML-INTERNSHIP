import streamlit as st
import joblib
import numpy as np

import os
import joblib

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the correct absolute path to the model file
model_path = os.path.join(BASE_DIR, "insurance_model.pkl")

# Load the model
model = joblib.load(model_path)

# Page configuration
st.set_page_config(
    page_title="Insurance Sales Prediction",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 Insurance Sales Prediction")
st.write("Predict whether a person will buy insurance using Logistic Regression.")

# User Input
age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=100,
    value=25,
    step=1
)

# Prediction
if st.button("Predict"):

    prediction = model.predict(np.array([[age]]))[0]
    probability = model.predict_proba(np.array([[age]]))[0]

    if prediction == 1:
        st.success("✅ Prediction: Insurance Purchased (YES)")
    else:
        st.error("❌ Prediction: Insurance Not Purchased (NO)")

    st.subheader("Prediction Confidence")
    st.write(f"❌ No Probability: **{probability[0]*100:.2f}%**")
    st.write(f"✅ Yes Probability: **{probability[1]*100:.2f}%**")
