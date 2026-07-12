import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(
    page_title="Canada Per Capita Income Prediction",
    page_icon="🍁",
    layout="centered"
)

# 1. Safely locate the model file inside your folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "canada_income_model.pkl")

# 2. Cache the model loading process to save memory and prevent server crashes
@st.cache_resource
def load_my_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_my_model()

st.title("🍁 Canada Per Capita Income Prediction")
st.write("Predict Canada's Per Capita Income using Linear Regression")

# 3. Check if the model loaded successfully before showing UI elements
if model is None:
    st.error(f"⚠️ Missing Model File! Please ensure 'canada_income_model.pkl' is uploaded to GitHub in the exact same folder as app.py.")
else:
    # User Input
    year = st.number_input(
        "Enter Year",
        min_value=1970,
        max_value=2050,
        value=2020,
        step=1
    )

    # Prediction
    if st.button("Predict Income"):
        prediction = model.predict(pd.DataFrame([[year]], columns=["year"]))
        st.success(f"Predicted Per Capita Income: ${prediction[0]:,.2f}")

# About
st.markdown("---")
st.subheader("About Project")
st.write("""
- Model: Linear Regression
- Dataset: Canada Per Capita Income
- Feature Used: Year
- Target: Per Capita Income (US$)
""")
