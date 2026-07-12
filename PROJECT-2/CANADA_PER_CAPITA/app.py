import streamlit as st
import joblib
import pandas as pd

# Load Model
model = joblib.load("canada_income_model.pkl")

st.set_page_config(
    page_title="Canada Per Capita Income Prediction",
    page_icon="🍁",
    layout="centered"
)

st.title("🍁 Canada Per Capita Income Prediction")
st.write("Predict Canada's Per Capita Income using Linear Regression")

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