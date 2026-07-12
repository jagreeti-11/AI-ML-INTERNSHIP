import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Predict house price using Linear Regression")

# Find the correct directory folder path dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "houseprice.csv")

# Cache the dataset loading and training phase to prevent lag/crashes
@st.cache_resource
def load_and_train():
    if not os.path.exists(csv_path):
        return None, None
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Split into features and target
    X = df.drop("price", axis=1)
    y = df["price"]
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    
    return df, model

df, model = load_and_train()

# Check if the data exists before continuing
if df is None:
    st.error(f"⚠️ Could not find 'houseprice.csv' in the folder: {BASE_DIR}. Please make sure you uploaded the CSV file to GitHub!")
else:
    st.subheader("Dataset")
    st.dataframe(df.head())
    
    st.markdown("---")
    st.subheader("Enter House Details")
    
    # Create inputs dynamically based on the features the model needs
    feature_names = df.drop("price", axis=1).columns
    user_inputs = {}
    
    for feature in feature_names:
        user_inputs[feature] = st.number_input(f"Enter {feature}", min_value=0.0, step=1.0)
        
    # Prediction phase
    if st.button("Predict Price"):
        # Format inputs into a clean DataFrame matching feature names
        input_df = pd.DataFrame([user_inputs])
        prediction = model.predict(input_df)
        
        st.success(f"💰 Estimated House Price: ${prediction[0]:,.2f}")
