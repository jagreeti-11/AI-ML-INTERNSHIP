import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# Page Configuration
st.set_page_config(
    page_title="Iris K-Means Clustering",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower K-Means Clustering")

# Load Iris Dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Load Trained KMeans Model
import os

# Get the path where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "kmeans_model.pkl")

# Load the model using the absolute path
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"⚠️ Model file not found. Looked for 'kmeans_model.pkl' at: {model_path}")

# Predict Cluster for all flowers
df["Cluster"] = model.predict(df)

# Show Dataset
st.subheader("Iris Dataset with Cluster Labels")
st.dataframe(df)

# Show Cluster Count
st.subheader("Number of Flowers in Each Cluster")
st.bar_chart(df["Cluster"].value_counts().sort_index())

# Show Cluster Centers
st.subheader("Cluster Centers")
centers = pd.DataFrame(
    model.cluster_centers_,
    columns=iris.feature_names
)
st.dataframe(centers)

# Project Information
st.subheader("Project Information")
st.write("""
- **Algorithm:** K-Means Clustering
- **Dataset:** Iris Dataset
- **Number of Clusters:** 3
- **Features Used:**
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
""")
