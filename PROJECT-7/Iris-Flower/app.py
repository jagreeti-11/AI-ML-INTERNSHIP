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
model = joblib.load("kmeans_model.pkl")

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
