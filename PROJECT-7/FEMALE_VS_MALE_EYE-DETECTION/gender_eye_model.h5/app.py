import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("👁️ Female vs Male Eye Detection")

# --- Step 1: Safely Find and Load the Model File ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path option A: File is in the same folder as app.py
path_a = os.path.join(current_dir, "gender_eye_model.h5")
# Path option B: File is one folder level above app.py (if caught in a folder trap)
path_b = os.path.join(os.path.dirname(current_dir), "gender_eye_model.h5")

# Decide which path actually exists
if os.path.exists(path_a):
    model_path = path_a
else:
    model_path = path_b

tf.keras.backend.clear_session()

# Load the model with fallback error messages
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    model_loaded = True
except Exception as e:
    try:
        model = tf.keras.models.load_model(model_path)
        model_loaded = True
    except Exception as e2:
        model_loaded = False
        st.error("⚠️ Model file could not be loaded. Please ensure 'gender_eye_model.h5' is successfully uploaded to GitHub.")

# --- Step 2: Main Application Execution ---
uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Open and display the image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # 2. Only run prediction if the model loaded successfully
    if model_loaded:
        # Preprocess the image to match model expectations (64x64)
        img_resized = img.resize((64, 64))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        prediction = model.predict(img_array)
        
        # Display Result
        if prediction[0][0] > 0.5:
            st.success("Prediction: Male")
        else:
            st.success("Prediction: Female")
    else:
        st.warning("Prediction skipped because the model file is missing or corrupted.")
