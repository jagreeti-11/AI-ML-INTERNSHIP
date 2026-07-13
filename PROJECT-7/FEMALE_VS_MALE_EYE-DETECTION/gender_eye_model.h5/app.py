import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("👁️ Female vs Male Eye Detection")


import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "gender_eye_model.h5")
# If GitHub turned it into a double folder, check inside it
if not os.path.exists(model_path):
    model_path = os.path.join(current_dir, "gender_eye_model.h5")

# Force Keras to clear session bugs
tf.keras.backend.clear_session()

# Try loading by bypassing rigid compilation checks
try:
    model = tf.keras.models.load_model(model_path, compile=False)
except Exception as e:
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e2:
        st.error(f"Model file structure issue. Details: {e2}")

if uploaded_file is not None:
    # 1. Open and display the image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # 2. Preprocess the image to match model expectations
    img = img.resize((64, 64))
    img_array = tf.keras.utils.img_to_array(img)  # Safe and explicit Keras conversion
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # 3. Predict
    prediction = model.predict(img_array)
    
    # 4. Display Result
    if prediction[0][0] > 0.5:
        st.success("Prediction: Male")
    else:
        st.success("Prediction: Female")
