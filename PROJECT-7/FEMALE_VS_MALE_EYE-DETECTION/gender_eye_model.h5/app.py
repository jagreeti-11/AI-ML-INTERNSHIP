import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("👁️ Female vs Male Eye Detection")

# Get the path where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "gender_eye_model.h5")

tf.keras.backend.clear_session()

model_loaded = False
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model_loaded = True
    except Exception as e:
        st.error(f"⚠️ Keras cannot read the file. Details: {e}")
else:
    st.error(f"⚠️ 'gender_eye_model.h5' not found at: {model_path}")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    if model_loaded:
        img_resized = img.resize((64, 64))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        prediction = model.predict(img_array)
        
        if prediction[0][0] > 0.5:
            st.success("Prediction: Male")
        else:
            st.success("Prediction: Female")
