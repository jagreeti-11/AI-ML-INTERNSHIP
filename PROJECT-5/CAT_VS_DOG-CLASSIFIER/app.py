import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# --- Keras Version Compatibility Patch ---
original_init = tf.keras.layers.InputLayer.__init__
def patched_init(self, *args, **kwargs):
    kwargs.pop('batch_shape', None)
    original_init(self, *args, **kwargs)
tf.keras.layers.InputLayer.__init__ = patched_init
# ------------------------------------------

st.title("🐶 Cat vs Dog Classifier 🐱")

# --- Cloud Safe Path Validation ---
model_folder = "train_model" 
model_path = os.path.join(model_folder, "dog_cat_model.h5") 

if not os.path.exists(model_path):
    model_path = "dog_cat_model.h5"  # Fallback to the root directory

# --- Load the Pre-trained Model Directly ---
@st.cache_resource
def load_trained_model():
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        # Show a clear error on Streamlit UI if your model file is missing
        st.error(f"❌ Model file not found at `{model_path}`. Please upload `dog_cat_model.h5` to your repository.")
        st.stop()

model = load_trained_model()

# --- Streamlit UI File Uploader ---
uploaded_file = st.file_uploader("Upload an Image of a Cat or a Dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Image Preprocessing (Matches your model's expected 64x64 input dimension)
    img_resized = img.resize((64, 64))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0                  
    
    # Run prediction using the loaded model
    with st.spinner("Classifying the image..."):
        prediction = model.predict(img_array)
    
    score = prediction[0][0]
    
    # Display Results (Assuming 1 = Dog, 0 = Cat)
    if score > 0.5:
        confidence = score * 100
        st.success(f"Prediction: **Dog** 🐶 (Confidence: {confidence:.2f}%)")
    else:
        confidence = (1 - score) * 100
        st.success(f"Prediction: **Cat** 🐱 (Confidence: {confidence:.2f}%)")
