import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Title
st.set_page_config(page_title="Female vs Male Eye Detection", page_icon="👁")

st.title("👁 Female vs Male Eye Detection")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_eye_model.h5")

model = load_model()

# Upload Image
uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    img = image.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.success("Prediction: Male 👨")
    else:
        st.success("Prediction: Female 👩")
