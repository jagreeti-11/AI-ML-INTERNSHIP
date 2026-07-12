import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("👁 Female vs Male Eye Detection")

model = tf.keras.models.load_model("gender_eye_model.keras")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.success("Prediction: Male")
    else:
        st.success("Prediction: Female")
