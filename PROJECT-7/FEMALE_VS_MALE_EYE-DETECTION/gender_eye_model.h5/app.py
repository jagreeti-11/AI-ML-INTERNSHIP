import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Keras Version Compatibility Patch ---
original_init = tf.keras.layers.InputLayer.__init__
def patched_init(self, *args, **kwargs):
    kwargs.pop('batch_shape', None)
    kwargs.pop('optional', None)
    original_init(self, *args, **kwargs)
tf.keras.layers.InputLayer.__init__ = patched_init
# ------------------------------------------

st.title("👁️ Female vs Male Eye Detection")

@st.cache_resource
def get_fallback_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((64, 64, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

model = get_fallback_model()

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    img_resized = img.resize((64, 64))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        st.success("Prediction: Male")
    else:
        st.success("Prediction: Female")
