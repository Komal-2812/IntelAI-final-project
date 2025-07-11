import streamlit as st
import tensorflow as tf 
import numpy as np
from PIL import Image
import io

# Load the model
model = tf.keras.models.load_model("keras.h5")

# Define class labels
class_names = ['Normal', 'Defective']

# Page setup
st.set_page_config(page_title="🔍 Wafer Defect Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: #00BFFF;'>🔬 Semiconductor Wafer Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image or take a photo to detect whether the wafer is <b>Normal</b> or <b>Defective</b>.</p>", unsafe_allow_html=True)
st.markdown("---")

# Choose input method
input_method = st.radio("📸 Choose Image Input Method:", ("Upload Image", "Use Camera"), horizontal=True)

image = None

with st.container():
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("🖼️ Upload a wafer image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            try:
                image = Image.open(io.BytesIO(uploaded_file.read()))
            except Exception as e:
                st.error("⚠ Could not read the image. Please upload a valid image file.")
    elif input_method == "Use Camera":
        camera_image = st.camera_input("📷 Take a picture")
        if camera_image:
            try:
                image = Image.open(camera_image)
            except Exception as e:
                st.error("⚠ Could not access the image from camera.")

# Predict if image is loaded
if image:
    st.markdown("---")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    img_array = np.asarray(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Result Output
    st.markdown(f"<h3 style='color: #4CAF50;'>🧠 Prediction: {predicted_class}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color: #555;'>📊 Confidence: {confidence*100:.2f}%</h4>", unsafe_allow_html=True)
    st.progress(float(confidence))
else:
    st.info("📂 Please upload or capture a wafer image to start.")
