import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the model
model = tf.keras.models.load_model("keras.h5")

# Class labels
class_names = ['Pass', 'Fail']

# Set Streamlit config
st.set_page_config(page_title="Wafer Classifier", layout="wide")

# 💅 Dark theme custom CSS
st.markdown("""
    <style>
    body {
        background-color: #0f0f0f;
        color: white;
    }
    .stApp {
        background-color: #0f0f0f;
    }
    .title {
        font-size: 48px;
        color: #00ffe7;
        text-align: center;
        margin-top: 30px;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cccccc;
        margin-bottom: 30px;
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: auto;
        max-width: 800px;
    }
    .result-pass {
        color: #00ff91;
        font-size: 32px;
        text-align: center;
        font-weight: bold;
    }
    .result-fail {
        color: #ff3c3c;
        font-size: 32px;
        text-align: center;
        font-weight: bold;
    }
    .emoji {
        font-size: 60px;
        text-align: center;
        margin-bottom: 10px;
    }
    .confidence-bar .stProgress > div > div {
        background-image: linear-gradient(to right, #1cefff, #00ff91);
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Title & intro
st.markdown("<div class='title'>🔬 Wafer Pass/Fail Prediction</div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload or capture a wafer image to predict whether it passes ✅ or fails ❌ the inspection.</p>", unsafe_allow_html=True)

# Input method
input_method = st.radio("📷 Select Image Input Method:", ["📁 Upload Image", "📸 Use Camera"], horizontal=True, label_visibility="collapsed")

image = None

# Input block
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader("📂 Upload Wafer Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            try:
                image = Image.open(io.BytesIO(uploaded_file.read()))
            except:
                st.error("❌ Could not read the uploaded file.")
    elif input_method == "📸 Use Camera":
        camera_image = st.camera_input("📷 Capture Wafer Photo")
        if camera_image:
            try:
                image = Image.open(camera_image)
            except:
                st.error("❌ Camera input failed.")
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction block
if image:
    st.image(image, caption="📸 Input Image", use_container_width=True)

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.asarray(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    result_icon = "✅" if predicted_class == "Pass" else "❌"
    result_class = "result-pass" if predicted_class == "Pass" else "result-fail"
    emoji_line = "🟢🟢🟢" if predicted_class == "Pass" else "🔴🔴🔴"

    # Output result
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='emoji'>{emoji_line}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='{result_class}'>{result_icon} Result: {predicted_class.upper()}</div>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:18px;'>Confidence: <b>{confidence * 100:.2f}%</b></p>", unsafe_allow_html=True)

    # Confidence bar
    with st.container():
        st.markdown("<div class='confidence-bar'>", unsafe_allow_html=True)
        st.progress(confidence)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<br>")
    st.info("📂 Please upload or capture a wafer image to get the prediction.")
