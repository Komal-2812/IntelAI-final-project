import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the model
model = tf.keras.models.load_model("keras.h5")

# Class names
class_names = ['Pass', 'Fail']

# Page configuration (full screen)
st.set_page_config(
    page_title="🧪 Wafer Pass/Fail Classifier",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom right, #e3f2fd, #fce4ec);
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            font-size: 48px;
            color: #004d7a;
            text-align: center;
            margin-top: 10px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #333;
        }
        .card {
            background-color: white;
            padding: 2rem;
            margin: 2rem auto;
            border-radius: 16px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.1);
            max-width: 900px;
        }
        .result-pass {
            color: #2e7d32;
            font-size: 32px;
            font-weight: bold;
        }
        .result-fail {
            color: #c62828;
            font-size: 32px;
            font-weight: bold;
        }
        .emoji {
            font-size: 60px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='main-title'>🧪 Semiconductor Wafer Pass/Fail Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>📷 Upload or capture a wafer image to predict whether it PASSES ✅ or FAILS ❌ the quality check.</div>", unsafe_allow_html=True)

# Spacer
st.markdown("<br>", unsafe_allow_html=True)

# Input Section
with st.container():
    input_method = st.radio("🔍 Choose Image Source:", ("📁 Upload Image", "📸 Use Camera"), horizontal=True)

    image = None
    with st.container():
        if input_method == "📁 Upload Image":
            uploaded_file = st.file_uploader("Drag or select an image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                try:
                    image = Image.open(io.BytesIO(uploaded_file.read()))
                except Exception:
                    st.error("❌ Invalid image file.")
        elif input_method == "📸 Use Camera":
            camera_image = st.camera_input("Capture Wafer Photo")
            if camera_image:
                try:
                    image = Image.open(camera_image)
                except Exception:
                    st.error("❌ Unable to access camera image.")

# Prediction
if image:
    # Card UI for prediction section
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, caption="📷 Input Wafer Image", use_container_width=True)

        # Preprocessing
        image = image.resize((224, 224))
        img_array = np.asarray(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        # Results
        result_icon = "✅" if predicted_class == "Pass" else "❌"
        result_class = "result-pass" if predicted_class == "Pass" else "result-fail"
        emoji_display = "🟢✅🟢" if predicted_class == "Pass" else "🔴❌🔴"

        st.markdown(f"<div class='emoji'>{emoji_display}</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='{result_class}' style='text-align:center;'>{result_icon} Result: <strong>{predicted_class.upper()}</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; font-size: 18px;'>Confidence: <strong>{confidence*100:.2f}%</strong></p>", unsafe_allow_html=True)
        st.progress(confidence)

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("📂 Please upload or capture a wafer image to get started.")
