import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model
model = tf.keras.models.load_model("keras.h5")

# Class labels
class_names = ['Pass', 'Fail']

# Set Streamlit page config
st.set_page_config(page_title="Wafer Pass/Fail Classifier", layout="centered")

# Custom CSS for modern styling
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f0f4f7;
        }
        .stApp {
            max-width: 700px;
            margin: auto;
        }
        .title {
            text-align: center;
            color: #004D7A;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #333;
            margin-top: 0;
        }
        .result-box {
            border-radius: 20px;
            background: linear-gradient(135deg, #e0f7fa, #e3f2fd);
            padding: 30px;
            text-align: center;
            box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        }
        .result-pass {
            color: #2e7d32;
            font-weight: bold;
            font-size: 24px;
        }
        .result-fail {
            color: #c62828;
            font-weight: bold;
            font-size: 24px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='title'>🧪 Wafer Pass/Fail Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload or capture a semiconductor wafer image to determine if it passes quality inspection.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input method
input_method = st.radio("🖼️ Choose Image Source:", ("Upload Image", "Use Camera"), horizontal=True)
image = None

# Upload or Camera input
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("📁 Upload Wafer Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
        except Exception:
            st.error("⚠️ Invalid image file.")
elif input_method == "Use Camera":
    camera_image = st.camera_input("📷 Take a Picture")
    if camera_image:
        try:
            image = Image.open(camera_image)
        except Exception:
            st.error("⚠️ Failed to capture image.")

# Prediction section
if image:
    st.image(image, caption="🔍 Uploaded Wafer Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.asarray(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # Display result
    result_class = "result-pass" if predicted_class == "Pass" else "result-fail"
    icon = "✅" if predicted_class == "Pass" else "❌"
    result_label = f"{icon} {predicted_class.upper()}"

    st.markdown(f"""
        <div class='result-box'>
            <div class='{result_class}'>{result_label}</div>
            <p>Confidence: <strong>{confidence * 100:.2f}%</strong></p>
        </div>
    """, unsafe_allow_html=True)

    # Fancy confidence bar
    st.markdown("### 📊 Confidence Level")
    st.progress(confidence)
else:
    st.info("📂 Please upload or capture a wafer image to begin.")
