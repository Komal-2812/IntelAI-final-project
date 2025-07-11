import streamlit as st
import tensorflow as tf 
import numpy as np
from PIL import Image
import io

# Load model
model = tf.keras.models.load_model("keras.h5")

# Class labels
class_names = ['Pass', 'Fail']  # Match your real project classes

# Set page config
st.set_page_config(page_title="🧪 Wafer Pass/Fail Classifier", layout="centered")

# Custom Title
st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#1F77B4;">🧪 Semiconductor Wafer Pass/Fail Prediction</h1>
        <p>Upload or capture a wafer image to determine whether it <b>PASSES</b> or <b>FAILS</b> quality inspection.</p>
    </div>
    <hr style="border:1px solid #f0f0f0;">
""", unsafe_allow_html=True)

# Input method selection
input_method = st.radio("📸 Select Input Method:", ("Upload Image", "Use Camera"), horizontal=True)

image = None

with st.container():
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("🖼️ Upload Wafer Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            try:
                image = Image.open(io.BytesIO(uploaded_file.read()))
            except Exception as e:
                st.error("⚠️ Could not read the image.")
    elif input_method == "Use Camera":
        camera_image = st.camera_input("📷 Capture Wafer Image")
        if camera_image:
            try:
                image = Image.open(camera_image)
            except Exception as e:
                st.error("⚠️ Could not access image.")
                
# Predict if image is loaded
if image:
    st.image(image, caption="🔍 Input Wafer Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.asarray(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display
    result_color = "#28a745" if predicted_class == "Pass" else "#dc3545"
    result_icon = "✅" if predicted_class == "Pass" else "❌"

    st.markdown(f"""
        <div style="text-align:center; margin-top:30px;">
            <h2 style="color:{result_color};">{result_icon} Result: {predicted_class}</h2>
            <p style="font-size:18px;">Confidence: <b>{confidence * 100:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

    st.progress(float(confidence))

else:
    st.info("📂 Upload or capture a wafer image to start the prediction.")
