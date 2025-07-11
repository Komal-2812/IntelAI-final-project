import streamlit as st # type: ignore
import tensorflow as tf 
import numpy as np # type: ignore
from PIL import Image # type: ignore
import io



# Load the model
model = tf.keras.models.load_model(r"C:\Users\91743\Desktop\final project\keras.h5")

# Define class labels based on your training
class_names = ['Normal', 'Defective']  # Modify if you used different class names

# Page setup
st.set_page_config(page_title="Semiconducter wafer detector", layout="centered")
st.title("🔍 Prediction of semiconducter wafer")
st.write("Upload an image or take a photo to detect anomalies in the product.")

# Choose input method
input_method = st.radio("Choose Image Input Method:", ("Upload Image", "Use Camera"))

image = None

# Upload image block
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
        except Exception as e:
            st.error("⚠ Could not read the image. Please upload a valid image file.")

# Camera input block
elif input_method == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        try:
            image = Image.open(camera_image)
        except Exception as e:
            st.error("⚠ Could not access the image from camera.")

# If image is loaded, make prediction
if image:
    st.image(image, caption="Input Image", use_container_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    img_array = np.asarray(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display results
    st.markdown(f"### 🧠 Prediction: *{predicted_class}*")
    st.markdown(f"### 📊 Confidence: *{confidence * 100:.2f}%*")
