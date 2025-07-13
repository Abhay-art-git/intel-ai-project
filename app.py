import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from helpers.preprocess import preprocess_image  # renamed from utils to avoid cv2 conflict

# Load trained model
model = tf.keras.models.load_model('model/defect_model.h5')
labels = ['Defective', 'Non-Defective']

# Streamlit UI
st.title("📦 Visual Quality Check System")
st.markdown("Upload a product image to check if it's **Defective** or **Non-Defective**.")

# Image upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image as bytes and convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display image using PIL
    st.image(Image.fromarray(image_rgb), caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed = preprocess_image(image_rgb)
    prediction = model.predict(processed)[0]
    label = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show result
    st.markdown(f"### ✅ Prediction: `{label}`")
    st.markdown(f"### 🔢 Confidence: `{confidence:.2f}`")
