import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from utils.preprocess import preprocess_image

# ✅ Load model from directory (not .h5 file)
model = tf.keras.models.load_model('model/')
labels = ['Defective', 'Non-Defective']

st.title("📦 Visual Quality Check System")
st.markdown("Upload a product image to check if it's **Defective** or **Non-Defective**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img_cv = np.array(image)
    if img_cv.shape[-1] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)

    # Preprocess and predict
    processed = preprocess_image(img_cv)  # (1, 64, 64, 3)
    prediction = model.predict(processed)[0]
    label = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.markdown(f"### ✅ Prediction: `{label}`")
    st.markdown(f"### 🔢 Confidence: `{confidence:.2f}`")
