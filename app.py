import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from utils.preprocess import preprocess_image

# Load model
model = tf.keras.models.load_model('model/defect_model.h5')
labels = ['Defective', 'Non-Defective']

# App UI
st.title("📦 Visual Quality Check System")
st.markdown("Upload a product image to classify it as **Defective** or **Non-Defective**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV
    img_cv = np.array(image)
    if img_cv.shape[-1] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)

    # Preprocess and Predict
    processed = preprocess_image(img_cv)
    prediction = model.predict(processed)[0]
    predicted_label = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Output
    st.markdown(f"### ✅ Prediction: `{predicted_label}`")
    st.markdown(f"### 🔢 Confidence: `{confidence:.2f}`")
