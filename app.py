import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from helpers.preprocess import preprocess_image

# Load model (SavedModel format)
model = tf.keras.models.load_model('model/')
labels = ['Defective', 'Non-Defective']

st.title("📦 Visual Quality Check System")
st.markdown("Upload a product image to check if it's **Defective** or **Non-Defective**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")  # ensure RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to array (no cv2 needed)
    img_array = np.array(image)

    # Preprocess and predict
    processed = preprocess_image(img_array)  # (1, 64, 64, 3)
    prediction = model.predict(processed)[0]
    label = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.markdown(f"### ✅ Prediction: `{label}`")
    st.markdown(f"### 🔢 Confidence: `{confidence:.2f}`")
