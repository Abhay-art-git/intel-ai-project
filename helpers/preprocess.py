import numpy as np
import cv2

def preprocess_image(image, target_size=(64, 64)):
    resized = cv2.resize(image, target_size)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)  # shape: (1, 64, 64, 3)
