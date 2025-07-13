import numpy as np
from PIL import Image

def preprocess_image(img_array, target_size=(64, 64)):
    img = Image.fromarray(img_array).resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img
