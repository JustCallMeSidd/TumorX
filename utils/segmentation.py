import cv2
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st

@st.cache_resource
def load_unet(path):
    return load_model(path, compile=False)

def segment_image_heatmap(model, image_path, target_size=(128, 128), alpha=0.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    original_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_resized = cv2.resize(img, target_size)
    img_norm = img_resized / 255.0
    img_norm = np.expand_dims(img_norm, axis=-1)
    img_norm = np.expand_dims(img_norm, axis=0)

    prediction = model.predict(img_norm)[0].squeeze()
    prediction_resized = cv2.resize(prediction, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap((prediction_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)

    return overlay
