import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_classifier(path):
    # compile=False ensures it works for inference-only
    return load_model(path, compile=False)

# ---------------------------
# Class names
# ---------------------------
def load_class_names(classes_path="classes.npy", fallback=None):
    if os.path.exists(classes_path):
        return list(np.load(classes_path, allow_pickle=True))
    return fallback or ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# ---------------------------
# Preprocessing + Classification
# ---------------------------
def _get_input_hw_c(model):
    shape = model.input_shape
    if isinstance(shape, (list, tuple)) and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    _, H, W, C = shape
    return (H, W, C)

def _preprocess_image(img_pil, mode="none"):
    x = image.img_to_array(img_pil)
    if mode == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        x = preprocess_input(x)
    elif mode == "rescale01":
        x = x.astype(np.float32) / 255.0
    elif mode == "none":
        x = x.astype(np.float32)
    else:
        raise ValueError("mode must be 'none' | 'rescale01' | 'efficientnet'")
    return np.expand_dims(x, axis=0)

def classify_image(img_path, model, class_names, preprocess="none"):
    # Use model’s own input size
    H, W, _ = _get_input_hw_c(model)
    pil_img = image.load_img(img_path, target_size=(H, W))
    x = _preprocess_image(pil_img, mode=preprocess)

    preds = model.predict(x, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx]) * 100.0
    return class_names[pred_idx], confidence
