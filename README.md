# 🧠 Brain Tumor Classification + Segmentation (Streamlit App)

This app allows you to upload MRI brain scans and:
- Classify tumor type (Glioma, Meningioma, Pituitary, or No Tumor).
- Segment tumor region using a UNet model.
- Display both classification results and overlay segmentation heatmap.

## 🧪 RO Model Integration
We've now integrated a Region Optimization (RO) model to enhance segmentation precision. This model refines tumor boundaries post-UNet prediction for improved diagnostic clarity.

## 📦 Upcoming Kaggle Release
The complete dataset, model weights, and annotated results will be uploaded to Kaggle soon. Stay tuned for the release!

## 🎥 YouTube Trial Demo
Watch a quick demo of the app in action: [Trial Demo on YouTube](https://www.youtube.com/watch?v=TPH30zT_ViA)

## 🚀 Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
