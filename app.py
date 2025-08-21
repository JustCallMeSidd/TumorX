import streamlit as st
import os
from utils.classifier import classify_image, load_classifier
from utils.segmentation import segment_image_heatmap, load_unet
from PIL import Image
import base64

# ---- Safe Defaults ----
img_path = None
overlay = None
pred_class = None
confidence = None

# -----------------------------
# Page Config
# -----------------------------

def get_base64_encoded_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

favicon_b64 = get_base64_encoded_image("favicon.ico")

st.set_page_config(
    page_title="TumorX",
    page_icon=f"data:image/png;base64,{favicon_b64}" if favicon_b64 else "🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Enhanced Custom CSS with Animations
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Orbitron:wght@400;500;600;700;800;900&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* Remove top padding/margin everywhere */
    .stApp > div:first-child,
    .main .block-container,
    section.main > div,
    .element-container:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #000000 100%);
        background-attachment: fixed;
        min-height: 100vh;
        color: #ffffff;
        padding-top: 0 !important;
        margin-top: 0 !important;
        position: relative;
        overflow-x: hidden;
    }

    /* Animated background particles */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
        animation: floatingParticles 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }

    @keyframes floatingParticles {
        0%, 100% { transform: translateX(0) translateY(0); }
        25% { transform: translateX(-20px) translateY(-10px); }
        50% { transform: translateX(20px) translateY(-20px); }
        75% { transform: translateX(-10px) translateY(10px); }
    }

    /* Header - no spacing above the TumorX title */
    .header-section {
        text-align: center;
        padding-top: 0 !important;
        margin-top: 0 !important;
        position: relative;
        overflow: hidden;
        animation: slideDown 1.5s ease-out;
    }

    /* Add interactive click ripple effect */
    .header-section::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.2) 0%, transparent 70%);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none;
        z-index: -1;
        opacity: 0;
    }

    .logo-title:active ~ .header-section::after,
    .logo-title:focus ~ .header-section::after {
        animation: ripple 0.8s ease-out;
    }

    @keyframes ripple {
        0% {
            width: 0;
            height: 0;
            opacity: 1;
        }
        100% {
            width: 400px;
            height: 400px;
            opacity: 0;
        }
    }

    @keyframes slideDown {
        from {
            transform: translateY(-50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    .header-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 150px;
        height: 150px;
        background: conic-gradient(
            from 0deg,
            rgba(99, 102, 241, 0.1) 0deg,
            rgba(139, 92, 246, 0.15) 60deg,
            rgba(59, 130, 246, 0.1) 120deg,
            rgba(99, 102, 241, 0.05) 180deg,
            rgba(139, 92, 246, 0.1) 240deg,
            rgba(99, 102, 241, 0.15) 300deg,
            rgba(99, 102, 241, 0.1) 360deg
        );
        border-radius: 50%;
        z-index: 0;
        animation: rotate 8s linear infinite, breathe 4s ease-in-out infinite alternate;
    }

    @keyframes rotate {
        from { transform: translateX(-50%) rotate(0deg); }
        to { transform: translateX(-50%) rotate(360deg); }
    }

    @keyframes breathe {
        from { 
            scale: 1;
            opacity: 0.4;
        }
        to { 
            scale: 1.1;
            opacity: 0.7;
        }
    }

    .logo-title {
        font-family: 'Orbitron', monospace;
        font-size: clamp(2.5rem, 6vw, 4.5rem);
        font-weight: 900;
        background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 50%, #64748b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
        cursor: pointer;
        transition: all 0.3s ease;
        animation: float 6s ease-in-out infinite;
    }

    .logo-title:hover {
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        transform: translateY(-5px) scale(1.05);
        filter: drop-shadow(0 10px 20px rgba(99,102,241,0.4));
        animation: hoverPulse 0.6s ease-in-out infinite alternate;
    }

    @keyframes float {
        0%, 100% {
            transform: translateY(0);
        }
        25% {
            transform: translateY(-8px);
        }
        50% {
            transform: translateY(0);
        }
        75% {
            transform: translateY(-4px);
        }
    }

    @keyframes hoverPulse {
        from {
            text-shadow: 0 0 10px rgba(99,102,241,0.3);
        }
        to {
            text-shadow: 0 0 20px rgba(99,102,241,0.6);
        }
    }

    /* About Section Animation */
    .about-section {
        font-size: 1.1rem;
        line-height: 1.8;
        max-width: 800px;
        margin: 2rem auto;
        text-align: center;
        color: #e5e7eb;
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2rem;
        animation: fadeInUp 1s ease-out 0.3s both;
        position: relative;
        overflow: hidden;
    }

    .about-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.1), transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    @keyframes fadeInUp {
        from {
            transform: translateY(30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    /* Upload Section */
    .upload-section {
        animation: fadeInUp 1s ease-out 0.6s both;
        max-width: 600px;
        margin: 2rem auto;
    }

    /* Modern File Upload Area */
    .stFileUploader {
        border: none !important;
        background: none !important;
    }

    .stFileUploader > div {
        border: 2px dashed rgba(99, 102, 241, 0.3) !important;
        border-radius: 20px !important;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%) !important;
        backdrop-filter: blur(10px) !important;
        padding: 3rem 2rem !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stFileUploader > div:hover {
        border-color: rgba(99, 102, 241, 0.6) !important;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2) !important;
    }

    .stFileUploader > div::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.1), transparent);
        animation: shimmerUpload 3s infinite;
        pointer-events: none;
    }

    @keyframes shimmerUpload {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    /* Upload Icon and Text */
    .stFileUploader label {
        color: #e5e7eb !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 1rem !important;
        cursor: pointer !important;
    }

    .stFileUploader label::before {
        content: "📤";
        font-size: 3rem;
        margin-bottom: 0.5rem;
        animation: bounce 2s infinite;
    }

    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }

    /* File Upload Button */
    .stFileUploader button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        margin-top: 1rem !important;
    }

    .stFileUploader button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
    }

    .stFileUploader button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.3) !important;
    }

    /* File Info Display */
    .stFileUploader small {
        color: #9ca3af !important;
        font-size: 0.9rem !important;
        margin-top: 1rem !important;
        text-align: center !important;
    }

    /* Uploaded File Display */
    .stFileUploader div[data-testid="fileUploadedFileName"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        margin-top: 1rem !important;
        animation: slideInUp 0.5s ease-out !important;
    }

    @keyframes slideInUp {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    .stFileUploader div[data-testid="fileUploadedFileName"] span {
        color: #10b981 !important;
        font-weight: 600 !important;
    }

    /* Results Section */
    .results-section {
        animation: fadeInUp 1s ease-out 0.9s both;
    }

    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.04) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }

    .result-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99,102,241,0.5);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3), 0 0 20px rgba(99,102,241,0.2);
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #d946ef);
        animation: gradientMove 3s linear infinite;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1rem;
        text-align: center;
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(99,102,241,0.3);
        border-radius: 25px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        animation: scaleIn 0.8s ease-out, borderGlow 4s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }

    @keyframes scaleIn {
        from {
            transform: scale(0.9);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }

    @keyframes borderGlow {
        0%, 100% {
            border-color: rgba(99,102,241,0.3);
            box-shadow: 0 0 20px rgba(99,102,241,0.2);
        }
        50% {
            border-color: rgba(99,102,241,0.6);
            box-shadow: 0 0 30px rgba(99,102,241,0.4);
        }
    }

    .prediction-result {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 1.5rem 0;
        text-shadow: 0 0 20px currentColor;
        animation: textPulse 2s ease-in-out infinite;
    }

    @keyframes textPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    .confidence-score {
        font-size: 1.5rem;
        font-weight: 600;
        color: #60a5fa;
        margin: 1rem 0;
        animation: countUp 2s ease-out;
    }

    @keyframes countUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Enhanced Generate Report Button */
    .generate-report-section {
        display: flex;
        justify-content: center;
        margin: 3rem 0 2rem;
        animation: fadeInUp 1s ease-out 1.5s both;
    }

    .stButton > button {
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 1rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 10px 25px rgba(16, 185, 129, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        cursor: pointer !important;
        min-width: 280px !important;
        height: 60px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Button hover effect with animated background */
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.2),
            transparent
        );
        transition: left 0.5s ease;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #047857 0%, #059669 50%, #10b981 100%) !important;
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 
            0 15px 35px rgba(16, 185, 129, 0.4),
            0 5px 15px rgba(16, 185, 129, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }

    .stButton > button:active {
        transform: translateY(-1px) scale(1.01) !important;
        box-shadow: 
            0 8px 20px rgba(16, 185, 129, 0.4),
            inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }

    /* Add a subtle pulse animation */
    .stButton > button {
        animation: buttonPulse 3s ease-in-out infinite !important;
    }

    @keyframes buttonPulse {
        0%, 100% {
            box-shadow: 
                0 10px 25px rgba(16, 185, 129, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        50% {
            box-shadow: 
                0 12px 30px rgba(16, 185, 129, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.25);
        }
    }

    /* Button icon animation */
    .stButton > button::after {
        content: '📊';
        margin-left: 10px;
        font-size: 1.1em;
        animation: iconBounce 2s ease-in-out infinite;
    }

    @keyframes iconBounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-2px); }
    }

    /* Responsive button adjustments */
    @media (max-width: 768px) {
        .stButton > button {
            min-width: 240px !important;
            font-size: 1.1rem !important;
            padding: 0.9rem 2.5rem !important;
        }
    }

    /* Footer */
    .footer-info {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(30, 41, 59, 0.6) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 3rem 0 1rem;
        animation: fadeInUp 1s ease-out 1.2s both;
    }

    .footer-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #a5b4fc;
        margin-bottom: 1rem;
        text-align: center;
    }

    .footer-content {
        font-size: 1rem;
        line-height: 1.7;
        color: #d1d5db;
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
    }

    /* Spinner enhancement */
    .stSpinner {
        background: rgba(0,0,0,0.8);
        backdrop-filter: blur(10px);
        border-radius: 15px;
    }

    .stSpinner > div {
        border-color: #6366f1 transparent #8b5cf6 transparent;
        animation: spin 1s linear infinite, colorShift 2s ease-in-out infinite;
    }

    @keyframes colorShift {
        0%, 100% { border-top-color: #6366f1; border-bottom-color: #8b5cf6; }
        50% { border-top-color: #8b5cf6; border-bottom-color: #d946ef; }
    }

    /* Image hover effects */
    .stImage img {
        border-radius: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }

    .stImage:hover img {
        transform: scale(1.02);
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
    }

    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        animation: slideInLeft 0.5s ease-out;
    }

    @keyframes slideInLeft {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .logo-title {
            font-size: 2.5rem;
        }
        
        .about-section {
            margin: 1rem;
            padding: 1.5rem;
        }
        
        .prediction-card {
            margin: 1rem;
            padding: 2rem;
        }
        
        .prediction-result {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Main Container Start
# -----------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# -----------------------------
# Header Section with Enhanced Logo
# -----------------------------
st.markdown(
    """
    <div class="header-section">
        <div class="logo-title" onclick="this.style.animation='hoverPulse 0.6s ease-in-out'; setTimeout(() => this.style.animation='float 6s ease-in-out infinite', 600);">TumorX</div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# About Section
# -----------------------------
st.markdown(
    """
    <div class="about-section">
        🚀 Advanced AI-powered brain tumor detection and segmentation using cutting-edge deep learning technology.
        Upload an MRI scan for instant classification and detailed analysis with professional-grade accuracy.
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Upload Section
# -----------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📋 Drop your MRI scan here or click to browse", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    with st.spinner('🔄 Analyzing your MRI scan with advanced AI models...'):
        img_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            classifier_model = load_classifier("models/final_model.keras")
            unet_model = load_unet("models/best_unetmodel.keras")
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.stop()

        class_names = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
        try:
            pred_class, confidence = classify_image(img_path, classifier_model, class_names)
        except Exception as e:
            st.error(f"❌ Error in classification: {str(e)}")
            st.stop()

        try:
            overlay = segment_image_heatmap(unet_model, img_path)
        except Exception as e:
            st.warning(f"⚠️ Segmentation unavailable: {str(e)}")
            overlay = None

    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="result-title">🔬 Original MRI Scan</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if overlay is not None:
            st.markdown('<div class="result-title">🎯 AI Segmentation Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.image(overlay, channels="BGR", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-title">⚠️ Segmentation Unavailable</div>', unsafe_allow_html=True)
            st.info("Segmentation analysis could not be performed on this image.")

    result_color = "#ef4444" if pred_class != "No Tumor" else "#10b981"
    result_emoji = "⚠️" if pred_class != "No Tumor" else "✅"
    result_message = "Consult medical professional immediately" if pred_class != "No Tumor" else "No tumor detected - Scan appears normal"

    st.markdown(
        f"""
        <div class="prediction-card">
            <div style="font-size: 1.2rem; color: #d1d5db; margin-bottom: 1.5rem; font-weight: 600;">
                🩺 AI DIAGNOSTIC ANALYSIS
            </div>
            <div class="prediction-result" style="color: {result_color};">
                {result_emoji} {pred_class.upper()}
            </div>
            <div class="confidence-score">
                Model Confidence: {confidence:.1f}%
            </div>
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(0,0,0,0.3); border-radius: 10px; font-size: 1rem; color: #e5e7eb;">
                <strong>{result_message}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ---- Report Generation ----
from utils.report_generator import generate_pdf_report
import cv2, os

st.markdown('<div class="generate-report-section">', unsafe_allow_html=True)
if st.button("📑 Generate Report"):
    if not img_path:   # covers None or empty
        st.error("❌ Can't generate report — no MRI uploaded.")
    elif not pred_class or confidence is None:
        st.warning("⚠️ Can't generate report — please run classification first.")
    else:
        overlay_path = None
        if overlay is not None:
            overlay_path = "temp_overlay.png"
            cv2.imwrite(overlay_path, overlay)

        pdf_buffer = generate_pdf_report(img_path, overlay_path, pred_class, confidence)

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_buffer,
            file_name="TumorX_Report.pdf",
            mime="application/pdf"
        )
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footer-info">
        <div class="footer-title">🧬 About TumorX AI Platform</div>
        <div class="footer-content">
            TumorX represents the cutting edge of medical AI technology...
            <br><br>
            <strong>🏥 MEDICAL DISCLAIMER:</strong> This AI system is designed for research and educational purposes only.
            <br><br>
            <em>⚡ Powered by TensorFlow • Developed with medical imaging expertise • Built for the future of healthcare</em>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)