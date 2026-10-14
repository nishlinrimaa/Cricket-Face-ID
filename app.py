import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "cricket_face_model.keras"
LABELS_PATH = ROOT / "models" / "labels.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Cricket Face ID", page_icon="🏏", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#071b3a 0%,#0b3d5c 48%,#087e70 100%); color:#fff;}
.block-container {padding-top:2rem; max-width:1150px;}
.hero {padding:28px 30px;border:1px solid #ffffff30;border-radius:24px;
background:linear-gradient(120deg,#ffffff18,#ffffff08); margin-bottom:20px;}
.hero h1 {font-size:2.7rem;color:#fff;margin:0;}
.hero p {color:#d9f7f3;font-size:1.05rem;}
.pill {display:inline-block;background:#ffcc33;color:#12233d;padding:6px 12px;
border-radius:999px;font-weight:800;font-size:.8rem;}
.panel {padding:20px;border-radius:20px;background:#ffffff12;border:1px solid #ffffff25;}
.small {color:#d7e8f4;font-size:.9rem;}
</style>
<div class="hero">
<span class="pill">AI • COMPUTER VISION</span>
<h1>🏏 Cricket Face ID</h1>
<p>Upload a photo and let the model predict which trained Indian cricketer it most resembles.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None
    model = load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, labels

def detect_and_crop(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces):
        # Use the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.18 * max(w, h))
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(rgb.shape[1], x+w+pad), min(rgb.shape[0], y+h+pad)
        return Image.fromarray(rgb[y1:y2, x1:x2]), True
    return Image.fromarray(rgb), False

model, labels = load_artifacts()

left, right = st.columns([1.05, 1], gap="large")
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📸 Upload a player image")
    uploaded = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])
    st.caption("Tip: use a clear, front-facing photo with one person and good lighting.")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Your uploaded image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("✨ Prediction")
    if model is None:
        st.warning("The trained model isn't available yet.")
        st.write("1. Add player photos into `dataset/PlayerName/` folders.")
        st.write("2. Run `python train_model.py` in the VS Code terminal.")
        st.write("3. Refresh this page.")
        st.info("Example: dataset/Virat_Kohli/photo1.jpg")
    elif uploaded:
        try:
            face, found = detect_and_crop(image)
            st.image(face, caption="Face crop used by the model" if found else "No face detected — using full image", width=260)
            arr = np.array(face.resize(IMG_SIZE), dtype=np.float32)
            arr = preprocess_input(arr)
            probabilities = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
            best = int(np.argmax(probabilities))
            confidence = float(probabilities[best]) * 100
            player = labels[str(best)]
            st.markdown(f"### 🏆 {player.replace('_', ' ')}")
            st.metric("Model confidence", f"{confidence:.1f}%")
            st.progress(min(max(confidence / 100, 0.0), 1.0))
            if confidence < 55:
                st.warning("Low confidence. Try a clearer image or add more training photos.")
            st.caption("This is a machine-learning prediction, not a verified identity.")
            with st.expander("See all class scores"):
                ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
                for idx, score in ranked:
                    st.write(f"**{labels[str(idx)].replace('_', ' ')}** — {score*100:.1f}%")
                    st.progress(float(score))
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")
    else:
        st.write("Your result will appear here after you upload an image.")
        st.markdown("**What this project demonstrates**")
        st.write("• Face detection with OpenCV")
        st.write("• Transfer learning with MobileNetV2")
        st.write("• Multi-class image classification")
        st.write("• Interactive Streamlit interface")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="small">Built for educational use • Performance depends on dataset quality, image diversity, and training size.</p>', unsafe_allow_html=True)
