import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image
from pathlib import Path

# PAGE SETTINGS

st.set_page_config(
    page_title="SignVision AI",
    page_icon="🤟",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "signvision_model.joblib"

# CUSTOM DESIGN

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: white;
}
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #60a5fa;
}
.subtitle {
    color: #cbd5e1;
    font-size: 16px;
}
.prediction {
    background: #172235;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}
.prediction h1 {
    color: #60a5fa;
    font-size: 65px;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def create_hand_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.65
    )

# FEATURE EXTRACTION

def extract_features(hand_landmarks):
    points = np.array(
        [[p.x, p.y, p.z] for p in hand_landmarks.landmark],
        dtype=np.float32
    )

    points -= points[0]

    scale = np.max(np.linalg.norm(points[:, :2], axis=1))

    if scale > 1e-6:
        points /= scale

    return points.flatten()

# HEADER

st.markdown(
    '<div class="main-title">🤟 SIGNVISION AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Indian Sign Language Alphabet Recognition | A–Z</div>',
    unsafe_allow_html=True
)

st.divider()

# LOAD TRAINED MODEL

try:
    model = load_model()
    st.success("Trained model loaded successfully!")
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.info("Check whether models/signvision_model.joblib exists.")
    st.stop()

# CAMERA INPUT

left, right = st.columns([1.4, 1])

with left:
    st.subheader("📷 Capture Your Hand Gesture")

    captured_image = st.camera_input(
        "Show one alphabet hand gesture and capture"
    )

with right:
    st.subheader("🔍 Recognition Result")

    if captured_image is None:
        st.info("Capture a hand gesture to see the prediction.")

    else:
        image = Image.open(captured_image).convert("RGB")

        rgb_image = np.fliplr(np.array(image)).copy()

        detector = create_hand_detector()
        result = detector.process(rgb_image)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            features = extract_features(hand)

            probabilities = model.predict_proba([features])[0]

            classes = list(model.classes_)

            alphabet_indices = [
                i for i, label in enumerate(classes)
                if str(label) in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            ]

            if not alphabet_indices:
                st.error("No alphabet classes found in the model.")
                st.stop()

            best_index = max(
                alphabet_indices,
                key=lambda i: probabilities[i]
            )

            prediction = str(classes[best_index])
            confidence = float(probabilities[best_index])

            annotated = cv2.cvtColor(
                rgb_image.copy(), cv2.COLOR_RGB2BGR
            )

            mp.solutions.drawing_utils.draw_landmarks(
                annotated,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS
            )

            annotated = cv2.cvtColor(
                annotated, cv2.COLOR_BGR2RGB
            )

            st.image(
                annotated,
                caption="Detected Hand Landmarks",
                use_container_width=True
            )

            if confidence >= 0.60:
                st.markdown(
                    f'<div class="prediction">'
                    f'<p>Predicted Alphabet</p>'
                    f'<h1>{prediction}</h1>'
                    f'<p>Confidence: {confidence * 100:.1f}%</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning(
                    f"Uncertain prediction: {prediction} "
                    f"({confidence * 100:.1f}%). "
                    "Try a clearer hand gesture."
                )

        else:
            st.warning(
                "No hand detected. Keep your hand clearly visible."
            )

# FOOTER

st.divider()

st.caption(
    "SignVision AI | Static Hand Gesture Alphabet Recognition | Educational Prototype"
)