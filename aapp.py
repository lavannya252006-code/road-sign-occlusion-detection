import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Road Sign Occlusion Detector",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Road Sign Occlusion Detection")
st.write(
    "An AI-powered system that classifies traffic signs "
    "and detects whether they are partially blocked."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "road_sign_occlusion_model.keras"
    )

model = load_model()

st.subheader("Upload a Road Sign Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    resized = image.resize((32, 32))
    img_array = np.array(resized, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analysing road sign..."):
        sign_output, occ_output = model.predict(
            img_array, verbose=0
        )

    predicted_sign = int(np.argmax(sign_output[0]))
    confidence = float(np.max(sign_output[0])) * 100
    occlusion_probability = float(occ_output[0][0])

    with col2:
        st.subheader("AI Prediction")
        st.metric("Traffic Sign Class", predicted_sign)
        st.metric("Classification Confidence", f"{confidence:.2f}%")

        if occlusion_probability >= 0.5:
            st.error("⚠️ Occlusion Detected")
        else:
            st.success("✅ No Occlusion Detected")

        st.metric(
            "Occlusion Probability",
            f"{occlusion_probability * 100:.2f}%"
        )

st.divider()

st.subheader("About This Project")
st.write("""
This project uses a custom Convolutional Neural Network
with two output heads.

1. Traffic Sign Classification: Identifies one of 43 classes.
2. Occlusion Detection: Detects whether a road sign is blocked.

Synthetic occlusions are used during training to simulate
partially obstructed traffic signs.
""")
