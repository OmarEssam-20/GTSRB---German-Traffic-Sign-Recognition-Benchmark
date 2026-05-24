
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered"
)

SIGN_NAMES = {
    "0":"Speed limit (20km/h)",    "1":"Speed limit (30km/h)",
    "2":"Speed limit (50km/h)",    "3":"Speed limit (60km/h)",
    "4":"Speed limit (70km/h)",    "5":"Speed limit (80km/h)",
    "6":"End of speed limit (80km/h)", "7":"Speed limit (100km/h)",
    "8":"Speed limit (120km/h)",   "9":"No passing",
    "10":"No passing (>3.5 tons)", "11":"Right-of-way at next intersection",
    "12":"Priority road",          "13":"Yield",
    "14":"Stop",                   "15":"No vehicles",
    "16":"No vehicles (>3.5 tons)","17":"No entry",
    "18":"General caution",        "19":"Dangerous curve (left)",
    "20":"Dangerous curve (right)","21":"Double curve",
    "22":"Bumpy road",             "23":"Slippery road",
    "24":"Road narrows (right)",   "25":"Road work",
    "26":"Traffic signals",        "27":"Pedestrians",
    "28":"Children crossing",      "29":"Bicycles crossing",
    "30":"Beware of ice/snow",     "31":"Wild animals crossing",
    "32":"End all speed/passing limits", "33":"Turn right ahead",
    "34":"Turn left ahead",        "35":"Ahead only",
    "36":"Go straight or right",   "37":"Go straight or left",
    "38":"Keep right",             "39":"Keep left",
    "40":"Roundabout mandatory",   "41":"End of no passing",
    "42":"End no passing (>3.5 tons)"
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("finetuned_model_best.keras")

@st.cache_data
def load_class_names():
    with open("class_names.json") as f:
        return json.load(f)

model       = load_model()
class_names = load_class_names()

# ── UI ───────────────────────────────────────────────────────
st.title("🚦 German Traffic Sign Classifier")
st.markdown("Upload a traffic sign image — the model will identify it from 43 possible classes.")
st.markdown("**Model:** EfficientNetB0 Fine-Tuned on GTSRB")
st.divider()

uploaded_file = st.file_uploader(
    "Upload a traffic sign image",
    type=["jpg", "jpeg", "png", "ppm"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array   = np.array(img_resized, dtype="float32")
    img_array   = np.expand_dims(img_array, axis=0)

    with st.spinner("Classifying..."):
        preds    = model.predict(img_array, verbose=0)[0]
        top3_idx = np.argsort(preds)[::-1][:3]

    with col2:
        st.subheader("Result")
        pred_class = str(top3_idx[0])
        confidence = preds[top3_idx[0]] * 100
        sign_name  = SIGN_NAMES.get(pred_class, f"Class {pred_class}")

        st.markdown(f"### {sign_name}")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.divider()
        st.subheader("Top 3 Predictions")
        for rank, idx in enumerate(top3_idx, 1):
            name = SIGN_NAMES.get(str(idx), f"Class {idx}")
            prob = preds[idx] * 100
            st.progress(int(prob), text=f"{rank}. {name} — {prob:.1f}%")

st.divider()
st.caption("GTSRB Dataset · EfficientNetB0 Fine-Tuned · Built with TensorFlow & Streamlit")
