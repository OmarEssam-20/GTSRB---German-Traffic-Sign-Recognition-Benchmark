import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered"
)

SIGN_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("finetuned_model_best.keras")

model = load_model()

st.title("🚦 German Traffic Sign Recognition")
st.write("Upload a traffic sign image and the model will classify it into one of 43 GTSRB classes.")
st.markdown("**Model:** EfficientNetB0 Fine-Tuned on GTSRB")
st.divider()

uploaded_file = st.file_uploader(
    "Upload a traffic sign image",
    type=["jpg", "jpeg", "png", "ppm"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Classifying..."):
        prediction = model.predict(img_array, verbose=0)[0]

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    top_3 = np.argsort(prediction)[::-1][:3]

    with col2:
        st.subheader("Prediction Result")
        st.success(SIGN_NAMES[predicted_class])
        st.metric("Confidence", f"{confidence:.2f}%")

        st.divider()
        st.subheader("Top 3 Predictions")

        for i, class_id in enumerate(top_3, start=1):
            prob = float(prediction[class_id]) * 100
            st.write(f"{i}. {SIGN_NAMES[int(class_id)]} — {prob:.2f}%")
            st.progress(min(int(prob), 100))

else:
    st.info("Please upload a traffic sign image.")

st.divider()
st.caption("GTSRB Dataset · EfficientNetB0 Fine-Tuned · Built with TensorFlow & Streamlit")
