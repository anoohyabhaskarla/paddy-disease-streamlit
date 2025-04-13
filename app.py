import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import gdown
import os

model_path = "paddy_model_fusion_final.h5"
if not os.path.exists(model_path):
    # ⬇️ Download from your Google Drive (replace FILE_ID below)
    gdown.download("https://drive.google.com/uc?id=1j1zEA-e8jNm2LDek6K4MpJ56eajQvVIB", model_path, quiet=False)

model = tf.keras.models.load_model(model_path)

# ✅ Load the model
model = load_model(model_path)

# 🗺️ Class mapping
class_map = {
    0: "bacterial_leaf_blight",
    1: "bacterial_leaf_streak",
    2: "bacterial_panicle_blight",
    3: "black_stem_borer",
    4: "blast",
    5: "brown_spot",
    6: "downy_mildew",
    7: "hispa",
    8: "leaf_roller",
    9: "normal",
    10: "tungro",
    11: "white_stem_borer",
    12: "yellow_stem_borer"
}

st.set_page_config(page_title="🌾 Paddy Disease Classifier", layout="centered")
st.title("🌾 Paddy Disease Detection Dashboard")
st.markdown("Upload a paddy leaf image and input weather data to detect disease.")

# Weather inputs
temp = st.number_input("🌡️ Temperature (°C)", 10.0, 50.0, 28.0)
humidity = st.number_input("💧 Humidity (%)", 10.0, 100.0, 65.0)

# Image upload
uploaded_file = st.file_uploader("Upload paddy leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)
    weather_input = np.array([[temp, humidity]])

    prediction = model.predict([img_input, weather_input])
    pred_class = np.argmax(prediction)
    confidence = prediction[0][pred_class]

    st.image(img_rgb, caption="Uploaded Leaf Image", use_column_width=True)
    st.markdown(f"### 🧠 Prediction: **{class_map[pred_class]}**")
    st.markdown(f"Confidence Score: `{confidence:.2f}`")

    df = pd.DataFrame(prediction[0], index=[class_map[i] for i in range(13)], columns=["Confidence"])
    st.bar_chart(df)
