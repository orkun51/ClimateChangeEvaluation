import os
import sys
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 📦 Ensure model path is correctly resolved when bundled with PyInstaller
if getattr(sys, 'frozen', False):  # If bundled with PyInstaller
    model_path = os.path.join(sys._MEIPASS, "quantized_model.tflite")
else:
    model_path = "quantized_model.tflite"

# 📥 Load Model Function
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

# 🧠 Category Descriptions
category_descriptions = {
    0: "Carbon Emission (Factory and Cars): May include air pollution, smoke from industrial chimneys, or heavy traffic.",
    1: "Trees and Nature: May include green areas, clean air, and nature-related elements.",
    2: "Sun and Temperature Change: May show the effect of the sun, heatwaves, or melting glaciers.",
    3: "Air Pollution: May include exhaust fumes, dirty air, and other pollution elements.",
    4: "Drought and Water Scarcity: May show dry soil, people struggling with water shortage, or shrinking water sources.",
    5: "Melting Glaciers: May illustrate melting glaciers and rising sea levels due to global warming.",
    6: "Effects of Climate Change: May include natural disasters, extreme weather events, and ecological destruction."
}

# 🎨 Drawing Suggestions for Students
drawing_suggestions = {
    0: "You can highlight pollution by emphasizing smoke from factory chimneys or exhaust fumes.",
    1: "You can showcase nature’s beauty by adding more trees, flowers, or clean water sources.",
    2: "You can draw more prominent sun rays to show the heat effect. Adding sweating people can also be effective!",
    3: "You can depict air pollution with a smoky cityscape or people wearing masks.",
    4: "You can illustrate the impact of drought with dried lakes, cracked soil, or affected plants.",
    5: "You can emphasize global warming by drawing melting ice and falling ice blocks into the water.",
    6: "You can depict disasters caused by climate change, such as hurricanes, storms, or forest fires."
}

# 🖌 App Title
st.markdown("<h1 style='text-align: center;'>🎨 Climate Change Drawing Evaluation App</h1>", unsafe_allow_html=True)
st.write("Upload a drawing to see the evaluation results.")

# 📂 File Uploader
uploaded_file = st.file_uploader("📤 Upload your drawing", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # 🖼 Show Uploaded Image
        image = Image.open(uploaded_file)
        st.image(image, caption="🖌 Uploaded Drawing")

        # 📌 Image Preprocessing
        img = image.convert("RGB").resize((224, 224))  # Resize to model input size
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # 📊 Get Prediction
        if model is not None:
            input_details = model.get_input_details()
            output_details = model.get_output_details()

            model.set_tensor(input_details[0]['index'], img_array)
            model.invoke()
            prediction = model.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(prediction)
            confidence_score = np.max(prediction) * 100

            # 🔍 Display Results
            if predicted_class in category_descriptions:
                st.success(f"🔍 **Predicted Category:** {category_descriptions[predicted_class]}")
                st.info(f"🎯 **Confidence Score:** {confidence_score:.2f}%")
                st.markdown(f"✍️ **Drawing Suggestion:** {drawing_suggestions[predicted_class]}")
            else:
                st.warning("⚠️ The model could not make a prediction, please try again!")
        else:
            st.error("The model could not be loaded, please try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
