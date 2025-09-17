
import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import os

# Set path to your model folder
MODEL_DIR = os.path.join(os.path.dirname(__file__), "brain_tumor_vit_model")

# Load model and processor
@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    model.eval()
    return model, processor

model, processor = load_model()

# Label mapping (must match training)
id2label = {0: "glioma", 1: "meningioma", 2: "no_tumor", 3: "pituitary"}

# Streamlit UI
st.title("🧠 Brain Tumor Classifier (ViT)")
st.write("Upload an MRI image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()
        predicted_label = id2label[predicted_class_id]

    st.subheader(f"🔍 Prediction: **{predicted_label}**")


