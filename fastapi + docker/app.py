from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# Path to your saved model folder
MODEL_DIR = "brain_tumor_vit_model"

# Load model & processor
model = ViTForImageClassification.from_pretrained(MODEL_DIR)
processor = ViTImageProcessor.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Mapping of IDs to labels
id2label = {0: "glioma", 1: "meningioma", 2: "no_tumor", 3: "pituitary"}

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Brain Tumor Classifier API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")  # ensure RGB
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_id = logits.argmax(-1).item()
    
    return {"prediction": id2label[predicted_id]}
