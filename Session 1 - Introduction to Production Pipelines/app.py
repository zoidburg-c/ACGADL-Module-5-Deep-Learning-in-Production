
"""
Simple Image Classification API
================================
A minimal FastAPI application that serves a pre-trained ResNet18 model.
This is the simplest form of "model deployment" - a local API server.

To run this server:
    uvicorn app:app --host 0.0.0.0 --port 8000

To test it:
    curl -X POST http://localhost:8000/predict          -F "file=@your_image.jpg"
"""

import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import requests
import io

# ============================================================
# 1. INITIALIZE THE APPLICATION
# ============================================================
app = FastAPI(
    title="Image Classification API",
    description="A simple API that classifies images using ResNet18",
    version="1.0.0"
)

# ============================================================
# 2. LOAD THE MODEL (runs once when the server starts)
# ============================================================
# In production, you would load the saved model file:
#   model.load_state_dict(torch.load("saved_models/resnet18_imagenet.pth"))
# For simplicity, we load the pre-trained weights directly here.

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()  # Set to evaluation mode

# Load class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
imagenet_labels = requests.get(LABELS_URL).json()

# Define preprocessing (must match what the model was trained with)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

print(f"Model loaded. Ready to serve predictions!")

# ============================================================
# 3. DEFINE API ENDPOINTS
# ============================================================

@app.get("/")
def read_root():
    """Health check endpoint - confirms the server is running."""
    return {
        "status": "healthy",
        "model": "ResNet18",
        "description": "Image Classification API"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded image and return the top-5 predictions.

    This is the main prediction endpoint. It follows the same
    load -> preprocess -> infer -> postprocess pattern we built
    in Exercise 2.
    """
    # Read the uploaded image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    input_tensor = preprocess(img).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

    # Postprocess
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_indices = torch.topk(probabilities, 5)

    # Format the results as a JSON-friendly response
    predictions = [
        {
            "rank": i + 1,
            "label": imagenet_labels[idx.item()],
            "probability": round(prob.item(), 4)
        }
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
    ]

    return {
        "filename": file.filename,
        "predictions": predictions
    }
