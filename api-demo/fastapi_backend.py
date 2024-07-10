import os
from typing import Union

import torch
import uvicorn
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import ViTImageProcessor, ViTForImageClassification

load_dotenv()
app = FastAPI(title="fastapi-classification-demo")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")


def predict_imagenet_confidences(image: Union[Image.Image, str]) -> dict:
    """[A normal python function]
    Receive an image and predict confidences for ImageNet classes.

    Args:
        image (Union[Image.Image, str]): Image to predict confidences for.

    Returns:
        dict: Dictionary of 1000 classes in ImageNet and their confidence scores (float).
    """
    if isinstance(image, str):
        image = Image.open(image)
    # Get the model and processor
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    # Get confidence scores for all 1000 classes
    logits = outputs.logits
    confidences_id = torch.nn.functional.softmax(logits[0], dim=0)
    confidences_labels = {
        model.config.id2label[i]: float(confidences_id[i]) for i in range(1000)
    }

    return confidences_labels


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "name": "Class of Cinnamon AI Bootcamp 2023"}
    )


@app.post("/predict/")
async def predict(file: UploadFile):
    """[FastAPI endpoint]
    Predict confidences for ImageNet classes from an uploaded image.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        dict: Dictionary of 1000 classes in ImageNet and their confidence scores (float).
    """
    file_obj = file.file
    image = Image.open(file_obj)
    confidences = predict_imagenet_confidences(image)
    return confidences


def main():
    # Run web server with uvicorn
    uvicorn.run(
        "fastapi_backend:app",
        host=os.getenv("FASTAPI_HOST", "127.0.0.1"),
        port=int(os.getenv("FASTAPI_PORT", 8000)),
        # reload=True,  # Uncomment this for debug
    )


if __name__ == "__main__":
    main()
