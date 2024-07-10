import io
import os

import gradio as gr
import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


def predict_imagenet_confidences_via_request(image: Image.Image) -> dict:
    """[Send a POST request to the fastapi backend]
    Receive an image and predict confidences for ImageNet classes.

    Args:
        image: Image to predict confidences for.

    Returns:
        dict: Dictionary of 1000 classes in ImageNet and their confidence scores (float).
    """
    # Get the prediction endpoint
    url = os.getenv("PREDICT_ENDPOINT", "http://127.0.0.1:8000/predict/")

    # Convert PIL Image to bytes to send via requests POST
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Send POST request to predict endpoint
    files = {"file": img_byte_arr}
    response = requests.post(url, files=files)
    return response.json()


def main():
    # Gradio front-end interface
    gr_interface = gr.Interface(
        fn=predict_imagenet_confidences_via_request,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=5),
        examples=["assets/cat.jpg", "assets/lion.jpg"],
    )

    # Launch the web server
    gr_interface.launch(
        server_name=os.getenv("GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_PORT", 8080)),
    )


if __name__ == "__main__":
    main()
