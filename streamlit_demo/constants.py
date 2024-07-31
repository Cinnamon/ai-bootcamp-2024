import os
from pathlib import Path

APP_DATA_DIR = Path(__file__).parent / "app_data"
os.makedirs(APP_DATA_DIR, exist_ok=True)

FEEDBACK_DIR = APP_DATA_DIR / "feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

FEEDBACK_SQL_PATH = f"sqlite:///{FEEDBACK_DIR / 'feedback.sql'}"

YOLO_OPTIONS = [
    "yolov8s.pt",
    "yolov8n.pt"
]

YOLO_SUPPORTED_EXTENSIONS = ["jpg", "png", "jpeg"]

USER_DATA_DIR = APP_DATA_DIR / "user_data" / "images"
os.makedirs(USER_DATA_DIR, exist_ok=True)

AI_MODEL_CONFIGS = {
    "yolov8": {
        "model_name": "yolov8s.pt",
        "device": "cuda"
    }
}
AI_MODEL = "yolov8"

CLASSES = ['Person', 'Bicycle', 'Car', 'Motorcycle', 'Airplane', 'Bus', 'Train', 'Truck', 'Boat', 'Traffic light',
           'Fire hydrant', 'Stop sign', 'Parking meter', 'Bench', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow',
           'Elephant', 'Bear', 'Zebra', 'Giraffe', 'Backpack', 'Umbrella', 'Handbag', 'Tie', 'Suitcase', 'Frisbee',
           'Skis', 'Snowboard', 'Sports ball', 'Kite', 'Baseball bat', 'Baseball glove', 'Skateboard', 'Surfboard',
           'Tennis racket', 'Bottle', 'Wine glass', 'Cup', 'Fork', 'Knife', 'Spoon', 'Bowl', 'Banana', 'Apple',
           'Sandwich', 'Orange', 'Broccoli', 'Carrot', 'Hot dog', 'Pizza', 'Donut', 'Cake', 'Chair', 'Couch',
           'Potted plant', 'Bed', 'Dining table', 'Toilet', 'Tv', 'Laptop', 'Mouse', 'Remote', 'Keyboard', 'Cell phone',
           'Microwave', 'Oven', 'Toaster', 'Sink', 'Refrigerator', 'Book', 'Clock', 'Vase', 'Scissors', 'Teddy bear',
           'Hair drier', 'Toothbrush']
