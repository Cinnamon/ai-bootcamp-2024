from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from ultralytics.models import YOLO
from ultralytics.engine.results import Results
from loguru import logger

from .base import BaseAIModel
from shared.schemas import Parameters, ModelOutput


class Yolov8(BaseAIModel):
    def __init__(self, model_name: str, device: Literal["cpu", "cuda"] = "cuda"):
        self._model = YOLO(model_name, task="detect")

        if device in ["cuda"] and torch.cuda.is_available():
            self._device = torch.device(device)
        else:
            self._device = torch.device("cpu")

        self._model.to(self._device)

    @staticmethod
    def get_default() -> dict:
        return {
            "augment": False,
            "agnostic_nms": False,
            "imgsz": 640,
            "iou": 0.5,
            "conf": 0.01,
            "verbose": False
        }

    def process(
        self,
        image_in: Path | str | np.ndarray,
        *args,
        **kwargs,
    ) -> Path:
        if type(image_in) is [str, Path]:
            image_in = cv2.imread(image_in, cv2.IMREAD_COLOR)

        default_params: dict = self.get_default()
        if kwargs.get("params", None):
            params: Parameters = kwargs["params"]

            # Update
            default_params["augment"] = params.augment
            default_params["agnostic_nms"] = params.agnostic_nms
            default_params["imgsz"] = params.image_size
            default_params["iou"] = params.min_iou
            default_params["conf"] = params.min_confident_score

        logger.debug(f"Run with config: {default_params}")

        results: Results = self._model(image_in, **default_params)
        result = results[0].cpu().numpy()

        model_out_params = {
            "xyxysc": result.boxes.data
        }

        return ModelOutput(**model_out_params)
