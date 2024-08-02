from functools import lru_cache

import streamlit as st

from .base import BaseAIModel
from .yolov8 import Yolov8


@st.cache_resource
def get_ai_model(name: str, model_params: dict) -> BaseAIModel | None:
    factory: dict[str, BaseAIModel] = {
        "yolov8": Yolov8(**model_params),
    }

    return factory.get(name, None)
