from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseAIModel(ABC):
    @abstractmethod
    def process(self, image_in: Path | str | np.ndarray, *args, **kwargs) -> Path:
        ...
