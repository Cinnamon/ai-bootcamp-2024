from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Base:
    def to_dict(self):
        return asdict(self)


@dataclass
class Parameters(Base):
    augment: bool
    agnostic_nms: bool
    image_size: int
    min_iou: float
    min_confident_score: float


@dataclass
class ModelInput(Base):
    upload_image: str
    params: Parameters


@dataclass
class ModelOutput(Base):
    xyxysc: np.ndarray  # x_min, y_min, x_max, y_max, score, class

    def __len__(self):
        return len(self.xyxysc)

    def __getitem__(self, item_id: int) -> np.ndarray:
        return self.xyxysc[item_id]

    def count(self) -> dict[int, int]:
        cls_dict: dict[int, int] = {}
        for c in self.xyxysc[:, -1]:
            c = int(c)
            if c not in cls_dict:
                cls_dict[c] = 0
            cls_dict[c] += 1

        return cls_dict

    def to_dict(self) -> dict[int, list]:
        result_dict: dict[int, list] = {}
        for i, elem in enumerate(self.xyxysc):
            x_min, y_min, x_max, y_max = map(int, elem[:4])
            score = float(elem[-2])
            cls = int(elem[-1])

            result_dict[i] = [
                x_min, y_min, x_max, y_max, score, cls
            ]
        return result_dict


@dataclass
class EditedOutput(Base):
    cls: int
