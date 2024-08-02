from typing import Dict, List

import numpy as np
from loguru import logger

from shared.schemas import ModelOutput


class DataProcessor:
    def __init__(self, filled_color: str = "rgba(0, 151, 255, 0.25)"):
        self._filled_color = filled_color

    def prepare_canvas_data(
        self,
        data: ModelOutput,
    ):
        canvas_rects = []

        for i, box in enumerate(data.xyxysc):
            box: np.ndarray

            canvas_rect = self.construct_canvas_group(
                box[:4].astype(int),
                True,
                self._filled_color
            )
            canvas_rects += [canvas_rect]

        payload = {"version": "4.4.0", "objects": canvas_rects}
        return payload

    @staticmethod
    def get_location_from_canvas_rect(canvas_rect: dict) -> tuple:
        x2 = round(canvas_rect['left'] + (canvas_rect['width'] * canvas_rect['scaleX']))
        y2 = round(canvas_rect['top'] + (canvas_rect['height'] * canvas_rect['scaleY']))
        x1 = round(canvas_rect['left'])
        y1 = round(canvas_rect['top'])

        return x1, y1, x2, y2

    @staticmethod
    def construct_canvas_group(
        box: np.ndarray,
        visibility: bool,
        filled_color: str
    ):
        x_min, y_min, x_max, y_max = map(int, box)

        canvas_rect = {
            "type": "rect",
            "version": "4.4.0",
            "originX": "left",
            "originY": "top",
            "left": x_min,
            "top": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min,
            "fill": filled_color,
            "stroke": "rgba(0, 50, 255, 0.7)",
            "strokeWidth": 2,
            "strokeDashArray": None,
            "strokeLineCap": "butt",
            "strokeDashOffset": 0,
            "strokeLineJoin": "miter",
            "strokeUniform": True,
            "strokeMiterLimit": 4,
            "scaleX": 1,
            "scaleY": 1,
            "angle": 0,
            "flipX": False,
            "flipY": False,
            "opacity": 1,
            "shadow": None,
            "visible": visibility,
            "backgroundColor": "",
            "fillRule": "nonzero",
            "paintFirst": "fill",
            "globalCompositeOperation": "source-over",
            "skewX": 0,
            "skewY": 0,
            "rx": 0,
            "ry": 0,
        }

        return canvas_rect

    def prepare_rect_data(
        self,
        canvas_data,
        regions_in: ModelOutput,
        select_index: int = -1
    ):
        regions = []
        n_in = len(regions_in.xyxysc)
        n_out = len(canvas_data["objects"])

        if n_in <= n_out:
            # For adding & modify
            for i, canvas_rect in enumerate(canvas_data["objects"]):
                x_min, y_min, x_max, y_max = self.get_location_from_canvas_rect(canvas_rect)
                if i < n_in:
                    # modifying: update location
                    old_region = regions_in.xyxysc[i]
                    old_region[:4] = [x_min, y_min, x_max, y_max]
                    regions += [old_region]
                else:
                    # adding
                    region = np.array([x_min, y_min, x_max, y_max, 0.0, -1])
                    regions += [region]
        elif n_in > n_out:
            """
            For deleting
            """
            regions = [r for i, r in enumerate(regions_in.xyxysc) if i != select_index]
            select_index = -1

        xyxysc = np.array(regions)

        return ModelOutput(xyxysc=xyxysc), select_index
