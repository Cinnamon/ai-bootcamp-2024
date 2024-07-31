import os
from functools import lru_cache
from typing import Literal

import streamlit as st
import streamlit.components.v1 as components
import streamlit.elements.image as st_image
from PIL import Image

from .processor import DataProcessor
from shared.schemas import ModelOutput


_RELEASE = True  # on packaging, pass this to True


if not _RELEASE:
    _component_func = components.declare_component(
        "st_sparrow_labeling",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("st_sparrow_labeling", path=build_dir)


@lru_cache(1)
def get_background_image_bytes(image_path: str):
    background_image = Image.open(image_path)
    width, height = background_image.size

    format = st_image._validate_image_format_string(background_image, "PNG")
    image_data = _pil_to_bytes(background_image, format)

    return image_data, width


def check_image_url(url):
    import requests
    try:
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        return False


def _pil_to_bytes(
    image: st_image.PILImage,
    format: st_image.ImageFormat = "JPEG",
    quality: int = 100,
) -> bytes:
    import io

    """Convert a PIL image to bytes."""
    tmp = io.BytesIO()

    # User must have specified JPEG, so we must convert it
    if format == "JPEG" and st_image._image_may_have_alpha_channel(image):
        image = image.convert("RGB")

    image.save(tmp, format=format, quality=quality)

    return tmp.getvalue()


def st_annotate_tool(
    regions: ModelOutput,
    fill_color: str = "#eee",
    stroke_width: int = 20,
    stroke_color: str = "black",
    background_image: Image = None,
    drawing_mode: Literal["transform", "rect"] = "transform",
    point_display_radius: int = 3,
    canvas_height: int = 600,
    canvas_width: int = 600,
    key=None,
) -> tuple[ModelOutput, int]:
    """Create a drawing canvas in Streamlit app. Retrieve the RGBA image data into a 4D numpy array (r, g, b, alpha)
    on mouse up event.

    Parameters
    ----------
    regions: ModelOutput
        Output from ai model, list of (x_min, y_min, x_max, y_max, score, cls)
    fill_color: str
        Color of fill for Rect in CSS color property. Defaults to "#eee".
    stroke_width: str
        Width of drawing brush in CSS color property. Defaults to 20.
    stroke_color: str
        Color of drawing brush in hex. Defaults to "black".
    background_image: Image
        Pillow Image to display behind canvas.
        Automatically resized to canvas dimensions.
        Being behind the canvas, it is not sent back to Streamlit on mouse event.
    drawing_mode: {'freedraw', 'transform', 'line', 'rect', 'circle', 'point', 'polygon'}
        Enable free drawing when "freedraw", object manipulation when "transform", "line", "rect", "circle", "point", "polygon".
        Defaults to "freedraw".
    point_display_radius: int
        The radius to use when displaying point objects. Defaults to 3.
    canvas_height: int
        Height of canvas in pixels. Defaults to 600.
    canvas_width: int
        Width of canvas in pixels. Defaults to 600.
    key: str
        An optional string to use as the unique key for the widget.
        Assign a key so the component is not remount every time the script is rerun.

    Returns
    -------
    new_model_output: contains edited bounding boxes
    selected_index: select index
    """
    # Resize background_image to canvas dimensions by default
    # Then override background_color
    if canvas_height == 0 or canvas_width == 0:
        return regions, -1

    background_image_url = None
    if background_image:
        image_bytes, width = get_background_image_bytes(background_image)

        # Reduce network traffic and cache when switch another configure,
        # use streamlit in-mem filemanager to convert image to URL
        background_image_url = st_image.image_to_url(
            image_bytes, width, True, "RGB", "PNG",
            f"drawable-canvas-bg-{background_image}-{key}"
        )
        background_image_url = st._config.get_option("server.baseUrlPath") + background_image_url

    data_processor = DataProcessor()
    canvas_rects = data_processor.prepare_canvas_data(regions)

    component_value = _component_func(
        fillColor=fill_color,
        strokeWidth=stroke_width,
        strokeColor=stroke_color,
        backgroundImageURL=background_image_url,
        canvasHeight=canvas_height,
        canvasWidth=canvas_width,
        drawingMode=drawing_mode,
        initialDrawing=canvas_rects,
        displayRadius=point_display_radius,
        key=f"{key}_canvas",
        default=None,
        realtimeUpdateStreamlit=True,
        showingMode="All",
        displayToolbar=False
    )

    if component_value is None:
        return regions, -1

    select_index = component_value.get('selectIndex', -1)
    new_model_output, select_index = data_processor.prepare_rect_data(
        component_value["raw"],
        regions,
        select_index
    )

    return (
        new_model_output,
        select_index,
    )
