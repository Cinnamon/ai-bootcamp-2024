import os.path

from loguru import logger
from PIL import Image


def save_uploaded_file(file, dir_out: str) -> str:
    """Save uploaded file to local"""
    pil_image = Image.open(file)

    path_out = os.path.join(dir_out, file.name)
    pil_image.save(
        path_out
    )

    assert os.path.isfile(path_out)
    logger.info(f"Save file at: {path_out}")

    return path_out
