import sys

from loguru import logger


def custom_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<fg #003385>[{time:MM/DD HH:mm:ss}]</> <level>{level: ^8}</level>| <level>{message}</level>",
    )
