from __future__ import annotations
import logging
from typing import Optional

_LOGGER_NAME = "pyoephys"


def configure(level: str = "INFO") -> logging.Logger:
    """
    Configure a namespaced logger (idempotent). Call this in your app/CLI
    if you want console logs immediately.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def get_logger(child: Optional[str] = None) -> logging.Logger:
    """
    Get a namespaced logger, e.g., get_logger("interface.playback")
    """
    name = _LOGGER_NAME if not child else f"{_LOGGER_NAME}.{child}"
    return logging.getLogger(name)
