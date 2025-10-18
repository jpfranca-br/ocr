"""Model loading utilities for the ALPR pipeline."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from .config import Settings

LOGGER = logging.getLogger(__name__)


class ModelBundle:
    """Container for the detection and recognition models."""

    def __init__(self, detection: YOLO, recognition: YOLO) -> None:
        self.detection = detection
        self.recognition = recognition


@lru_cache(maxsize=1)
def load_models(settings: Optional[Settings] = None) -> ModelBundle:
    """Load and cache the YOLO models required for the pipeline."""
    settings = settings or Settings.from_env()
    LOGGER.info("Loading detection model from %s", settings.detection_model_path)
    detection_model = _load_model(settings.detection_model_path)
    LOGGER.info("Loading recognition model from %s", settings.recognition_model_path)
    recognition_model = _load_model(settings.recognition_model_path)
    return ModelBundle(detection_model, recognition_model)


def _load_model(path: Path) -> YOLO:
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path}")
    return YOLO(str(path))


__all__ = ["ModelBundle", "load_models"]
