"""Configuration management for the ALPR service."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import List, Sequence

LOGGER = logging.getLogger(__name__)


def _split_env_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class Settings:
    """Application settings loaded from environment variables."""

    detection_model_path: Path = field(
        default_factory=lambda: Path(os.getenv("ALPR_DETECTION_MODEL", "models/vehicle_plate.pt"))
    )
    recognition_model_path: Path = field(
        default_factory=lambda: Path(os.getenv("ALPR_RECOGNITION_MODEL", "models/plate_recognition.pt"))
    )
    output_directory: Path = field(
        default_factory=lambda: Path(os.getenv("ALPR_OUTPUT_DIR", "alpr/output"))
    )
    debug: bool = field(default_factory=lambda: os.getenv("ALPR_DEBUG", "false").lower() in {"1", "true", "yes"})
    vehicle_classes: Sequence[str] = field(
        default_factory=lambda: _split_env_list(os.getenv("ALPR_VEHICLE_CLASSES"))
        or ["car", "truck", "bus", "motorcycle", "vehicle"]
    )
    plate_classes: Sequence[str] = field(
        default_factory=lambda: _split_env_list(os.getenv("ALPR_PLATE_CLASSES"))
        or ["license-plate", "license_plate", "plate", "number-plate"]
    )

    def ensure_output_directory(self) -> None:
        """Create the output directory if it does not already exist."""
        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOGGER.error("Unable to create output directory %s", self.output_directory)
            raise

    def to_json(self) -> str:
        data = {
            "detection_model_path": str(self.detection_model_path),
            "recognition_model_path": str(self.recognition_model_path),
            "output_directory": str(self.output_directory),
            "debug": self.debug,
            "vehicle_classes": list(self.vehicle_classes),
            "plate_classes": list(self.plate_classes),
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls()
        settings.ensure_output_directory()
        return settings


__all__ = ["Settings"]
