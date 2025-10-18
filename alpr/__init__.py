"""Automatic License Plate Recognition (ALPR) service package."""

from .config import Settings
from .service import create_app

__all__ = ["Settings", "create_app"]
