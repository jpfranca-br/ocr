"""Image processing pipeline for vehicle and license plate detection."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import Settings
from .models import ModelBundle, load_models

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @classmethod
    def from_xyxy(cls, coords: Sequence[float], image_shape: Tuple[int, int]) -> "BoundingBox":
        height, width = image_shape
        xmin, ymin, xmax, ymax = coords
        return cls(
            xmin=int(max(0, min(width - 1, round(xmin)))),
            ymin=int(max(0, min(height - 1, round(ymin)))),
            xmax=int(max(0, min(width - 1, round(xmax)))),
            ymax=int(max(0, min(height - 1, round(ymax)))),
        )

    def to_dict(self) -> Dict[str, int]:
        return {"xmin": self.xmin, "ymin": self.ymin, "xmax": self.xmax, "ymax": self.ymax}

    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    def area(self) -> int:
        return self.width() * self.height()

    def intersection(self, other: "BoundingBox") -> int:
        x1 = max(self.xmin, other.xmin)
        y1 = max(self.ymin, other.ymin)
        x2 = min(self.xmax, other.xmax)
        y2 = min(self.ymax, other.ymax)
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def iou(self, other: "BoundingBox") -> float:
        inter = self.intersection(other)
        union = self.area() + other.area() - inter
        if union <= 0:
            return 0.0
        return inter / union


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    box: BoundingBox


@dataclass(slots=True)
class PlateCandidate:
    plate: str
    score: float


@dataclass(slots=True)
class PlateResult:
    plate_box: Detection
    vehicle: Optional[Detection]
    candidates: List[PlateCandidate]
    timestamp: datetime

    def to_dict(self) -> Dict[str, object]:
        top_candidate = self.candidates[0] if self.candidates else PlateCandidate("", 0.0)
        vehicle_dict = None
        if self.vehicle:
            vehicle_dict = {
                "score": round(self.vehicle.confidence, 3),
                "type": self.vehicle.label,
                "box": self.vehicle.box.to_dict(),
            }
        return {
            "box": self.plate_box.box.to_dict(),
            "plate": top_candidate.plate,
            "region": {"code": None, "score": 0.0},
            "score": round(self.plate_box.confidence, 3),
            "candidates": [
                {"score": round(candidate.score, 3), "plate": candidate.plate}
                for candidate in self.candidates
            ],
            "dscore": round(self.plate_box.confidence, 3),
            "vehicle": vehicle_dict,
        }


def _extract_detections(result, image_shape: Tuple[int, int], classes: Sequence[str]) -> List[Detection]:
    detections: List[Detection] = []
    names = result.names
    for box in getattr(result, "boxes", []):
        cls_idx = int(box.cls[0])
        label = names.get(cls_idx, str(cls_idx))
        if label not in classes:
            continue
        confidence = float(box.conf[0]) if box.conf is not None else 0.0
        bbox = BoundingBox.from_xyxy(box.xyxy[0].tolist(), image_shape)
        detections.append(Detection(label=label, confidence=confidence, box=bbox))
    return detections


def _recognize_plate(model_bundle: ModelBundle, plate_image: np.ndarray) -> List[PlateCandidate]:
    recognition_results = model_bundle.recognition.predict(plate_image, verbose=False)
    if not recognition_results:
        return []
    result = recognition_results[0]
    if not getattr(result, "boxes", None):
        return []

    characters = []
    names = result.names
    for box in result.boxes:
        cls_idx = int(box.cls[0])
        label = names.get(cls_idx, str(cls_idx))
        confidence = float(box.conf[0]) if box.conf is not None else 0.0
        x_center = float(box.xywh[0][0])
        characters.append((x_center, label, confidence))

    if not characters:
        return []

    characters.sort(key=lambda item: item[0])
    plate_text = "".join(label for _, label, _ in characters)
    score = float(np.mean([conf for _, _, conf in characters])) if characters else 0.0
    return [PlateCandidate(plate=plate_text, score=score)]


def _attach_vehicle(plate: Detection, vehicles: Sequence[Detection]) -> Optional[Detection]:
    if not vehicles:
        return None
    best_vehicle = max(vehicles, key=lambda vehicle: plate.box.iou(vehicle.box))
    if plate.box.iou(best_vehicle.box) == 0:
        return None
    return best_vehicle


def annotate_image(image: np.ndarray, plate_results: Sequence[PlateResult]) -> np.ndarray:
    annotated = image.copy()
    for result in plate_results:
        # Draw vehicle box if available
        if result.vehicle:
            box = result.vehicle.box
            cv2.rectangle(annotated, (box.xmin, box.ymin), (box.xmax, box.ymax), (255, 0, 0), 2)
        # Draw plate box
        plate_box = result.plate_box.box
        cv2.rectangle(annotated, (plate_box.xmin, plate_box.ymin), (plate_box.xmax, plate_box.ymax), (0, 0, 255), 2)
        if result.candidates:
            text = ", ".join(f"{candidate.plate} ({candidate.score:.2f})" for candidate in result.candidates)
            text_position = (plate_box.xmin, min(plate_box.ymax + 20, annotated.shape[0] - 5))
            cv2.putText(
                annotated,
                text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return annotated


def process_image(
    image_bytes: bytes,
    filename: str,
    settings: Optional[Settings] = None,
    model_bundle: Optional[ModelBundle] = None,
) -> Dict[str, object]:
    settings = settings or Settings.from_env()
    settings.ensure_output_directory()

    np_img = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image data")

    start_time = time.perf_counter()

    bundle = model_bundle or load_models(settings)

    detection_results = bundle.detection.predict(image, verbose=False)
    if not detection_results:
        raise RuntimeError("Detection model did not return any results")
    detection = detection_results[0]

    vehicles = _extract_detections(detection, image.shape[:2], settings.vehicle_classes)
    plates = _extract_detections(detection, image.shape[:2], settings.plate_classes)

    plate_results: List[PlateResult] = []
    for plate in plates:
        plate_image = image[plate.box.ymin : plate.box.ymax, plate.box.xmin : plate.box.xmax]
        candidates = _recognize_plate(bundle, plate_image)
        if not candidates:
            candidates = [PlateCandidate(plate="", score=0.0)]
        vehicle = _attach_vehicle(plate, vehicles)
        plate_results.append(
            PlateResult(
                plate_box=plate,
                vehicle=vehicle,
                candidates=candidates,
                timestamp=datetime.now(timezone.utc),
            )
        )

    annotated = annotate_image(image, plate_results)

    output_filename = _build_output_filename(filename)
    output_path = settings.output_directory / output_filename
    cv2.imwrite(str(output_path), annotated)

    processing_time = time.perf_counter() - start_time

    response = {
        "processing_time": round(processing_time, 2),
        "results": [result.to_dict() for result in plate_results],
        "filename": filename,
        "version": 1,
        "camera_id": None,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "output_image": str(output_path),
    }

    LOGGER.info("Processed %s in %.3f seconds", filename, processing_time)
    return response


def _build_output_filename(filename: str) -> str:
    stem = Path(filename).stem or "processed"
    suffix = Path(filename).suffix or ".jpg"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stem}_{timestamp}{suffix}"


__all__ = ["process_image", "annotate_image", "BoundingBox", "PlateResult"]
