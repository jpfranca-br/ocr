# ALPR Service

This repository provides a FastAPI-based Automatic License Plate Recognition (ALPR) service that exposes an HTTP endpoint on port **9999**. The service detects vehicles and license plates in uploaded images, recognises plate characters using YOLOv8, and returns structured JSON metadata alongside an annotated image saved to `alpr/output/`.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Provide YOLOv8 model weights for:

- Vehicle and license plate detection (`ALPR_DETECTION_MODEL`, default `models/vehicle_plate.pt`)
- Character recognition (`ALPR_RECOGNITION_MODEL`, default `models/plate_recognition.pt`)

The service expects the detection model to output bounding boxes labelled with vehicle classes (e.g. `car`, `truck`, `bus`) and plate classes (e.g. `license_plate`). The recognition model should output character-level detections whose class names are the plate characters.

## Running the service

Start the API with:

```bash
uvicorn alpr.service:app --host 0.0.0.0 --port 9999
```

Optional environment variables:

- `ALPR_DEBUG`: set to `1` or `true` to enable verbose logging.
- `ALPR_OUTPUT_DIR`: directory to write annotated images (`alpr/output` by default).
- `ALPR_DETECTION_MODEL`: path to the detection YOLOv8 weights.
- `ALPR_RECOGNITION_MODEL`: path to the recognition YOLOv8 weights.
- `ALPR_VEHICLE_CLASSES`: comma-separated list of detection labels treated as vehicles.
- `ALPR_PLATE_CLASSES`: comma-separated list of detection labels treated as license plates.

## API

- `GET /health` – health check endpoint.
- `POST /alpr` – accepts an image file upload. Returns JSON with detection metadata, candidate plates, and the path to the annotated image saved in `alpr/output/`.

## Output

For each processed image the service:

1. Draws blue bounding boxes around detected vehicles and red bounding boxes around plates, writing candidate plate strings and confidence scores below each plate. The annotated image is saved under `alpr/output/`.
2. Returns JSON describing processing time, vehicle and plate detections, candidate licence plate strings, and image metadata.

Example response snippet:

```json
{
  "processing_time": 1.23,
  "results": [
    {
      "box": {"xmin": 100, "ymin": 200, "xmax": 250, "ymax": 240},
      "plate": "abc1234",
      "score": 0.98,
      "candidates": [
        {"plate": "abc1234", "score": 0.98}
      ],
      "vehicle": {
        "type": "car",
        "score": 0.92,
        "box": {"xmin": 50, "ymin": 100, "xmax": 320, "ymax": 260}
      }
    }
  ],
  "filename": "input.jpg",
  "output_image": "alpr/output/input_20240101T101010000000Z.jpg"
}
```

Ensure you supply appropriate YOLOv8 weights before running the service.
