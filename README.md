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

### Obtaining YOLOv8 weights

1. Visit the [Ultralytics model hub](https://github.com/ultralytics/ultralytics/releases) or your internal model registry and download the trained weight files (`.pt`).
2. Place the files in a directory accessible to the service (for example, `models/`).
3. Set the environment variables before starting the service so the FastAPI app can load the files:

   ```bash
   export ALPR_DETECTION_MODEL=/absolute/path/to/vehicle_plate.pt
   export ALPR_RECOGNITION_MODEL=/absolute/path/to/plate_recognition.pt
   ```

You can also override the paths inline when launching Uvicorn:

```bash
ALPR_DETECTION_MODEL=models/vehicle_plate.pt \
ALPR_RECOGNITION_MODEL=models/plate_recognition.pt \
uvicorn alpr.service:app --host 0.0.0.0 --port 9999
```

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

### Example cURL request

```bash
curl -X POST "http://localhost:9999/alpr" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

The response contains the processed metadata and the relative path to the annotated image in `alpr/output/`.

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
