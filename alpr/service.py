"""FastAPI service exposing the ALPR pipeline."""
from __future__ import annotations

import logging
import os
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import Settings
from .models import load_models
from .process import process_image

LOGGER = logging.getLogger(__name__)


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class HealthResponse(BaseModel):
    status: str


class ProcessResponse(BaseModel):
    processing_time: float
    results: list
    filename: str
    version: int
    camera_id: str | None
    timestamp: str
    image_width: int
    image_height: int
    output_image: str


async def get_settings() -> Settings:
    settings = Settings.from_env()
    return settings


def create_app() -> FastAPI:
    settings = Settings.from_env()
    configure_logging(settings.debug)
    app = FastAPI(title="ALPR Service", version="1.0.0")

    @app.on_event("startup")
    async def _load_model_on_startup() -> None:  # pragma: no cover - best effort
        try:
            load_models(settings)
            LOGGER.info("Models loaded successfully")
        except Exception as exc:  # noqa: BLE001 - ensure failure is visible
            LOGGER.exception("Failed to load models: %s", exc)
            raise

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/alpr", response_model=ProcessResponse)
    async def run_alpr(
        file: UploadFile = File(...),
        settings: Annotated[Settings, Depends(get_settings)] = None,
    ) -> JSONResponse:
        if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "image/bmp"}:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        image_bytes = await file.read()
        try:
            result = process_image(image_bytes, file.filename or "input.jpg", settings=settings)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - return safe error message
            LOGGER.exception("Processing error")
            raise HTTPException(status_code=500, detail="Processing error") from exc

        return JSONResponse(content=result)

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    host = os.getenv("ALPR_HOST", "0.0.0.0")
    port = int(os.getenv("ALPR_PORT", "9999"))
    uvicorn.run("alpr.service:app", host=host, port=port, reload=False)
