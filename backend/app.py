"""
Mosquito Classification API
FastAPI backend for:
  - Adult mosquito species identification (Aedes aegypti vs Aedes albopictus)
  - Mosquito larvae identification (Aedes aegypti vs Culex quinquefasciatus)
"""
from __future__ import annotations

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import uvicorn

from model_handler import get_classifier
from utils import save_upload_file, cleanup_upload_file, validate_image_file
from larvae_classifier import get_larvae_classifier

ResponseDict = Dict[str, Any]

app = FastAPI(
    title="Mosquito Classification API",
    description=(
        "API for classifying mosquito species.\n\n"
        "**Adult mosquito:** Aedes aegypti vs Aedes albopictus → `/predict`\n\n"
        "**Larvae:** Aedes aegypti vs Culex quinquefasciatus → `/larvae/identify`"
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
#  STARTUP — load both models
# ─────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    print("Starting Mosquito Classification API...")

    try:
        get_classifier()
        print("Adult mosquito model loaded successfully!")
    except Exception as e:
        print(f"Failed to load adult mosquito model: {e}")

    try:
        get_larvae_classifier()
        print("Larvae model loaded successfully!")
    except Exception as e:
        print(f"Failed to load larvae model: {e}")


# ─────────────────────────────────────────
#  HEALTH / INFO
# ─────────────────────────────────────────

@app.get("/", tags=["Health"])
async def health_check() -> ResponseDict:
    """Check if API and both models are ready."""
    adult_classifier = get_classifier()
    larvae_classifier = get_larvae_classifier()
    return {
        "status": "online",
        "adult_model_loaded": adult_classifier.model is not None,
        "larvae_model_loaded": larvae_classifier.model is not None,
    }


@app.get("/model-info", tags=["Health"])
async def get_model_info() -> ResponseDict:
    """Get information about the loaded adult mosquito model."""
    classifier = get_classifier()
    return {
        "model_path": str(classifier.model_path),
        "classes": classifier.class_names,
        "input_size": classifier.img_size,
        "model_loaded": classifier.model is not None,
    }


@app.get("/larvae/model-info", tags=["Larvae"])
async def get_larvae_model_info() -> ResponseDict:
    """Get information about the loaded larvae model."""
    classifier = get_larvae_classifier()
    return {
        "model_path": str(classifier.model_path),
        "classes": classifier.class_names,
        "input_size": classifier.img_size,
        "model_loaded": classifier.model is not None,
    }


# ─────────────────────────────────────────
#  ADULT MOSQUITO — existing endpoint
# ─────────────────────────────────────────

@app.post("/predict", tags=["Adult Mosquito"])
async def predict_mosquito(file: UploadFile = File(...)) -> ResponseDict:
    """
    Predict adult mosquito species from uploaded image.
    Classes: **aegypti** | **albopictus**
    """
    if not validate_image_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image (jpg, jpeg, png, bmp, tiff)"
        )

    file_path = None
    try:
        file_path = save_upload_file(file)
        print(f"Received file: {file.filename}")
        classifier = get_classifier()
        result = classifier.predict(file_path)
        print(f"Prediction: {result}")
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path:
            cleanup_upload_file(file_path)


# ─────────────────────────────────────────
#  LARVAE — new endpoints
# ─────────────────────────────────────────

LARVAE_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@app.post("/larvae/identify", tags=["Larvae"])
async def identify_larvae(file: UploadFile = File(...)) -> ResponseDict:
    """
    Identify mosquito larvae species from uploaded image.
    Classes: **Aedes aegypti** | **Culex quinquefasciatus**
    """
    if file.content_type not in LARVAE_ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type '{file.content_type}'. Allowed: jpeg, png, webp, bmp"
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        print(f"Received larvae image: {file.filename}")
        classifier = get_larvae_classifier()
        result = classifier.predict(image_bytes)
        result["filename"] = file.filename
        print(f"Larvae prediction: {result}")
        return result
    except Exception as e:
        print(f"Error during larvae prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/larvae/classes", tags=["Larvae"])
async def get_larvae_classes() -> ResponseDict:
    """Return supported larvae classes and associated diseases."""
    return {
        "classes": [
            {
                "name": "Aedes aegypti",
                "common_name": "Yellow Fever Mosquito",
                "diseases": ["Dengue", "Zika", "Chikungunya", "Yellow Fever"]
            },
            {
                "name": "Culex quinquefasciatus",
                "common_name": "Southern House Mosquito",
                "diseases": ["West Nile Virus", "Lymphatic Filariasis", "St. Louis Encephalitis"]
            }
        ]
    }


# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)