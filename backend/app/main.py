from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import DATASET_DIR, MODEL_PATH, MODELS_DIR
from app.schemas import DatasetSummaryResponse, ErrorResponse, ModelStatusResponse
from app.services.dataset_service import get_dataset_summary
from app.services.model_service import get_model_status
from app.services.prediction_service import predict_image
from app.services.reset_service import reset_application_state
from app.services.training_service import train_model
from app.utils.file_utils import is_allowed_image, sanitize_class_name, save_uploaded_image


app = FastAPI(title="Teachable Machine Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/dataset", StaticFiles(directory=str(DATASET_DIR)), name="dataset")


def handle_value_error(error: ValueError) -> None:
    raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Teachable Machine Backend is running"}


@app.get("/dataset-summary", response_model=DatasetSummaryResponse)
def dataset_summary() -> dict:
    return get_dataset_summary(DATASET_DIR)


@app.get(
    "/model-status",
    response_model=ModelStatusResponse,
    responses={400: {"model": ErrorResponse}},
)
def model_status() -> dict:
    try:
        return get_model_status(MODEL_PATH)
    except ValueError as error:
        handle_value_error(error)


@app.post("/reset-app")
def reset_app() -> dict:
    try:
        return reset_application_state(DATASET_DIR, MODELS_DIR)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Reset failed: {error}") from error


@app.post("/upload-sample")
def upload_sample(
    class_name: str = Form(...),
    files: List[UploadFile] = File(...),
) -> dict[str, object]:
    trimmed_class_name = class_name.strip()
    if len(trimmed_class_name) > 50:
        raise HTTPException(status_code=400, detail="Class name is too long. Please use 50 characters or fewer.")

    try:
        sanitized_class_name = sanitize_class_name(trimmed_class_name)
    except ValueError as error:
        handle_value_error(error)

    if not files:
        raise HTTPException(status_code=400, detail="At least one image file must be uploaded.")

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Each uploaded file must have a filename.")

        if not is_allowed_image(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type for '{file.filename}'. Allowed types: .jpg, .jpeg, .png, .webp.",
            )

    class_dir = DATASET_DIR / sanitized_class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    saved_files = [save_uploaded_image(file, class_dir) for file in files]

    return {
        "message": "Images uploaded successfully",
        "class_name": sanitized_class_name,
        "saved_count": len(saved_files),
        "saved_files": saved_files,
    }


@app.post("/train")
def train() -> dict:
    try:
        return train_model(DATASET_DIR, MODEL_PATH)
    except ValueError as error:
        handle_value_error(error)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Training failed: {error}") from error


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> dict:
    if not file.filename or not is_allowed_image(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload JPG, JPEG, PNG, or WEBP.",
        )

    try:
        return predict_image(file, MODEL_PATH)
    except ValueError as error:
        handle_value_error(error)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error
