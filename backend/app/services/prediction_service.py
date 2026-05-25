from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.services.training_service import build_feature_extractor


REQUIRED_MODEL_KEYS = {"classifier", "classes", "feature_extractor", "image_size"}


def load_model_package(model_path: Path) -> dict:
    if not model_path.exists():
        raise ValueError("No trained model found. Please train the model first.")

    model_package = joblib.load(model_path)

    if not isinstance(model_package, dict) or not REQUIRED_MODEL_KEYS.issubset(model_package):
        raise ValueError("Invalid model package. Please retrain the model.")

    return model_package


def build_prediction_feature_extractor() -> tuple[torch.nn.Module, transforms.Compose]:
    return build_feature_extractor()


def extract_single_image_features(
    image_file: UploadFile,
    feature_extractor: torch.nn.Module,
    transform,
) -> np.ndarray:
    try:
        image_file.file.seek(0)
        with Image.open(image_file.file) as image:
            image_tensor = transform(image.convert("RGB")).unsqueeze(0)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("Invalid image file. Please upload a valid image.") from error

    with torch.no_grad():
        output = feature_extractor(image_tensor)

    return output.flatten(start_dim=1).cpu().numpy()


def predict_image(image_file: UploadFile, model_path: Path) -> dict:
    model_package = load_model_package(model_path)
    classifier = model_package["classifier"]

    feature_extractor, transform = build_prediction_feature_extractor()
    features = extract_single_image_features(image_file, feature_extractor, transform)
    probabilities = classifier.predict_proba(features)[0]

    class_names = [str(class_name) for class_name in classifier.classes_]
    probability_percentages = {
        class_name: round(float(probability * 100), 2)
        for class_name, probability in zip(class_names, probabilities)
    }

    best_index = int(np.argmax(probabilities))
    predicted_class = class_names[best_index]
    confidence = round(float(probabilities[best_index] * 100), 2)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_percentages,
    }
