from pathlib import Path

import joblib


REQUIRED_MODEL_KEYS = {"classifier", "classes", "feature_extractor", "image_size"}


def get_model_status(model_path: Path) -> dict:
    if not model_path.exists():
        return {
            "model_exists": False,
            "model_path": None,
            "classes": [],
            "feature_extractor": None,
            "image_size": None,
        }

    try:
        model_package = joblib.load(model_path)
    except Exception as error:
        raise ValueError("Saved model is invalid. Please retrain the model.") from error

    if not isinstance(model_package, dict) or not REQUIRED_MODEL_KEYS.issubset(model_package):
        raise ValueError("Saved model is invalid. Please retrain the model.")

    return {
        "model_exists": True,
        "model_path": str(model_path),
        "classes": [str(class_name) for class_name in model_package.get("classes", [])],
        "feature_extractor": model_package.get("feature_extractor"),
        "image_size": model_package.get("image_size"),
    }
