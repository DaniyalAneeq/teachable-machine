from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from app.services.dataset_service import get_dataset_summary


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_SIZE = 224


def get_class_folders(dataset_dir: Path) -> list[Path]:
    """Return class directories sorted by name for stable label ordering."""
    if not dataset_dir.exists():
        return []

    return sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )


def validate_dataset(class_folders: list[Path]) -> None:
    if not class_folders:
        raise ValueError("No training data found. Please upload images before training.")

    if len(class_folders) < 2:
        raise ValueError("Training requires at least 2 classes. Please upload images for two or more classes.")

    for class_folder in class_folders:
        image_count = sum(
            1
            for path in class_folder.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )

        if image_count < 2:
            raise ValueError(f"Class '{class_folder.name}' must contain at least 2 images.")


def get_image_paths_and_labels(class_folders: list[Path]) -> tuple[list[Path], list[str]]:
    image_paths: list[Path] = []
    labels: list[str] = []

    for class_folder in class_folders:
        valid_images = sorted(
            [
                path
                for path in class_folder.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ],
            key=lambda path: path.name.lower(),
        )

        image_paths.extend(valid_images)
        labels.extend([class_folder.name] * len(valid_images))

    return image_paths, labels


def build_feature_extractor() -> tuple[torch.nn.Module, transforms.Compose]:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    # Keep MobileNet's convolutional backbone and pooling layer to turn images
    # into feature vectors that a simple classifier can learn from.
    feature_extractor = torch.nn.Sequential(model.features, model.avgpool)

    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    feature_extractor.eval()
    preprocessing = weights.transforms(crop_size=IMAGE_SIZE, resize_size=IMAGE_SIZE)
    return feature_extractor, preprocessing


def extract_features(
    image_paths: list[Path],
    labels: list[str],
    feature_extractor: torch.nn.Module,
    transform,
) -> tuple[np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    extended_labels: list[str] = []

    # Augmentation pipeline to generate variations and prevent overfitting
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    with torch.no_grad():
        for image_path, label in zip(image_paths, labels):
            try:
                with Image.open(image_path) as image:
                    rgb_image = image.convert("RGB")
                    
                    # 1. Original Image
                    image_tensor = transform(rgb_image).unsqueeze(0)
                    output = feature_extractor(image_tensor)
                    features.append(output.flatten(start_dim=1).cpu().numpy()[0])
                    extended_labels.append(label)

                    # 2. Augmented Variations (4 per image) to improve generalization
                    for _ in range(4):
                        aug_image = augmentation(rgb_image)
                        aug_tensor = transform(aug_image).unsqueeze(0)
                        aug_output = feature_extractor(aug_tensor)
                        features.append(aug_output.flatten(start_dim=1).cpu().numpy()[0])
                        extended_labels.append(label)

            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"Could not read image '{image_path.name}'. Please remove or replace it.") from error

    if not features:
        raise ValueError("No valid images were found for training.")

    return np.asarray(features), extended_labels


def train_classifier(features: np.ndarray, labels: list[str]) -> LogisticRegression:
    # Use L2 regularization (C=0.1) and balanced weights to prevent overfitting on tiny datasets
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
    classifier.fit(features, labels)
    return classifier


def train_model(dataset_dir: Path, model_path: Path) -> dict:
    class_folders = get_class_folders(dataset_dir)
    validate_dataset(class_folders)
    dataset_summary = get_dataset_summary(dataset_dir)

    image_paths, labels = get_image_paths_and_labels(class_folders)
    feature_extractor, transform = build_feature_extractor()
    
    # Extract features with data augmentation
    features, augmented_labels = extract_features(image_paths, labels, feature_extractor, transform)
    
    classifier = train_classifier(features, augmented_labels)

    classes = [class_folder.name for class_folder in class_folders]
    model_package = {
        "classifier": classifier,
        "classes": classes,
        "feature_extractor": "mobilenet_v3_small",
        "image_size": IMAGE_SIZE,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, model_path)

    return {
        "message": "Model trained successfully",
        "classes": classes,
        "total_images": len(image_paths),
        "model_path": str(model_path),
        "dataset_summary": dataset_summary,
    }
