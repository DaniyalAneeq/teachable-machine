import re
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sanitize_class_name(class_name: str) -> str:
    """Return a folder-safe class name."""
    cleaned_name = class_name.strip().replace(" ", "_")
    cleaned_name = re.sub(r"[^A-Za-z0-9_-]", "", cleaned_name)

    if not cleaned_name:
        raise ValueError("Class name must contain at least one letter, number, underscore, or hyphen.")

    return cleaned_name


def is_allowed_image(filename: str) -> bool:
    """Check whether the uploaded filename has a supported image extension."""
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def save_uploaded_image(file: UploadFile, class_dir: Path) -> str:
    """Save an uploaded image with a UUID filename and return the saved filename."""
    original_extension = Path(file.filename or "").suffix.lower()
    saved_filename = f"{uuid4().hex}{original_extension}"
    destination = class_dir / saved_filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return saved_filename
