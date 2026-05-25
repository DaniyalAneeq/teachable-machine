from pathlib import Path

from app.utils.file_utils import ALLOWED_IMAGE_EXTENSIONS


def get_dataset_summary(dataset_dir: Path) -> dict:
    if not dataset_dir.exists():
        return {"total_classes": 0, "total_images": 0, "classes": []}

    class_summaries = []

    for class_dir in sorted((path for path in dataset_dir.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
        images = [
            path.name
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        ]
        class_summaries.append({
            "class_name": class_dir.name, 
            "image_count": len(images),
            "images": images
        })

    total_images = sum(class_summary["image_count"] for class_summary in class_summaries)

    return {
        "total_classes": len(class_summaries),
        "total_images": total_images,
        "classes": class_summaries,
    }
