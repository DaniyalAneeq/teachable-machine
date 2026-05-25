import shutil
from pathlib import Path


def clear_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def reset_application_state(dataset_dir: Path, models_dir: Path) -> dict:
    clear_directory_contents(dataset_dir)
    clear_directory_contents(models_dir)

    return {
        "message": "Application state reset successfully",
        "dataset_cleared": True,
        "model_cleared": True,
    }
