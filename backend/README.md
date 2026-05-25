# Teachable Machine Backend

FastAPI backend for a Teachable Machine-style image classification app.

The backend manages uploaded image datasets, trains a lightweight transfer-learning classifier, runs predictions, and exposes status APIs for the frontend.

## Setup

Create and activate a virtual environment:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run The Server

Start FastAPI with Uvicorn:

```powershell
uvicorn app.main:app --reload
```

Or run it with `uv`:

```powershell
uv run fastapi dev app/main.py
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## Test Uploads In Swagger UI

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Use the `POST /upload-sample` endpoint:

1. Click `Try it out`.
2. Enter a `class_name`, such as `cats`.
3. Upload one or more image files with `.jpg`, `.jpeg`, `.png`, or `.webp` extensions.
4. Click `Execute`.

Uploaded images are saved under:

```text
backend/dataset/<class_name>/
```

Each saved image receives a UUID-based filename while preserving the original file extension.

## Step 2: Train the Model

Before training, upload images for at least two classes. Each class should contain at least two valid image files.

If you use `uv`, sync the new ML dependencies first:

```powershell
uv sync
```

If `uv sync` fails while extracting Torch because the `C:` drive is full, move the `uv` cache to the project drive and run sync again:

```powershell
$env:UV_CACHE_DIR="D:\smit-bootcamp\teachable-machine\.uv-cache"
uv sync
```

In Git Bash:

```bash
export UV_CACHE_DIR=/d/smit-bootcamp/teachable-machine/.uv-cache
uv sync
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Use the `POST /train` endpoint and click `Execute`.

The backend will:

1. Load images from `backend/dataset/`.
2. extract MobileNetV3 Small feature vectors.
3. train a Logistic Regression classifier.
4. save the trained classifier package to:

```text
backend/models/model.pkl
```

## Step 3: Predict an Image

Before prediction:

1. Upload images for at least two classes with `POST /upload-sample`.
2. Train the model with `POST /train`.
3. Confirm `backend/models/model.pkl` exists.

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Use the `POST /predict` endpoint:

1. Click `Try it out`.
2. Upload one JPG, JPEG, PNG, or WEBP image.
3. Click `Execute`.

The endpoint returns the predicted class, confidence percentage, and probability distribution across trained classes.

## Status APIs

### GET /dataset-summary

Returns the current dataset state. Only valid image files are counted.

Example:

```json
{
  "total_classes": 2,
  "total_images": 20,
  "classes": [
    {"class_name": "Cat", "image_count": 10},
    {"class_name": "Dog", "image_count": 10}
  ]
}
```

### GET /model-status

Returns whether a trained model exists and its metadata.

Example:

```json
{
  "model_exists": true,
  "model_path": "D:\\smit-bootcamp\\teachable-machine\\backend\\models\\model.pkl",
  "classes": ["Cat", "Dog"],
  "feature_extractor": "mobilenet_v3_small",
  "image_size": 224
}
```

## Common User Mistakes

- No backend running: frontend shows a backend connection error.
- Uploading unsupported files: API returns `400` with supported image types.
- Empty or invalid class name: API returns `400` with a readable class-name message.
- Class name over 50 characters: API returns `400`.
- Training with fewer than two classes: API returns `400`.
- Training with fewer than two images in a class: API returns `400`.
- Predicting before training: API returns `400`.
- Predicting with a corrupted image: API returns `400`.
