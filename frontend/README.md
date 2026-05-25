# Teachable Machine Clone Frontend

Modern Streamlit frontend for the FastAPI Teachable Machine-style image classification backend.

The UI includes a product-style hero, status dashboard, multi-class Dataset Studio, training checklist, and live prediction workspace.

## Prerequisites

Start the FastAPI backend first. It must be running at:

```text
http://127.0.0.1:8000
```

From the backend folder:

```bash
cd backend
uv run fastapi dev app/main.py
```

## Setup

From the project root:

```powershell
cd frontend
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Use the frontend virtual environment explicitly:

```powershell
cd frontend
venv\Scripts\python.exe -m streamlit run app.py
```

Or run the Windows helper:

```powershell
cd frontend
.\run_frontend.bat
```

## Usage Flow

1. Open the `Dataset Studio` tab.
2. Two class cards appear by default: `Class 1` and `Class 2`.
3. Rename each class card, for example `Cat` and `Dog`.
4. Upload image examples inside each class card.
5. Use `+ Add New Class` to add more classes such as `Bottle`, `Notebook`, or `Phone`.
6. Open the `Train Model` tab and click `Train AI Model`.
7. Open the `Live Prediction` tab.
8. Upload or capture a test image and click `Run Prediction`.

Each class should have at least two training images before training.
For better results, use 10-25+ images per class.

## Multi-Class Dataset Studio

The app is not limited to two classes. You can add 3, 4, 5, or more class cards with `+ Add New Class`.

Each class card has:

- stable class name input
- image uploader for that class
- upload button scoped to that class
- remove button when more than two cards exist

Uploaded images are saved by the backend in `backend/dataset/{class_name}/`.

## Status Dashboard

The frontend shows a dashboard near the top with:

- total dataset classes
- total uploaded images
- model status

The dataset table comes from the backend `/dataset-summary` endpoint. The model status comes from `/model-status`.

The backend remains the source of truth. If Streamlit refreshes or restarts, the UI reloads dataset and model state from the backend automatically.

If new images are uploaded after a model already exists, the UI warns you to retrain before trusting updated predictions.

## Prediction Experience

The `Live Prediction` tab is locked until a trained model exists.

Once unlocked, you can test with:

- one uploaded image
- one camera capture
- live webcam prediction

If both are available, the uploaded image is used.

The result includes:

- predicted class
- confidence percentage
- probability bars
- probability bar chart

## Webcam Features

File upload is still supported for both training and prediction.

For training data, each class card includes a `Webcam Capture` tab:

1. Allow camera permission in your browser.
2. Choose a capture interval.
3. Click `Start Recording Samples`.
4. Move or pose examples for the selected class.
5. Click `Stop Recording` when finished.

The app captures webcam frames repeatedly until stopped and uploads them to the backend as training samples for that class.

For live prediction, use the `Live Webcam` tab in Preview:

- train a model first
- click `Enable Live Prediction`
- predictions update about once per second
- click `Disable Live Prediction` to stop live backend calls

Browser camera permission is required. The FastAPI backend must be running at `http://127.0.0.1:8000`.

## Common Issues

- Backend not running: start FastAPI at `http://127.0.0.1:8000`.
- Not enough classes: upload images for at least two classes before training.
- Not enough images: each class needs at least two images.
- Model not trained: train the model before using prediction.
- New data uploaded after training: retrain the model so predictions include the new images.
- Webcam not starting: allow camera permission in the browser and refresh the page.
