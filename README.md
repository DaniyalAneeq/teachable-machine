# Teachable Machine Clone

A local Teachable Machine-style image classification app built with a FastAPI backend and a Streamlit frontend. The app lets you create image classes, upload or capture training samples, train a lightweight classifier, and test predictions from uploaded images or webcam frames.

The backend uses MobileNetV3 Small as a frozen feature extractor and trains a Logistic Regression classifier on top of those features. The frontend provides the full workflow in one Streamlit workspace.

## Features

- Multi-class image dataset builder
- Upload-based training samples
- Browser camera capture for training samples
- MobileNetV3 Small feature extraction
- Logistic Regression image classifier
- Model training from the local dataset
- Uploaded image, camera image, and live webcam prediction
- Dataset summary and model status dashboard
- Backend reset flow for clearing generated dataset/model state
- FastAPI Swagger UI for direct API testing

## Tech Stack

| Layer | Tools |
| --- | --- |
| Backend | FastAPI, Uvicorn, Pillow, PyTorch, Torchvision, scikit-learn, Joblib, NumPy |
| Frontend | Streamlit, Requests, Pandas, Pillow, streamlit-webrtc, OpenCV, AV, NumPy |
| Model | MobileNetV3 Small feature extractor + Logistic Regression classifier |

## Project Structure

```text
teachable-machine/
|-- .streamlit/
|   `-- config.toml
|-- backend/
|   |-- app/
|   |   |-- services/
|   |   |   |-- dataset_service.py
|   |   |   |-- model_service.py
|   |   |   |-- prediction_service.py
|   |   |   |-- reset_service.py
|   |   |   `-- training_service.py
|   |   |-- utils/
|   |   |   `-- file_utils.py
|   |   |-- config.py
|   |   |-- main.py
|   |   `-- schemas.py
|   |-- dataset/
|   |   `-- .gitkeep
|   |-- images/
|   |-- models/
|   |   `-- .gitkeep
|   |-- pyproject.toml
|   |-- requirements.txt
|   `-- uv.lock
|-- frontend/
|   |-- app.py
|   |-- package.json
|   |-- requirements.txt
|   `-- run_frontend.bat
|-- .gitignore
`-- README.md
```

## Prerequisites

- Python 3.10 or newer
- pip
- A modern browser
- Webcam permission in the browser if you want camera or live webcam features

The backend `pyproject.toml` currently declares Python `>=3.14`, but the dependency files can also be installed with `pip` from `requirements.txt`.

## Setup

Clone the project and open the root folder:

```bash
git clone <your-repository-url>
cd teachable-machine
```

Create and install the backend environment:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create and install the frontend environment:

```powershell
cd ..\frontend
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run The App

Start the backend first:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload or fastapi dev app/main.py
```

Backend URLs:

- API root: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

In a second terminal, start the frontend:

```powershell
cd frontend
venv\Scripts\python.exe -m streamlit run app.py
```

You can also start the frontend on Windows with:

```powershell
cd frontend
.\run_frontend.bat
```

Streamlit will print the local frontend URL in the terminal, usually:

```text
http://localhost:8501
```

## Usage

1. Start the FastAPI backend.
2. Start the Streamlit frontend.
3. Add at least two classes in the Model Builder.
4. Upload or capture at least two images for each class.
5. Click `Train Model`.
6. Upload, capture, or stream a test image in the Preview area.
7. Run prediction and review the predicted class, confidence, and probability breakdown.

For better results, use more varied images per class. A practical starting point is 10 to 25 images per class.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Backend health message |
| `GET` | `/dataset-summary` | Returns classes, image counts, and image filenames |
| `GET` | `/model-status` | Returns saved model availability and metadata |
| `POST` | `/upload-sample` | Uploads one or more images for a class |
| `POST` | `/train` | Trains and saves the image classifier |
| `POST` | `/predict` | Predicts the class of one image |
| `POST` | `/reset-app` | Clears generated dataset and model files |

Supported image types are `.jpg`, `.jpeg`, `.png`, and `.webp`.

## Runtime Data

The app generates local data while it runs:

- Uploaded training images are stored in `backend/dataset/`
- Trained model files are stored in `backend/models/`
- Python caches, virtual environments, local package caches, and `node_modules` are ignored by Git

Only `.gitkeep` placeholders are committed inside `backend/dataset/` and `backend/models/` so the folder structure is present in a fresh clone.

## GitHub Notes

Before pushing, initialize Git from the project root if this root folder is not already a repository:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

If `backend/.git` exists from an older nested repository, remove or move that nested Git metadata before adding the full project from the root. Otherwise Git may treat `backend` as a separate repository instead of normal project files.

## Common Issues

- Backend connection error: make sure FastAPI is running at `http://127.0.0.1:8000`.
- Training disabled: each class needs at least two valid images.
- Prediction locked: train the model first.
- Outdated predictions: retrain after adding new training images.
- Webcam not available: allow camera permission in the browser and refresh the page.
- Invalid image error: upload a supported and readable image file.
