import io
import threading
import time
from typing import Any
from uuid import uuid4

import av
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from requests import Response
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


BACKEND_URL = "http://127.0.0.1:8000"
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


st.set_page_config(
    page_title="Teachable Machine Clone",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        :root {
            --bg-color: #f0f4f8;
            --card-bg: #ffffff;
            --text-main: #202124;
            --text-muted: #5f6368;
            --border-color: #dadce0;
            --primary-blue: #e8f0fe;
            --primary-blue-text: #1967d2;
            --primary-green: #10b981;
            --radius: 8px;
        }

        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: var(--bg-color) !important;
            color: var(--text-main) !important;
            font-family: 'Roboto', sans-serif !important;
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 4rem;
            max-width: 1400px;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid var(--border-color);
        }
        [data-testid="stSidebar"] * {
            font-family: 'Roboto', sans-serif !important;
            color: var(--text-main) !important;
        }

        /* Shared Card Styles via Advanced CSS Marker Strategy */
        div[data-testid="stVerticalBlock"]:has(> .element-container .st-card-marker) {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
            padding: 1.5rem !important;
            transition: box-shadow 0.3s ease !important;
            animation: fadeIn 0.5s ease-out;
            position: relative;
        }
        div[data-testid="stVerticalBlock"]:has(> .element-container .st-card-marker):hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }
        
        .st-card-marker {
            display: none;
        }
        
        .hero-card {
            text-align: center;
            margin-bottom: 2rem;
            padding: 2.5rem;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
        }
        .hero-card h1 {
            color: var(--text-main);
            font-size: 2.5rem;
            font-weight: 700;
            margin: 1rem 0;
        }
        .hero-card p {
            color: var(--text-muted);
            font-size: 1.1rem;
            margin: 0 auto 1.5rem auto;
            max-width: 800px;
        }

        /* Specific Cards */
        .class-card { border-top: 4px solid var(--primary-blue-text); }
        .prediction-card { border-top: 4px solid var(--primary-green); }
        .empty-card {
            background: #f8f9fa;
            border: 1px dashed #bdc1c6;
            color: var(--text-muted);
            text-align: center;
        }
        .warning-card { background: #fef7e0; border-color: #fbbc04; color: #b06000; }
        .success-card { background: #e6f4ea; border-color: #34a853; color: #137333; }

        /* Builder Nodes */
        .builder-shell {
            margin-top: 1.5rem;
            padding: 1.5rem;
            background: transparent;
        }
        .flow-label { display: none; }
        
        .add-class-node {
            border-radius: var(--radius);
            border: 2px dashed #bdc1c6;
            background: transparent;
            padding: 1.25rem;
            text-align: center;
            color: var(--text-muted);
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s, color 0.2s;
            margin-top: 1rem;
        }
        .add-class-node:hover {
            background: rgba(0,0,0,0.02);
            color: var(--text-main);
        }
        
        /* The connecting arrows */
        .flow-arrow {
            text-align: center;
            color: #dadce0;
            font-size: 2rem;
            margin-top: 6rem;
            animation: pulse-arrow 2s infinite ease-in-out;
        }

        /* Pills & Badges */
        .status-pill, .success-pill, .warning-pill, .danger-pill, .step-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 4px;
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
            font-weight: 500;
            margin: 0.25rem 0.5rem 0.25rem 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-pill { background: #f1f3f4; color: #5f6368; border: 1px solid #dadce0; }
        .success-pill { background: #e6f4ea; color: #137333; border: 1px solid #ceead6; }
        .warning-pill { background: #fef7e0; color: #b06000; border: 1px solid #feefc3; }
        .danger-pill { background: #fce8e6; color: #c5221f; border: 1px solid #fad2cf; }
        .step-badge { background: #e8f0fe; color: #1967d2; border: 1px solid #d2e3fc; }

        /* Typography */
        .section-title {
            color: var(--text-main);
            font-size: 1.5rem;
            font-weight: 500;
            margin: 2rem 0 0.5rem 0;
        }
        .muted-text { color: var(--text-muted); font-size: 1rem; margin-bottom: 1.5rem; }
        .small-caption { color: var(--text-muted); font-size: 0.85rem; margin-top: 0.25rem; }

        /* Metrics */
        div[data-testid="stMetric"] {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 1.25rem 1.5rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        div[data-testid="stMetric"] label { color: var(--text-muted) !important; font-weight: 500; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text-main) !important; font-weight: 700; font-size: 2rem; }

        /* General Streamlit */
        .stMarkdown, label, p, span { color: var(--text-main); font-family: 'Roboto', sans-serif; }
        h1, h2, h3, h4, h5, h6 { color: var(--text-main) !important; font-family: 'Roboto', sans-serif !important; }

        /* Buttons like the picture */
        .stButton > button {
            border-radius: 6px;
            border: 1px solid #dadce0;
            font-weight: 500;
            padding: 0.5rem 1.5rem;
            background: #ffffff;
            color: var(--primary-blue-text) !important;
            transition: all 0.2s ease;
            box-shadow: none;
        }
        .stButton > button * { color: var(--primary-blue-text) !important; }
        .stButton > button:hover {
            background: #f8f9fa;
            border-color: #d2e3fc;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .stButton > button:active { background: #e8eaed; }
        
        /* Primary button */
        .stButton > button[kind="primary"] {
            background: var(--primary-blue-text);
            color: #ffffff !important;
            border: none;
        }
        .stButton > button[kind="primary"] * { color: #ffffff !important; }
        .stButton > button[kind="primary"]:hover {
            background: #174ea6;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        
        /* Disabled button */
        .stButton > button:disabled {
            background: #f1f3f4;
            color: #bdc1c6 !important;
            border-color: #f1f3f4;
        }
        .stButton > button:disabled * { color: #bdc1c6 !important; }

        /* Inputs */
        [data-testid="stFileUploader"] {
            background: transparent !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background-color: var(--primary-blue) !important;
            border: 1px dashed #aecbfa !important;
            border-radius: 8px !important;
            transition: border-color 0.2s ease, background 0.2s ease !important;
            color: var(--primary-blue-text) !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--primary-blue-text) !important;
            background-color: #d2e3fc !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--primary-blue-text) !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            background-color: #ffffff !important;
            color: var(--primary-blue-text) !important;
            border: 1px solid #d2e3fc !important;
            border-radius: 6px !important;
            padding: 0.25rem 1rem !important;
            font-weight: 500 !important;
            margin-top: 0.5rem !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            background-color: #f8f9fa !important;
            border-color: var(--primary-blue-text) !important;
        }

        /* Uploaded File Chips */
        [data-testid="stUploadedFile"] {
            background-color: #ffffff !important;
            border: 1px solid #d2e3fc !important;
            border-radius: 6px !important;
        }
        [data-testid="stUploadedFile"] div, [data-testid="stUploadedFile"] span {
            color: var(--primary-blue-text) !important;
        }
        [data-testid="stUploadedFile"] button {
            background-color: #f1f3f4 !important;
            border: none !important;
            color: #5f6368 !important;
        }
        [data-testid="stUploadedFile"] button:hover {
            background-color: #e8eaed !important;
        }

        [data-testid="stTextInput"] input {
            background: #ffffff;
            color: var(--text-main);
            border-radius: 4px;
            border: 1px solid var(--border-color);
            padding: 0.6rem 1rem;
            transition: border-color 0.2s;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: var(--primary-blue-text);
            outline: none;
        }

        /* Animations */
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse-arrow { 0% { transform: translateX(0px); opacity: 0.3; } 50% { transform: translateX(5px); opacity: 0.8; } 100% { transform: translateX(0px); opacity: 0.3; } }

        /* Progress Bar */
        .progress-bar-container {
            width: 100%;
            background: #e8eaed;
            border-radius: 4px;
            height: 8px;
            margin-top: 8px;
            margin-bottom: 16px;
            overflow: hidden;
            position: relative;
        }
        .progress-bar-fill {
            height: 100%;
            background: var(--primary-blue-text);
            border-radius: 4px;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .prob-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
            font-size: 0.95rem;
            color: var(--text-main);
        }
        .prob-name { font-weight: 500; }
        .prob-value { color: var(--primary-blue-text); font-weight: 700; }

        /* Teachable Machine Split Layout */
        .webcam-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: var(--primary-blue-text);
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        .webcam-close {
            cursor: pointer;
            font-size: 1.2rem;
            color: var(--primary-blue-text);
        }
        .gallery-header {
            color: var(--text-main);
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 4px;
            max-height: 400px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .gallery-grid img {
            width: 100%;
            aspect-ratio: 1 / 1;
            object-fit: cover;
            border-radius: 4px;
            background: #e8eaed;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(label: str, status_type: str) -> None:
    css_class = {
        "success": "success-pill",
        "warning": "warning-pill",
        "danger": "danger-pill",
    }.get(status_type, "status-pill")
    st.markdown(f"<span class='{css_class}'>{label}</span>", unsafe_allow_html=True)


def render_metric_card(label: str, value: str | int, helper_text: str | None = None) -> None:
    st.metric(label, value)
    if helper_text:
        st.caption(helper_text)


def render_empty_state(title: str, message: str, tone: str = "empty") -> None:
    css_class = {
        "warning": "warning-card",
        "success": "success-card",
    }.get(tone, "empty-card")
    st.markdown(
        f"""
        <div class="{css_class}">
            <strong>{title}</strong>
            <div class="small-caption">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def extract_error_message(response: Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or "Something went wrong. Please try again."

    detail = body.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        return " ".join(str(item.get("msg", item)) for item in detail)
    return response.text or "Something went wrong. Please try again."


def request_json(method: str, url: str, *, timeout: int, **kwargs: Any) -> dict:
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as error:
        raise ValueError("Could not connect to the backend. Make sure FastAPI is running.") from error
    except requests.exceptions.Timeout as error:
        raise ValueError("The backend request timed out. Please try again.") from error
    except requests.RequestException as error:
        raise ValueError(f"Request failed: {error}") from error

    if response.status_code != 200:
        raise ValueError(extract_error_message(response))

    try:
        return response.json()
    except ValueError as error:
        raise ValueError("Backend returned an invalid response.") from error


def frame_to_jpeg_bytes(frame_ndarray: np.ndarray) -> bytes:
    if frame_ndarray.ndim != 3:
        raise ValueError("Invalid webcam frame.")

    rgb_frame = cv2.cvtColor(frame_ndarray, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    return pil_to_jpeg_bytes(image)


def pil_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def upload_webcam_frame(class_name: str, image_bytes: bytes) -> dict:
    files = [("files", ("webcam_sample.jpg", image_bytes, "image/jpeg"))]
    return request_json(
        "POST",
        f"{BACKEND_URL}/upload-sample",
        data={"class_name": class_name},
        files=files,
        timeout=30,
    )


def predict_webcam_frame(image_bytes: bytes) -> dict:
    files = {"file": ("webcam_prediction.jpg", image_bytes, "image/jpeg")}
    return request_json("POST", f"{BACKEND_URL}/predict", files=files, timeout=30)


class WebcamCaptureProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_frame_bytes: bytes | None = None
        self.is_recording = False
        self.class_name: str | None = None
        self.interval = 0.7
        self.last_capture_time = 0.0
        self.captured_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        now = time.time()
        try:
            jpeg_bytes = frame_to_jpeg_bytes(image)
            with self.lock:
                self.latest_frame_bytes = jpeg_bytes

            if self.is_recording and self.class_name:
                if now - self.last_capture_time >= self.interval:
                    self.last_capture_time = now
                    
                    def _upload(c_name: str, data: bytes) -> None:
                        try:
                            upload_webcam_frame(c_name, data)
                            with self.lock:
                                self.captured_count += 1
                        except Exception:
                            pass

                    threading.Thread(target=_upload, args=(self.class_name, jpeg_bytes), daemon=True).start()
        except ValueError:
            pass
        return frame

    def get_latest_frame_bytes(self) -> bytes | None:
        with self.lock:
            return self.latest_frame_bytes


class LivePredictionProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_prediction: dict | None = None
        self.latest_error: str | None = None
        self.last_prediction_time = 0.0
        self.prediction_interval = 1.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        now = time.time()
        if now - self.last_prediction_time < self.prediction_interval:
            return frame

        self.last_prediction_time = now
        image = frame.to_ndarray(format="bgr24")

        try:
            image_bytes = frame_to_jpeg_bytes(image)
            prediction = predict_webcam_frame(image_bytes)
            with self.lock:
                self.latest_prediction = prediction
                self.latest_error = None
        except Exception as error:
            with self.lock:
                self.latest_error = str(error)

        return frame

    def get_latest_prediction(self) -> dict | None:
        with self.lock:
            return self.latest_prediction

    def get_latest_error(self) -> str | None:
        with self.lock:
            return self.latest_error


def check_backend_health() -> bool:
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def default_class_inputs() -> list[dict[str, str]]:
    return [
        {"id": uuid4().hex, "name": "Class 1"},
        {"id": uuid4().hex, "name": "Class 2"},
    ]


def initialize_session_state() -> None:
    defaults = {
        "class_inputs": default_class_inputs(),
        "uploaded_classes": [],
        "model_trained": False,
        "last_prediction": None,
        "dataset_summary": None,
        "model_status": None,
        "selected_test_image": None,
        "data_changed_after_training": False,
        "backend_reset_for_session": False,
        "webcam_capture_active": False,
        "webcam_capture_class_id": None,
        "webcam_capture_class_name": None,
        "webcam_captured_count": 0,
        "webcam_last_frame": None,
        "live_prediction_result": None,
        "live_prediction_enabled": False,
        "live_prediction_last_time": 0.0,
        "webcam_capture_interval": 0.7,
        "last_capture_time": 0.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_dataset_summary() -> dict:
    return request_json("GET", f"{BACKEND_URL}/dataset-summary", timeout=10)


def get_model_status() -> dict:
    return request_json("GET", f"{BACKEND_URL}/model-status", timeout=10)


def reset_backend_state() -> dict:
    return request_json("POST", f"{BACKEND_URL}/reset-app", timeout=30)


def reset_frontend_workbench_state() -> None:
    st.session_state.class_inputs = default_class_inputs()
    st.session_state.uploaded_classes = []
    st.session_state.model_trained = False
    st.session_state.last_prediction = None
    st.session_state.dataset_summary = None
    st.session_state.model_status = None
    st.session_state.selected_test_image = None
    st.session_state.data_changed_after_training = False
    st.session_state.webcam_capture_active = False
    st.session_state.webcam_capture_class_id = None
    st.session_state.webcam_capture_class_name = None
    st.session_state.webcam_captured_count = 0
    st.session_state.webcam_last_frame = None
    st.session_state.live_prediction_result = None
    st.session_state.live_prediction_enabled = False
    st.session_state.live_prediction_last_time = 0.0
    st.session_state.webcam_capture_interval = 0.7
    st.session_state.last_capture_time = 0.0


def refresh_app_state_from_backend() -> None:
    dataset_summary = get_dataset_summary()
    model_status = get_model_status()

    st.session_state.dataset_summary = dataset_summary
    st.session_state.model_status = model_status
    st.session_state.uploaded_classes = [item["class_name"] for item in dataset_summary.get("classes", [])]
    st.session_state.model_trained = bool(model_status.get("model_exists")) and not st.session_state.data_changed_after_training


def upload_samples(class_name: str, uploaded_files: list[Any]) -> dict:
    files = [
        (
            "files",
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            ),
        )
        for uploaded_file in uploaded_files
    ]

    return request_json(
        "POST",
        f"{BACKEND_URL}/upload-sample",
        data={"class_name": class_name},
        files=files,
        timeout=30,
    )


def train_model() -> dict:
    return request_json("POST", f"{BACKEND_URL}/train", timeout=300)


def predict_image(image_file: Any) -> dict:
    files = {
        "file": (
            image_file.name,
            image_file.getvalue(),
            image_file.type or "application/octet-stream",
        )
    }
    return request_json("POST", f"{BACKEND_URL}/predict", files=files, timeout=30)


def add_new_class() -> None:
    next_number = len(st.session_state.class_inputs) + 1
    st.session_state.class_inputs.append({"id": uuid4().hex, "name": f"Class {next_number}"})


def remove_class(class_id: str) -> None:
    if len(st.session_state.class_inputs) <= 2:
        return
    st.session_state.class_inputs = [
        class_item for class_item in st.session_state.class_inputs if class_item["id"] != class_id
    ]


def reset_ui_state() -> None:
    st.session_state.class_inputs = default_class_inputs()
    st.session_state.uploaded_classes = []
    st.session_state.last_prediction = None
    st.session_state.selected_test_image = None
    st.session_state.data_changed_after_training = False
    refresh_app_state_from_backend()


def training_readiness() -> tuple[bool, list[tuple[str, bool]]]:
    dataset_summary = st.session_state.dataset_summary or {"total_classes": 0, "classes": []}
    has_two_classes = dataset_summary.get("total_classes", 0) >= 2
    has_two_images_each = bool(dataset_summary.get("classes")) and all(
        class_item.get("image_count", 0) >= 2 for class_item in dataset_summary.get("classes", [])
    )
    backend_connected = bool(st.session_state.get("backend_connected", False))

    checks = [
        ("At least 2 classes uploaded", has_two_classes),
        ("Each class has at least 2 images", has_two_images_each),
        ("Backend connected", backend_connected),
    ]
    return all(status for _, status in checks), checks


def render_hero(backend_online: bool) -> None:
    status_label = "Connected" if backend_online else "Offline"
    status_class = "success-pill" if backend_online else "danger-pill"
    st.markdown(
        f"""
        <div class="hero-card">
            <span class="{status_class}">Backend {status_label}</span>
            <h1>Build Your Own Teachable Machine</h1>
            <p>Train a custom image classifier in minutes — upload examples, teach the model, and test predictions instantly.</p>
            <span class="status-pill">FastAPI Backend</span>
            <span class="status-pill">Streamlit UI</span>
            <span class="status-pill">MobileNetV3</span>
            <span class="status-pill">Real-time Prediction</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Control Center")
        st.markdown("### Workflow")
        st.markdown(
            """
            1. Add classes
            2. Upload image examples
            3. Train model
            4. Test prediction
            """
        )

        st.divider()
        st.markdown("### Backend")
        st.caption("Backend URL")
        st.code(BACKEND_URL)
        render_status_pill("Connected" if st.session_state.backend_connected else "Offline", "success" if st.session_state.backend_connected else "danger")

        st.divider()
        st.markdown("### Actions")
        if st.button("Refresh Status", use_container_width=True):
            try:
                refresh_app_state_from_backend()
                st.success("Status refreshed.")
            except ValueError as error:
                st.error(str(error))

        st.caption("Reset only clears the UI. Uploaded images and trained model remain saved in the backend.")
        if st.button("Reset UI State", use_container_width=True):
            try:
                reset_ui_state()
                st.success("UI state reset.")
                st.rerun()
            except ValueError as error:
                st.error(str(error))


def render_dashboard() -> None:
    dataset_summary = st.session_state.dataset_summary or {"total_classes": 0, "total_images": 0, "classes": []}
    model_status = st.session_state.model_status or {"model_exists": False}
    is_ready, _ = training_readiness()

    columns = st.columns(4)
    with columns[0]:
        render_metric_card("Total Classes", dataset_summary.get("total_classes", 0), "Classes in backend dataset")
    with columns[1]:
        render_metric_card("Total Images", dataset_summary.get("total_images", 0), "Valid training examples")
    with columns[2]:
        render_metric_card("Model Status", "Trained" if model_status.get("model_exists") else "Not Trained", "Saved model.pkl")
    with columns[3]:
        render_metric_card("Training Readiness", "Ready" if is_ready else "Needs Data", "Minimum data checks")

    if st.session_state.data_changed_after_training:
        render_empty_state(
            "Retraining recommended",
            "New training data was added after the last training run. Retrain to update predictions.",
            tone="warning",
        )


def render_class_card(class_item: dict[str, str], index: int) -> None:
    class_id = class_item["id"]
    name_key = f"class_name_{class_id}"
    files_key = f"files_{class_id}"
    upload_key = f"upload_btn_{class_id}"
    remove_key = f"remove_btn_{class_id}"

    if name_key not in st.session_state:
        st.session_state[name_key] = class_item["name"]

    st.markdown(
        f"""
        <div class="class-card">
            <span class="step-badge">Class {index + 1}</span>
            <div class="small-caption">Create or rename this class, then upload image examples for it.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card_columns = st.columns([1.1, 1.8, 0.7])
    with card_columns[0]:
        class_name = st.text_input("Class Name", key=name_key, max_chars=50)
        class_item["name"] = class_name

    with card_columns[1]:
        uploaded_files = st.file_uploader(
            "Upload examples",
            type=ALLOWED_IMAGE_TYPES,
            accept_multiple_files=True,
            key=files_key,
        )

    with card_columns[2]:
        st.write("")
        st.write("")
        upload_label = f"Upload {class_name.strip() or 'Class'} Samples"
        if st.button(upload_label, key=upload_key, use_container_width=True):
            if not class_name.strip():
                st.warning("Please enter a class name before uploading.")
            elif not uploaded_files:
                st.warning("Please select at least one image file.")
            else:
                model_exists_before_upload = bool((st.session_state.model_status or {}).get("model_exists"))
                try:
                    with st.spinner(f"Uploading samples for {class_name.strip()}..."):
                        result = upload_samples(class_name.strip(), uploaded_files)

                    if model_exists_before_upload:
                        st.session_state.data_changed_after_training = True
                        st.session_state.model_trained = False

                    refresh_app_state_from_backend()
                    st.success(f"Uploaded {result['saved_count']} image(s) for '{result['class_name']}'.")
                except ValueError as error:
                    st.error(str(error))

        if len(st.session_state.class_inputs) > 2:
            if st.button("Remove", key=remove_key, use_container_width=True):
                remove_class(class_id)
                st.rerun()


def render_dataset_table() -> None:
    dataset_summary = st.session_state.dataset_summary or {"classes": []}
    classes = dataset_summary.get("classes", [])

    st.markdown('<div class="section-title">Current Dataset</div>', unsafe_allow_html=True)
    if not classes:
        render_empty_state(
            "No training data uploaded yet.",
            "Add classes above and upload examples to build your dataset.",
        )
        return

    table = pd.DataFrame(classes).rename(columns={"class_name": "Class Name", "image_count": "Image Count"})
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Each class should have at least 2 images. For better results, use 10-25+ images per class.")


def render_dataset_studio() -> None:
    st.markdown('<div class="section-title">Dataset Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted-text">Create classes and upload example images for each category.</div>', unsafe_allow_html=True)

    top_columns = st.columns([1, 4])
    with top_columns[0]:
        if st.button("+ Add New Class", type="primary", use_container_width=True):
            add_new_class()
            st.rerun()

    for index, class_item in enumerate(st.session_state.class_inputs):
        render_class_card(class_item, index)

    st.divider()
    render_dataset_table()


def render_builder_class_card(class_item: dict[str, str], index: int) -> None:
    class_id = class_item["id"]
    name_key = f"class_name_{class_id}"
    files_key = f"files_{class_id}"
    upload_key = f"upload_btn_{class_id}"
    remove_key = f"remove_btn_{class_id}"
    webcam_key = f"open_webcam_{class_id}"

    if name_key not in st.session_state:
        st.session_state[name_key] = class_item["name"]

    with st.container(border=True):
        st.markdown("<div class='st-card-marker'></div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-top: 0;'>Class {index + 1}</h3>", unsafe_allow_html=True)
        st.caption("Rename this class and add image samples.")

        class_name = st.text_input("Class name", key=name_key, max_chars=50)
        class_item["name"] = class_name

        uploaded_files = st.file_uploader(
            "Add image samples",
            type=ALLOWED_IMAGE_TYPES,
            accept_multiple_files=True,
            key=files_key,
        )

        button_columns = st.columns([1, 1, 1])
        with button_columns[0]:
            if st.button("Upload", key=upload_key, use_container_width=True):
                if not class_name.strip():
                    st.warning("Please enter a class name before uploading.")
                elif not uploaded_files:
                    st.warning("Please select at least one image file.")
                else:
                    model_exists_before_upload = bool((st.session_state.model_status or {}).get("model_exists"))
                    try:
                        with st.spinner(f"Uploading {class_name.strip()} samples..."):
                            result = upload_samples(class_name.strip(), uploaded_files)

                        if model_exists_before_upload:
                            st.session_state.data_changed_after_training = True
                            st.session_state.model_trained = False

                        refresh_app_state_from_backend()
                        st.success(f"Uploaded {result['saved_count']} image(s).")
                    except ValueError as error:
                        st.error(str(error))

        with button_columns[1]:
            if st.button("Webcam", key=webcam_key, use_container_width=True):
                if not class_name.strip():
                    st.warning("Please enter a class name before opening webcam.")
                else:
                    st.session_state.webcam_capture_active = False
                    st.session_state.webcam_capture_class_id = class_id
                    st.session_state.webcam_capture_class_name = class_name.strip()
                    st.session_state.webcam_captured_count = 0
                    st.session_state.last_capture_time = 0.0
                    st.rerun()

        with button_columns[2]:
            if len(st.session_state.class_inputs) > 2:
                if st.button("Remove", key=remove_key, use_container_width=True):
                    remove_class(class_id)
                    st.rerun()

        if st.session_state.get("webcam_capture_class_id") == class_id:
            st.divider()
            render_inline_webcam_capture(class_id, class_name)


def render_training_node() -> None:
    is_ready, checks = training_readiness()

    with st.container(border=True):
        st.markdown("<div class='st-card-marker'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0;'>Training</h3>", unsafe_allow_html=True)
        st.caption("Teach the classifier from uploaded examples.")

        for label, status in checks:
            render_status_pill(label, "success" if status else "warning")

        if not is_ready:
            render_empty_state(
                "Training is not ready yet.",
                "Add at least two classes with two or more images each.",
                tone="warning",
            )

        if st.button("Train Model", type="primary", disabled=not is_ready, use_container_width=True):
            try:
                with st.spinner("Training model with MobileNetV3 feature extraction..."):
                    result = train_model()

                st.session_state.data_changed_after_training = False
                refresh_app_state_from_backend()
                st.session_state.model_trained = True
                st.success(result["message"])
                st.metric("Total Images", result.get("total_images", 0))
                st.caption(f"Classes: {', '.join(result.get('classes', []))}")
            except ValueError as error:
                st.error(str(error))

        with st.expander("Advanced"):
            st.write(
                "MobileNetV3 extracts visual feature vectors. Logistic Regression learns the class boundaries and saves a lightweight model package."
            )


def render_inline_webcam_capture(class_id: str, class_name: str) -> None:
    # 2-column layout for webcam and gallery
    left_col, right_col = st.columns([1, 1], gap="medium")

    with left_col:
        st.markdown(
            f"""
            <div class="webcam-header">
                <div>Webcam</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        interval = 0.5
        webrtc_ctx = webrtc_streamer(
            key=f"webcam_{class_id}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=WebcamCaptureProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        is_recording = st.session_state.webcam_capture_active

        if is_recording:
            if st.button("Stop Recording", type="secondary", key=f"stop_{class_id}", use_container_width=True):
                st.session_state.webcam_capture_active = False
                st.rerun()
        else:
            if st.button("Hold to Record", type="primary", key=f"start_{class_id}", use_container_width=True):
                st.session_state.webcam_capture_active = True
                st.session_state.last_capture_time = 0.0
                st.rerun()
                
        if st.button("Close Webcam", type="secondary", key=f"close_{class_id}", use_container_width=True):
            st.session_state.webcam_capture_active = False
            st.session_state.webcam_capture_class_id = None
            st.rerun()

    # Right Column: Gallery
    with right_col:
        # Find the class summary to get the images
        class_summary = next(
            (c for c in (st.session_state.dataset_summary or {}).get("classes", []) if c["class_name"] == class_name),
            None,
        )
        image_count = class_summary["image_count"] if class_summary else 0
        images = class_summary.get("images", []) if class_summary else []

        st.markdown(f'<div class="gallery-header">{image_count} Image Samples</div>', unsafe_allow_html=True)

        if images:
            # Build the grid HTML using the new /dataset static route
            backend_url = os.getenv("API_URL", "http://localhost:8000")
            grid_html = '<div class="gallery-grid">'
            for img_name in images:
                # Construct absolute URL to backend static files
                img_url = f"{backend_url}/dataset/{class_name}/{img_name}"
                grid_html += f'<img src="{img_url}" alt="Sample">'
            grid_html += '</div>'
            st.markdown(grid_html, unsafe_allow_html=True)
        else:
            st.caption("No samples yet. Click 'Hold to Record' to collect some!")

    # Background processing sync
    processor = webrtc_ctx.video_processor if webrtc_ctx else None
    if is_recording:
        if processor is None:
            time.sleep(0.5)
            st.rerun()

        # Sync state to background processor
        processor.class_name = class_name
        processor.interval = interval
        processor.is_recording = is_recording

        # Sync count from background processor to frontend
        with processor.lock:
            # Only update UI when capturing happens to prevent flicker
            pass 

        # Slow refresh to update UI and gallery without locking up Streamlit buttons
        time.sleep(1.0)
        refresh_app_state_from_backend()
        st.rerun()


def render_preview_node() -> None:
    model_status = st.session_state.model_status or {"model_exists": False}

    with st.container(border=True):
        st.markdown("<div class='st-card-marker'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0;'>Preview</h3>", unsafe_allow_html=True)
        st.caption("Test the trained model with a new image.")

        if not model_status.get("model_exists"):
            render_empty_state(
                "Prediction is locked.",
                "Train a model on the left before you can preview it here.",
                tone="warning",
            )
            return

        if st.session_state.data_changed_after_training:
            render_empty_state(
                "Retrain recommended.",
                "New data was uploaded after training. Preview uses the last trained model.",
                tone="warning",
            )

        upload_tab, camera_tab, live_tab = st.tabs(["Upload", "Camera", "Live Webcam"])
        with upload_tab:
            uploaded_test_image = st.file_uploader(
                "Test image",
                type=ALLOWED_IMAGE_TYPES,
                accept_multiple_files=False,
                key="test_image_upload",
            )

        with camera_tab:
            camera_image = st.camera_input("Capture image", key="camera_image")

        selected_image = uploaded_test_image or camera_image
        if uploaded_test_image is not None and camera_image is not None:
            st.info("Using uploaded image for prediction.")

        if selected_image is not None:
            st.image(selected_image, caption="Selected image", use_container_width=True)
        else:
            render_empty_state("No preview image.", "Upload or capture an image to test the model.")

        if st.button("Run Prediction", type="primary", use_container_width=True):
            if selected_image is None:
                st.warning("Please upload or capture one image before predicting.")
            else:
                try:
                    with st.spinner("Analyzing image..."):
                        st.session_state.last_prediction = predict_image(selected_image)
                except ValueError as error:
                    st.error(str(error))

        if st.session_state.last_prediction:
            display_prediction_result(st.session_state.last_prediction)

        with live_tab:
            st.caption("For performance, live prediction updates about once per second.")

            live_columns = st.columns(2)
            with live_columns[0]:
                if st.button(
                    "Enable Live Prediction",
                    disabled=st.session_state.live_prediction_enabled,
                    use_container_width=True,
                ):
                    st.session_state.live_prediction_enabled = True
                    st.session_state.live_prediction_result = None
                    st.rerun()

            with live_columns[1]:
                if st.button(
                    "Disable Live Prediction",
                    disabled=not st.session_state.live_prediction_enabled,
                    use_container_width=True,
                ):
                    st.session_state.live_prediction_enabled = False
                    st.rerun()

            render_status_pill(
                "Running" if st.session_state.live_prediction_enabled else "Waiting",
                "success" if st.session_state.live_prediction_enabled else "warning",
            )

            if st.session_state.live_prediction_enabled:
                live_ctx = webrtc_streamer(
                    key="live_prediction_webcam",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=LivePredictionProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )

                processor = live_ctx.video_processor
                if processor:
                    latest_error = processor.get_latest_error()
                    latest_prediction = processor.get_latest_prediction()
                    if latest_error:
                        st.warning(latest_error)
                    if latest_prediction:
                        st.session_state.live_prediction_result = latest_prediction
                        render_status_pill("Prediction available", "success")
                        display_prediction_result(latest_prediction)
                    else:
                        st.info("Waiting for webcam prediction...")
                else:
                    st.info("Allow camera permission in your browser. Waiting for webcam frame...")

                time.sleep(1.0)
                st.rerun()
            else:
                render_empty_state(
                    "Live webcam is paused.",
                    "Enable live prediction to stream frames to the trained model.",
                )


def render_model_builder_workspace() -> None:
    st.markdown('<div class="builder-shell">', unsafe_allow_html=True)
    left, arrow_one, middle, arrow_two, right = st.columns([2.5, 0.22, 1.25, 0.22, 1.6])

    with left:
        st.markdown('<div class="flow-label">Dataset Classes</div>', unsafe_allow_html=True)
        for index, class_item in enumerate(st.session_state.class_inputs):
            render_builder_class_card(class_item, index)

        st.markdown('<div class="add-class-node">', unsafe_allow_html=True)
        if st.button("+ Add a class", type="primary", use_container_width=True):
            add_new_class()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


    with arrow_one:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

    with middle:
        st.markdown('<div class="flow-label">Model</div>', unsafe_allow_html=True)
        render_training_node()

    with arrow_two:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="flow-label">Output</div>', unsafe_allow_html=True)
        render_preview_node()

    st.markdown("</div>", unsafe_allow_html=True)


def render_training_tab() -> None:
    st.markdown('<div class="section-title">Train Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted-text">Use uploaded examples to teach the classifier.</div>', unsafe_allow_html=True)

    is_ready, checks = training_readiness()
    st.markdown("#### Training readiness")
    for label, status in checks:
        render_status_pill(label, "success" if status else "warning")

    if not is_ready:
        render_empty_state(
            "Training is not ready yet.",
            "Add at least two classes with two or more images each.",
            tone="warning",
        )

    if st.button("Train AI Model", type="primary", disabled=not is_ready):
        try:
            with st.spinner("Training model with MobileNetV3 feature extraction..."):
                result = train_model()

            st.session_state.data_changed_after_training = False
            refresh_app_state_from_backend()
            st.session_state.model_trained = True
            st.success(result["message"])

            result_columns = st.columns(2)
            result_columns[0].metric("Trained Classes", ", ".join(result.get("classes", [])))
            result_columns[1].metric("Total Images", result.get("total_images", 0))
        except ValueError as error:
            st.error(str(error))

    with st.expander("How training works"):
        st.write(
            "MobileNetV3 reads each image and extracts feature vectors. "
            "Logistic Regression learns simple class boundaries from those vectors. "
            "The saved model.pkl stores the lightweight classifier and metadata."
        )


def display_prediction_result(result: dict) -> None:
    predicted_class = result.get("predicted_class", "Unknown")
    confidence = float(result.get("confidence", 0))
    probabilities = result.get("probabilities", {})

    st.markdown(
        f"""
<div class="prediction-card">
    <span class="step-badge">Prediction Result</span>
    <h2 style="margin: 0.5rem 0 0.25rem 0; font-size: 2.2rem; color: #202124;">{predicted_class}</h2>
    <div class="small-caption" style="margin-bottom: 1.25rem; font-size: 0.95rem;">Confidence: <span style="color: #1967d2; font-weight: 600;">{confidence:.2f}%</span></div>
    <div class="progress-bar-container">
        <div class="progress-bar-fill" style="width: {min(max(confidence, 0.0), 100.0)}%;"></div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Probability breakdown")
    for class_name, probability in probabilities.items():
        probability_value = float(probability)
        st.markdown(
            f"""
<div style="margin-bottom: 12px;">
    <div class="prob-row">
        <span class="prob-name">{class_name}</span>
        <span class="prob-value">{probability_value:.2f}%</span>
    </div>
    <div class="progress-bar-container" style="margin-bottom: 4px; height: 6px;">
        <div class="progress-bar-fill" style="width: {min(max(probability_value, 0.0), 100.0)}%; opacity: 0.9;"></div>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_tab() -> None:
    st.markdown('<div class="section-title">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted-text">Upload or capture a new image and see how the trained model classifies it.</div>',
        unsafe_allow_html=True,
    )

    model_status = st.session_state.model_status or {"model_exists": False}
    if not model_status.get("model_exists"):
        render_empty_state(
            "Prediction is locked.",
            "Train a model first to unlock live testing.",
            tone="warning",
        )
        return

    if st.session_state.data_changed_after_training:
        render_empty_state(
            "Predictions may be outdated.",
            "New data was uploaded after training. Retrain when you want updated results.",
            tone="warning",
        )

    input_column, result_column = st.columns([1, 1.1])

    with input_column:
        st.markdown("#### Test Input")
        upload_tab, camera_tab = st.tabs(["Upload Image", "Camera"])

        with upload_tab:
            uploaded_test_image = st.file_uploader(
                "Choose one test image",
                type=ALLOWED_IMAGE_TYPES,
                accept_multiple_files=False,
                key="test_image_upload",
            )

        with camera_tab:
            camera_image = st.camera_input("Capture an image", key="camera_image")

        selected_image = uploaded_test_image or camera_image
        if uploaded_test_image is not None and camera_image is not None:
            st.info("Using uploaded image for prediction.")

        if selected_image is not None:
            st.session_state.selected_test_image = selected_image
            st.image(selected_image, caption="Selected image", use_container_width=True)
        else:
            render_empty_state("No image selected.", "Upload or capture an image to run prediction.")

        if st.button("Run Prediction", type="primary", use_container_width=True):
            if selected_image is None:
                st.warning("Please upload or capture one image before predicting.")
            else:
                try:
                    with st.spinner("Analyzing image..."):
                        st.session_state.last_prediction = predict_image(selected_image)
                except ValueError as error:
                    st.error(str(error))

    with result_column:
        st.markdown("#### Prediction Result")
        if st.session_state.last_prediction:
            display_prediction_result(st.session_state.last_prediction)
        else:
            render_empty_state("Prediction results will appear here.", "Run a prediction to see confidence and class probabilities.")


def main() -> None:
    inject_custom_css()
    initialize_session_state()

    backend_online = check_backend_health()
    st.session_state.backend_connected = backend_online
    render_hero(backend_online)

    if not backend_online:
        render_empty_state(
            "Backend is not running.",
            f"Start FastAPI first on {BACKEND_URL}",
            tone="warning",
        )
        st.stop()

    if not st.session_state.backend_reset_for_session:
        try:
            reset_backend_state()
            reset_frontend_workbench_state()
            st.session_state.backend_reset_for_session = True
        except ValueError as error:
            st.error(str(error))
            st.stop()

    try:
        refresh_app_state_from_backend()
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.markdown('<div class="section-title">Model Builder</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted-text">Add sample classes, train the model, then preview predictions in one connected workspace.</div>',
        unsafe_allow_html=True,
    )
    render_model_builder_workspace()


if __name__ == "__main__":
    main()
