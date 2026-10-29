"""
VigiLens Streamlit application.

Upload a visible video (and optionally an aligned IR/thermal video) and watch
the frame-level anomaly score, status and FPS update in real time.
"""

from __future__ import annotations

import tempfile
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from config.config import get_config
from models.anomaly_model import VigiLensModel
from models.segmentation import YOLOSegmentation
from pipeline.frame_source import create_frame_source
from pipeline.inference import InferencePipeline
from training.calibration import AnomalyCalibrator

st.set_page_config(page_title="VigiLens", layout="wide")

st.markdown(
    """
    <style>
    .metric-box { padding: 10px; border-radius: 8px; background-color: #1c1f26; }
    </style>
    """,
    unsafe_allow_html=True,
)

col_title, col_status = st.columns([6, 1])
with col_title:
    st.markdown("### VigiLens")
    st.caption("Multimodal Visible + IR Anomaly Detection System")
with col_status:
    st.success("Active")

st.markdown("---")


def _mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


@st.cache_resource(show_spinner=False)
def load_runtime(device: str,
                 checkpoint_path: str,
                 calibration_path: str,
                 checkpoint_mtime: float,
                 calibration_mtime: float,
                 yolo_model_path: str):
    """Load model, calibration and YOLO once per checkpoint/device."""
    del checkpoint_mtime, calibration_mtime  # cache-key only

    model = VigiLensModel(device=device)
    messages = []
    if Path(checkpoint_path).exists():
        model.load_checkpoint(checkpoint_path)
        messages.append(f"Checkpoint loaded: {checkpoint_path}")
    else:
        messages.append(f"Checkpoint not found: {checkpoint_path}. Using untrained model.")

    calibration_info = None
    if Path(calibration_path).exists():
        calibration_info = AnomalyCalibrator.load_calibration(calibration_path)
        messages.append(
            "Calibration loaded: threshold="
            f"{calibration_info.get('threshold', calibration_info.get('threshold_std', 0.5)):.6f}"
        )
    else:
        messages.append(f"Calibration not found: {calibration_path}. Using default threshold.")

    segmentation = YOLOSegmentation(model_path=yolo_model_path, device=device)
    return model, segmentation, calibration_info, messages


# ------------------ Sidebar ------------------
st.sidebar.header("System Configuration")
st.sidebar.subheader("Processing")
cuda_available = torch.cuda.is_available()
device = st.sidebar.selectbox(
    "Device",
    ["cuda"] if cuda_available else ["cpu"],
    index=0,
    disabled=True,
)
if cuda_available:
    st.sidebar.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("CUDA not available. Using CPU.")

config = get_config()
st.sidebar.subheader("Model")
checkpoint_path = st.sidebar.text_input("Checkpoint Path", config.inference.checkpoint_path)
calibration_path = st.sidebar.text_input("Calibration Path", config.calibration.calibration_path)
yolo_model_path = st.sidebar.text_input("YOLO Model", config.inference.yolo_model_path)

st.sidebar.subheader("Temporal Smoothing")
smoothing_method = st.sidebar.selectbox("Smoothing Method", ["moving_average", "exponential"])
window_size = st.sidebar.slider("Window Size", 1, 30, config.inference.smoothing_window)

st.sidebar.subheader("Display")
show_ir = st.sidebar.checkbox("Show IR Frame", value=True)
show_mask = st.sidebar.checkbox("Show Segmentation Mask", value=False)

if st.sidebar.button("Reload models"):
    load_runtime.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.text("Model: ResNet50 + LSTM")
st.sidebar.text("Method: Next-feature prediction")
st.sidebar.text("Training: Normal-only")
st.sidebar.text("Fusion: Modality-aware")

# ------------------ Inputs ------------------
st.subheader("Input Sources")
col_vis, col_ir = st.columns(2)
with col_vis:
    visible_file = st.file_uploader("Visible Video", type=["mp4", "avi", "mov", "mkv"], key="visible")
with col_ir:
    ir_file = st.file_uploader("IR/Thermal Video (Optional)", type=["mp4", "avi", "mov", "mkv"], key="ir")

if visible_file:
    visible_temp_path = None
    ir_temp_path = None
    frame_source = None
    pipeline = None

    try:
        suffix = Path(visible_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as visible_temp:
            visible_temp.write(visible_file.read())
            visible_temp_path = Path(visible_temp.name)

        if ir_file:
            ir_suffix = Path(ir_file.name).suffix or suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=ir_suffix) as ir_temp:
                ir_temp.write(ir_file.read())
                ir_temp_path = Path(ir_temp.name)
            st.info("IR video loaded - multimodal mode")
        else:
            st.warning("IR video not provided - visible-only mode")

        with st.spinner("Initializing models..."):
            model, segmentation, calibration_info, messages = load_runtime(
                device,
                checkpoint_path,
                calibration_path,
                _mtime(checkpoint_path),
                _mtime(calibration_path),
                yolo_model_path,
            )
            for message in messages:
                st.info(message)
            model.reset_stream()

            pipeline = InferencePipeline(
                model=model,
                segmentation=segmentation,
                calibration_info=calibration_info,
                smoothing_window=window_size,
                smoothing_method=smoothing_method,
            )
            frame_source = create_frame_source(
                str(visible_temp_path), str(ir_temp_path) if ir_temp_path else None
            )

        st.success("Pipeline initialized successfully")

        col_video, col_metrics = st.columns([3, 1])
        with col_video:
            video_placeholder = st.empty()
            ir_placeholder = st.empty() if show_ir and ir_temp_path else None
            mask_placeholder = st.empty() if show_mask else None

        with col_metrics:
            st.subheader("Metrics")
            m_score = st.empty()
            m_status = st.empty()
            m_fps = st.empty()
            m_frame = st.empty()
            m_ir = st.empty()
            st.markdown("---")
            st.subheader("Info")
            m_threshold = st.empty()

        status_box = st.empty()
        frame_count = 0
        score_history = deque(maxlen=50)

        while True:
            visible_frame, ir_frame = frame_source.read()
            if visible_frame is None and ir_frame is None:
                break

            result = pipeline.process_frame(visible_frame, ir_frame)
            mask = result.get("rgb_mask")

            if result["is_anomalous"] and visible_frame is not None:
                cv2.rectangle(
                    visible_frame, (0, 0),
                    (visible_frame.shape[1] - 1, visible_frame.shape[0] - 1),
                    (0, 0, 255), 4,
                )
                cv2.putText(
                    visible_frame, "ANOMALY", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                )
                cv2.putText(
                    visible_frame, f"Score: {result['smoothed_score']:.3f}", (40, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                )

            m_score.metric(
                "Anomaly Score",
                f"{result['smoothed_score']:.4f}" if result["score_available"] else "warming up",
            )
            m_status.metric("Status", "ANOMALY" if result["is_anomalous"] else "NORMAL")
            m_fps.metric("FPS", f"{1.0 / max(result['inference_time'], 1e-6):.1f}")
            m_frame.metric("Frame", frame_count)
            m_ir.metric("IR Available", "Yes" if result["ir_available"] else "No")
            m_threshold.metric("Threshold", f"{pipeline.threshold:.4f}")

            status_box.error("ANOMALY DETECTED") if result["is_anomalous"] else status_box.success("NORMAL")

            if visible_frame is not None:
                display = cv2.resize(visible_frame, (720, 480))
                video_placeholder.image(display, channels="BGR")

            if ir_placeholder is not None and ir_frame is not None:
                ir_display = cv2.resize(ir_frame, (720, 480))
                if ir_display.ndim == 2:
                    ir_placeholder.image(ir_display, channels="GRAY")
                else:
                    ir_placeholder.image(ir_display, channels="BGR")

            if mask_placeholder is not None and mask is not None:
                mask_display = cv2.resize(mask, (720, 480))
                mask_placeholder.image((mask_display * 255).astype(np.uint8), channels="GRAY")

            frame_count += 1
            score_history.append(result["smoothed_score"])

        st.success(f"Processing complete - {frame_count} frames processed")

    except Exception as exc:
        st.error(f"Error: {exc}")
        st.error(traceback.format_exc())
    finally:
        if frame_source is not None:
            try:
                frame_source.release()
            except Exception as exc:
                st.warning(f"Error releasing frame source: {exc}")
        if pipeline is not None:
            try:
                pipeline.reset_stream()
            except Exception as exc:
                st.warning(f"Error resetting pipeline: {exc}")

        def safe_delete(path: Path | None, max_retries: int = 3, delay: float = 0.5) -> None:
            if path is None or not path.exists():
                return
            for attempt in range(max_retries):
                try:
                    path.unlink(missing_ok=True)
                    return
                except PermissionError as exc:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    else:
                        st.warning(f"Could not delete temporary file {path}: {exc}")

        safe_delete(visible_temp_path)
        safe_delete(ir_temp_path)

else:
    st.info("Please upload a visible video to begin analysis")

st.markdown("---")
st.caption("VigiLens | Multimodal Anomaly Detection | NIT Rourkela")
