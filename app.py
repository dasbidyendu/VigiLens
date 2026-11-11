"""
VigiLens Streamlit application.

Upload a visible video (and optionally an aligned IR/thermal video), or use a
webcam, and watch the frame-level anomaly score, status and FPS update live.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from config.config import get_config
from models.anomaly_model import VigiLensModel
from models.segmentation import YOLOSegmentation
from pipeline.frame_source import CameraFrameSource, create_frame_source
from pipeline.inference import InferencePipeline
from training.calibration import AnomalyCalibrator, file_sha256

logger = logging.getLogger("vigilens.app")

REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
# Uploaded videos are written next to the repository (not to the system temp
# directory) unless VIGILENS_UPLOAD_DIR says otherwise.
UPLOAD_DIR = Path(os.environ.get("VIGILENS_UPLOAD_DIR", REPO_ROOT / ".tmp" / "uploads"))

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
    messages = []  # (level, text)
    checkpoint_ok = Path(checkpoint_path).is_file()
    if checkpoint_ok:
        model.load_checkpoint(checkpoint_path)
        messages.append(("info", f"Checkpoint loaded: {Path(checkpoint_path).name}"))
    else:
        messages.append(("warning", f"Checkpoint not found: {checkpoint_path}. "
                                    "Using an UNTRAINED model; scores are meaningless."))
    # Fusion/LSTM on the same device as the features, dropout disabled.
    model.to(model.device)
    model.eval()

    calibration_info = None
    if Path(calibration_path).is_file():
        candidate = AnomalyCalibrator.load_calibration(calibration_path)
        expected = candidate.get("checkpoint_sha256")
        if not checkpoint_ok:
            messages.append(("warning", "Calibration ignored: there is no trained checkpoint to go with it."))
        elif expected and expected != file_sha256(checkpoint_path):
            messages.append(("warning", "Calibration ignored: it was computed for a different "
                                        "checkpoint. Re-run `python -m training.calibration`."))
        else:
            calibration_info = candidate
            if not expected:
                messages.append(("warning", "Calibration file does not record which checkpoint it "
                                            "belongs to; re-run calibration to be sure it matches."))
            messages.append(("info", f"Calibration loaded: {Path(calibration_path).name}"))
    else:
        messages.append(("warning", f"Calibration not found: {calibration_path}. Using default threshold."))

    # Compute features the way the checkpoint was trained: on YOLO person
    # masks, or on the full frame.
    if model.use_segmentation:
        segmentation = YOLOSegmentation(model_path=yolo_model_path, device=device)
    else:
        segmentation = None
        messages.append(("info", "This checkpoint was trained on full-frame features; "
                                 "person segmentation is off."))
    # One shared model instance holds per-stream state, so only one video can
    # be analysed at a time across all browser sessions.
    return model, segmentation, calibration_info, messages, threading.Lock()


def _choices(directory: Path, pattern: str, default: str) -> list:
    # Recursive, because the notebook saves runs to checkpoints/<dataset>/;
    # POSIX paths so the default ("checkpoints/best_model.pth") is not listed
    # twice on Windows.
    found = sorted(path.relative_to(REPO_ROOT).as_posix()
                   for path in directory.rglob(pattern)) if directory.exists() else []
    if default not in found:
        found.insert(0, default)
    return found


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
# Only files inside the repository's checkpoints/ directory can be selected.
checkpoint_choice = st.sidebar.selectbox(
    "Checkpoint", _choices(CHECKPOINT_DIR, "*.pth", config.inference.checkpoint_path)
)
calibration_choice = st.sidebar.selectbox(
    "Calibration", _choices(CHECKPOINT_DIR, "*.json", config.calibration.calibration_path)
)
checkpoint_path = str(REPO_ROOT / checkpoint_choice)
calibration_path = str(REPO_ROOT / calibration_choice)
yolo_model_path = config.inference.yolo_model_path

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

# ------------------ Analysis ------------------
def _start_pipeline():
    """Load the shared runtime, take the run lock, and build a fresh pipeline."""
    with st.spinner("Initializing models..."):
        model, segmentation, calibration_info, messages, lock = load_runtime(
            device,
            checkpoint_path,
            calibration_path,
            _mtime(checkpoint_path),
            _mtime(calibration_path),
            yolo_model_path,
        )
    for level, message in messages:
        (st.warning if level == "warning" else st.info)(message)

    if not lock.acquire(blocking=False):
        st.warning("Another video is being analysed right now. Please wait for it to finish.")
        st.stop()
    try:
        model.reset_stream()
        pipeline = InferencePipeline(
            model=model,
            segmentation=segmentation,
            calibration_info=calibration_info,
            smoothing_window=window_size,
            smoothing_method=smoothing_method,
            ema_alpha=config.inference.ema_alpha,
            default_threshold=config.inference.default_threshold,
            mask_hold_frames=config.inference.mask_hold_frames,
        )
    except BaseException:
        lock.release()
        raise
    if pipeline.threshold_source == "default":
        st.warning("Threshold is an uncalibrated placeholder; ANOMALY/NORMAL labels are not meaningful.")
    elif pipeline.threshold_source == "raw" and (smoothing_method != "moving_average" or window_size > 1):
        st.warning("Calibration only has a raw-score threshold; it does not match the selected "
                   "smoothing. Re-run calibration.")
    return pipeline, lock


def run_analysis(open_source, show_ir_frames: bool, max_seconds: float | None = None) -> None:
    """
    Run the pipeline on a frame source and stream the results to the page.

    ``open_source`` is called after the run lock is taken, so a camera is only
    opened when this session may use it.  Any widget interaction (e.g. the
    webcam Stop button) reruns the script, which interrupts this loop; the
    ``finally`` block still releases the camera and the lock.
    """
    pipeline, run_lock = _start_pipeline()
    frame_source = None
    try:
        frame_source = open_source()
        st.success("Pipeline initialized successfully")

        col_video, col_metrics = st.columns([3, 1])
        with col_video:
            video_placeholder = st.empty()
            ir_placeholder = st.empty() if show_ir_frames else None
            mask_placeholder = st.empty() if show_mask else None
            chart_placeholder = st.empty()

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
        anomaly_count = 0
        score_history = deque(maxlen=300)
        started = time.time()

        while max_seconds is None or time.time() - started < max_seconds:
            visible_frame, ir_frame = frame_source.read()
            if visible_frame is None and ir_frame is None:
                break

            result = pipeline.process_frame(visible_frame, ir_frame)
            mask = result.get("rgb_mask")
            anomaly_count += int(result["is_anomalous"])

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

            # A plain if/else: a bare conditional expression would be echoed
            # onto the page by Streamlit's "magic" display.
            if result["is_anomalous"]:
                status_box.error("ANOMALY DETECTED")
            else:
                status_box.success("NORMAL")

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
            if result["score_available"]:
                score_history.append(result["smoothed_score"])
            if score_history and frame_count % 5 == 0:
                chart_placeholder.line_chart({
                    "score": list(score_history),
                    "threshold": [pipeline.threshold] * len(score_history),
                }, height=200)

        st.success(f"Processing complete - {frame_count} frames processed, "
                   f"{anomaly_count} flagged as anomalous")
    finally:
        run_lock.release()
        if frame_source is not None:
            try:
                frame_source.release()
            except Exception as exc:
                st.warning(f"Error releasing frame source: {exc}")
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


# ------------------ Inputs ------------------
st.subheader("Input Sources")
source_mode = st.radio("Source", ["Video files", "Webcam"], horizontal=True)

if source_mode == "Webcam":
    st.caption("Uses a camera attached to the computer running this app (visible only, no IR). "
               "The score measures how unusual the motion is compared with the training footage, "
               "so a model trained on a different scene will not transfer well.")
    col_cam, col_time = st.columns(2)
    with col_cam:
        camera_index = st.number_input("Camera index", min_value=0, max_value=9, value=0, step=1)
    with col_time:
        max_seconds = st.slider("Run for (seconds)", 10, 600, 60, step=10)
    col_start, col_stop = st.columns([1, 6])
    start_clicked = col_start.button("Start webcam", type="primary")
    col_stop.button("Stop")  # any click reruns the script, which ends a running loop

    if start_clicked:
        try:
            run_analysis(lambda: CameraFrameSource(int(camera_index)), show_ir_frames=False,
                         max_seconds=float(max_seconds))
        except Exception as exc:
            logger.exception("Webcam analysis failed")
            st.error(f"Error: {exc}")
    else:
        st.info("Press **Start webcam** to begin live analysis.")

else:
    col_vis, col_ir = st.columns(2)
    with col_vis:
        visible_file = st.file_uploader("Visible Video", type=["mp4", "avi", "mov", "mkv"], key="visible")
    with col_ir:
        ir_file = st.file_uploader("IR/Thermal Video (Optional)", type=["mp4", "avi", "mov", "mkv"], key="ir")

    if visible_file:
        visible_temp_path = None
        ir_temp_path = None
        try:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            suffix = Path(visible_file.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as visible_temp:
                visible_temp.write(visible_file.read())
                visible_temp_path = Path(visible_temp.name)

            if ir_file:
                ir_suffix = Path(ir_file.name).suffix or suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=ir_suffix, dir=UPLOAD_DIR) as ir_temp:
                    ir_temp.write(ir_file.read())
                    ir_temp_path = Path(ir_temp.name)
                st.info("IR video loaded - multimodal mode")
            else:
                st.warning("IR video not provided - visible-only mode")

            run_analysis(
                lambda: create_frame_source(
                    str(visible_temp_path), str(ir_temp_path) if ir_temp_path else None
                ),
                show_ir_frames=bool(show_ir and ir_temp_path),
            )
        except Exception as exc:
            # Full traceback goes to the server log only.
            logger.exception("Video analysis failed")
            st.error(f"Error: {exc}")
        finally:
            safe_delete(visible_temp_path)
            safe_delete(ir_temp_path)
    else:
        st.info("Please upload a visible video to begin analysis")

st.markdown("---")
st.caption("VigiLens | Multimodal Anomaly Detection | NIT Rourkela")
