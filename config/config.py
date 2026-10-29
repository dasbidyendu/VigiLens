"""
Configuration module for VigiLens multimodal anomaly detection system.

This module contains all configurable parameters for the system including:
- Model settings (ResNet50, LSTM, fusion)
- Training parameters
- Inference parameters
- Calibration parameters
- Feature-cache and evaluation settings
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Feature extraction
    visible_encoder: str = "resnet50"
    ir_encoder: str = "resnet50"
    feature_dim: int = 2048
    input_size: tuple = (224, 224)

    # Fusion
    fusion_hidden_dim: int = 256
    fusion_output_dim: int = 512

    # LSTM
    lstm_hidden_size: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.2

    # Device (None -> auto-detect)
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Dataset
    sequence_length: int = 16
    sequence_stride: int = 1
    rgb_dropout_prob: float = 0.15
    ir_dropout_prob: float = 0.15

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 30
    train_ratio: float = 0.8
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 42
    num_workers: int = 0
    train_encoders: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 10

    # Feature cache
    cache_dir: str = "cache"

    # Video formats
    supported_video_extensions: list = field(
        default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"]
    )


@dataclass
class CalibrationConfig:
    """Configuration for anomaly calibration."""

    # Threshold method
    k_std: float = 3.0
    percentile: float = 95.0

    # Calibration file
    calibration_path: str = "checkpoints/calibration.json"


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    # Temporal smoothing
    smoothing_window: int = 10
    smoothing_method: str = "moving_average"  # "moving_average" or "exponential"
    ema_alpha: float = 0.3

    # Segmentation
    yolo_model_path: str = "yolov8n-seg.pt"
    yolo_conf: float = 0.25
    yolo_retina_masks: bool = True

    # Checkpoint
    checkpoint_path: str = "checkpoints/best_model.pth"

    # Default threshold used when calibration is unavailable
    default_threshold: float = 0.5


@dataclass
class EvaluationConfig:
    """Configuration for offline anomaly evaluation."""

    evaluation_dir: str = "Evaluation"
    manifest_path: str = "Evaluation/manifest.json"
    scores_csv: str = "Evaluation/scores.csv"
    results_json: str = "Evaluation/results.json"


@dataclass
class SystemConfig:
    """Main configuration class combining all sub-configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # System settings
    debug_mode: bool = False
    log_level: str = "INFO"

    def to_dict(self) -> dict:
        """Return a plain, checkpoint-safe dictionary representation."""
        return {
            "model": {
                "visible_encoder": self.model.visible_encoder,
                "ir_encoder": self.model.ir_encoder,
                "feature_dim": self.model.feature_dim,
                "input_size": list(self.model.input_size),
                "fusion_hidden_dim": self.model.fusion_hidden_dim,
                "fusion_output_dim": self.model.fusion_output_dim,
                "lstm_hidden_size": self.model.lstm_hidden_size,
                "lstm_layers": self.model.lstm_layers,
                "lstm_dropout": self.model.lstm_dropout,
                "device": self.model.device,
            },
            "training": {
                "sequence_length": self.training.sequence_length,
                "sequence_stride": self.training.sequence_stride,
                "rgb_dropout_prob": self.training.rgb_dropout_prob,
                "ir_dropout_prob": self.training.ir_dropout_prob,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "num_epochs": self.training.num_epochs,
                "train_ratio": self.training.train_ratio,
                "grad_clip": self.training.grad_clip,
                "amp": self.training.amp,
                "seed": self.training.seed,
                "num_workers": self.training.num_workers,
                "train_encoders": self.training.train_encoders,
                "checkpoint_dir": self.training.checkpoint_dir,
                "cache_dir": self.training.cache_dir,
            },
            "calibration": {
                "k_std": self.calibration.k_std,
                "percentile": self.calibration.percentile,
                "calibration_path": self.calibration.calibration_path,
            },
            "inference": {
                "smoothing_window": self.inference.smoothing_window,
                "smoothing_method": self.inference.smoothing_method,
                "ema_alpha": self.inference.ema_alpha,
                "yolo_model_path": self.inference.yolo_model_path,
                "yolo_conf": self.inference.yolo_conf,
                "yolo_retina_masks": self.inference.yolo_retina_masks,
                "checkpoint_path": self.inference.checkpoint_path,
                "default_threshold": self.inference.default_threshold,
            },
            "evaluation": {
                "evaluation_dir": self.evaluation.evaluation_dir,
                "manifest_path": self.evaluation.manifest_path,
                "scores_csv": self.evaluation.scores_csv,
                "results_json": self.evaluation.results_json,
            },
        }


def get_config() -> SystemConfig:
    """Create a fresh system configuration.

    Returning a new instance (instead of a shared singleton) prevents one
    script or Streamlit rerun from mutating global defaults.
    """
    return SystemConfig()
