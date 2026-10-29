"""
Anomaly calibration.

Calibration collects prediction errors on held-out **normal** videos and sets
the anomaly threshold to ``mean + k * std`` (with a percentile alternative).
The threshold is saved as JSON next to the model checkpoint and is loaded by
the Streamlit app and by offline evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from tqdm import tqdm

from training.scoring import prediction_errors

logger = logging.getLogger(__name__)


class AnomalyCalibrator:
    """Calibrate an anomaly threshold from normal prediction errors."""

    def __init__(self,
                 model,
                 k_std: float = 3.0,
                 percentile: float = 95.0):
        self.model = model
        self.k_std = float(k_std)
        self.percentile = float(percentile)

    def collect_errors_from_caches(self,
                                   cache_paths: Sequence[Union[str, Path]],
                                   device: Optional[str] = None,
                                   batch_size: int = 8,
                                   sequence_length: int = 16,
                                   limit: Optional[int] = None) -> List[float]:
        """Collect per-frame prediction errors from cached feature files."""
        cache_paths = [Path(path) for path in cache_paths]
        if limit is not None:
            cache_paths = cache_paths[:limit]
        if not cache_paths:
            raise ValueError("No cache files were provided for calibration")

        self.model.eval()
        errors = prediction_errors(
            self.model,
            tqdm(cache_paths, desc="Calibrating"),
            smoothing_window=1,
            device=device,
        )
        return errors.tolist()

    def collect_errors(self,
                       val_videos: Sequence[Union[str, Path]],
                       val_ir_videos: Optional[Sequence[Union[str, Path]]] = None,
                       device: Optional[str] = None,
                       batch_size: int = 8) -> List[float]:
        """
        Backward-compatible entry point.

        Calibration must use the same features as training.  Pass the ``.pt``
        files produced by ``python -m training.precompute``; passing raw video
        files raises an informative error instead of silently using a
        different feature path.
        """
        paths = [Path(path) for path in val_videos]
        if paths and all(path.suffix.lower() == ".pt" for path in paths):
            return self.collect_errors_from_caches(paths, device=device, batch_size=batch_size)
        raise ValueError(
            "Calibration expects precomputed .pt feature caches. Run "
            "`python -m training.precompute` first, then pass the cache files "
            "or use `--cache-dir`."
        )

    def compute_threshold(self, errors: Sequence[float]) -> Dict:
        """Compute calibration statistics from normal prediction errors."""
        values = np.asarray(errors, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Cannot calibrate: no finite prediction errors were collected")

        mean_error = float(np.mean(values))
        std_error = float(np.std(values))
        median_error = float(np.median(values))
        p95 = float(np.percentile(values, self.percentile))
        threshold_std = mean_error + self.k_std * std_error
        threshold_percentile = float(np.percentile(values, self.percentile))

        calibration_info = {
            "mean_error": mean_error,
            "std_error": std_error,
            "median_error": median_error,
            "p95_error": p95,
            "threshold": threshold_std,
            "threshold_std": threshold_std,
            "threshold_percentile": threshold_percentile,
            "k_std": self.k_std,
            "percentile": self.percentile,
            "num_samples": int(values.size),
            "method": "std",
        }

        logger.info(
            "Calibration: mean=%.6f std=%.6f threshold_std=%.6f threshold_p%.1f=%.6f",
            mean_error, std_error, threshold_std, self.percentile, threshold_percentile,
        )
        return calibration_info

    @staticmethod
    def save_calibration(calibration_info: Dict, save_path: Union[str, Path]) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(calibration_info, handle, indent=2)
        logger.info("Calibration saved to %s", path)

    @staticmethod
    def load_calibration(load_path: Union[str, Path]) -> Dict:
        with open(load_path, "r", encoding="utf-8") as handle:
            calibration_info = json.load(handle)
        logger.info("Calibration loaded from %s", load_path)
        return calibration_info


def calibrate_model(model,
                    val_videos: Sequence[Union[str, Path]],
                    val_ir_videos: Optional[Sequence[Union[str, Path]]] = None,
                    device: Optional[str] = None,
                    checkpoint_dir: str = "checkpoints",
                    k_std: float = 3.0,
                    percentile: float = 95.0) -> Dict:
    """Calibrate a model from cache files and save ``calibration.json``."""
    calibrator = AnomalyCalibrator(model, k_std=k_std, percentile=percentile)
    errors = calibrator.collect_errors(val_videos, val_ir_videos, device=device)
    calibration_info = calibrator.compute_threshold(errors)
    calibrator.save_calibration(
        calibration_info, Path(checkpoint_dir) / "calibration.json"
    )
    return calibration_info


def discover_caches(cache_dir: Union[str, Path]) -> List[Path]:
    directory = Path(cache_dir)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*.pt") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the VigiLens anomaly threshold")
    parser.add_argument("--cache-dir", required=True, help="Normal validation feature caches")
    parser.add_argument("--checkpoint", required=True, help="Trained model checkpoint")
    parser.add_argument("--output", default="checkpoints/calibration.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k-std", type=float, default=3.0)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--limit", type=int, default=None, help="Use at most N cache files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from models.anomaly_model import VigiLensModel

    caches = discover_caches(args.cache_dir)
    if not caches:
        raise SystemExit(f"No .pt caches found in {args.cache_dir}")

    model = VigiLensModel(device=args.device, load_extractors=False)
    model.load_checkpoint(args.checkpoint)
    model.to(args.device)

    calibrator = AnomalyCalibrator(model, k_std=args.k_std, percentile=args.percentile)
    errors = calibrator.collect_errors_from_caches(caches, device=args.device, limit=args.limit)
    info = calibrator.compute_threshold(errors)
    calibrator.save_calibration(info, args.output)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
