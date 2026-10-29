"""
Validation for VigiLens using precomputed feature caches.

Validation measures the next-feature prediction loss on held-out normal
videos and returns the per-frame prediction errors that calibration needs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import CachedFeatureDataset
from training.scoring import fuse_sequence

logger = logging.getLogger(__name__)


def _autocast(device: str, enabled: bool):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(enabled and str(device).startswith("cuda") and torch.cuda.is_available()),
    )


def validate_cached(model,
                    cache_paths: Sequence[Union[str, Path]],
                    device: str = "cpu",
                    sequence_length: int = 16,
                    batch_size: int = 8,
                    num_workers: int = 0,
                    amp: bool = False,
                    smoothing_window: int = 1,
                    limit_batches: Optional[int] = None) -> Dict:
    """
    Validate a model on cached features.

    Returns:
        Dictionary with ``val_loss``, ``errors`` (per scored frame), and
        ``num_frames`` / ``num_batches`` counts.
    """
    cache_paths = [Path(path) for path in cache_paths]
    if not cache_paths:
        raise ValueError("No validation caches were provided")

    dataset = CachedFeatureDataset(
        cache_paths,
        sequence_length=sequence_length,
        sequence_stride=1,
        rgb_dropout_prob=0.0,
        ir_dropout_prob=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )

    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_errors: List[np.ndarray] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="Validating", leave=False)):
            if limit_batches is not None and batch_index >= limit_batches:
                break

            rgb_sequence = batch["rgb_sequence"].to(device, non_blocking=True).float()
            ir_sequence = batch["ir_sequence"].to(device, non_blocking=True).float()
            rgb_available = batch["rgb_available"].to(device)
            ir_available = batch["ir_available"].to(device)

            with _autocast(device, amp):
                fused = fuse_sequence(
                    model, rgb_sequence, ir_sequence, rgb_available, ir_available
                )
                predictions, _ = model.temporal.forward_sequence(fused[:, :-1])
                targets = fused[:, 1:]
                loss = F.smooth_l1_loss(predictions, targets)

            total_loss += float(loss.item())
            num_batches += 1

            errors = F.mse_loss(predictions, targets, reduction="none").mean(dim=-1)
            all_errors.append(errors.detach().cpu().numpy().reshape(-1))

    errors = np.concatenate(all_errors) if all_errors else np.zeros((0,), dtype=np.float32)
    return {
        "val_loss": total_loss / num_batches if num_batches else float("nan"),
        "errors": errors.astype(np.float32),
        "num_frames": int(errors.size),
        "num_batches": num_batches,
        "smoothing_window": smoothing_window,
    }


def validate(model,
             cache_paths: Sequence[Union[str, Path]],
             device: str = "cpu",
             **kwargs) -> Dict:
    """Backward-compatible alias for :func:`validate_cached`."""
    return validate_cached(model, cache_paths, device=device, **kwargs)


def discover_caches(cache_dir: Union[str, Path]) -> List[Path]:
    """Find all ``.pt`` feature caches below a directory."""
    directory = Path(cache_dir)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*.pt") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VigiLens on cached features")
    parser.add_argument("--cache-dir", required=True, help="Directory with validation .pt caches")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from config.config import get_config
    from models.anomaly_model import VigiLensModel

    config = get_config()
    model = VigiLensModel(device=args.device, load_extractors=False)
    model.load_checkpoint(args.checkpoint)
    model.to(args.device)

    result = validate_cached(
        model,
        discover_caches(args.cache_dir),
        device=args.device,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
    )

    summary = {
        "val_loss": result["val_loss"],
        "num_frames": result["num_frames"],
        "num_batches": result["num_batches"],
        "mean_error": float(np.mean(result["errors"])) if result["errors"].size else None,
        "std_error": float(np.std(result["errors"])) if result["errors"].size else None,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
