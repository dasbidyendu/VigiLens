"""
Training for VigiLens normal-only anomaly detection.

Two training paths are supported:

1. **Cached features (recommended).** ``training/precompute.py`` runs YOLO and
   the frozen ResNet50 encoders once.  This script then trains the fusion
   module and the causal LSTM on the cached 2048-D features.  This is fast and
   is the path used by the Colab notebook.
2. **Online features (optional, slow).** ``--online`` decodes frames, runs YOLO
   and the encoders during training.  It can unfreeze ResNet stage 4 with
   ``--unfreeze-layer4`` for stage-2 fine-tuning.

The anomaly score is always the next-feature prediction error, so training
uses ``z_t -> z_(t+1)`` targets.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import get_config
from models.anomaly_model import VigiLensModel
from training.dataset import CachedFeatureDataset, NormalVideoDataset
from training.scoring import fuse_sequence
from training.utils import (
    DEFAULT_VIDEO_EXTENSIONS,
    discover_videos,
    extract_sequence_id,
    pair_videos_by_sequence_id,
)
from training.validate import validate_cached

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility / AMP helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _autocast(device: str, enabled: bool):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(enabled and str(device).startswith("cuda") and torch.cuda.is_available()),
    )


def _move_batch(batch: Dict, device: str) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


# ---------------------------------------------------------------------------
# Cached-feature training
# ---------------------------------------------------------------------------

def train_epoch_cached(model,
                       dataloader: DataLoader,
                       optimizer: torch.optim.Optimizer,
                       device: str,
                       amp: bool = False,
                       grad_clip: float = 1.0,
                       scaler: Optional[torch.amp.GradScaler] = None,
                       limit_batches: Optional[int] = None,
                       epoch: int = 0) -> float:
    """Train one epoch on cached feature sequences."""
    model.train()
    model.fusion.train()
    model.temporal.train()

    total_loss = 0.0
    num_batches = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    for batch_index, batch in enumerate(progress):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        batch = _move_batch(batch, device)
        rgb_sequence = batch["rgb_sequence"].float()
        ir_sequence = batch["ir_sequence"].float()
        rgb_available = batch["rgb_available"]
        ir_available = batch["ir_available"]

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            fused = fuse_sequence(
                model, rgb_sequence, ir_sequence, rgb_available, ir_available
            )
            predictions, _ = model.temporal.forward_sequence(fused[:, :-1])
            targets = fused[:, 1:]
            loss = F.smooth_l1_loss(predictions, targets)

        if not torch.isfinite(loss):
            logger.warning("Skipping non-finite loss at batch %d", batch_index)
            continue

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / num_batches if num_batches else float("nan")


def train_cached(model,
                 train_dataset: CachedFeatureDataset,
                 val_dataset: CachedFeatureDataset,
                 device: str,
                 epochs: int = 30,
                 batch_size: int = 8,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 grad_clip: float = 1.0,
                 amp: bool = True,
                 checkpoint_dir: Union[str, Path] = "checkpoints",
                 config_dict: Optional[Dict] = None,
                 resume: Optional[Union[str, Path]] = None,
                 num_workers: int = 0,
                 seed: int = 42,
                 limit_batches: Optional[int] = None,
                 val_batch_size: Optional[int] = None) -> Dict:
    """Train fusion + temporal model on cached features."""
    seed_everything(seed)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pth"
    last_path = checkpoint_dir / "last_model.pth"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size or batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and str(device).startswith("cuda")))

    start_epoch = 1
    best_val_loss = float("inf")
    history: List[Dict] = []

    if resume is not None and Path(resume).exists():
        logger.info("Resuming from %s", resume)
        try:
            checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        except Exception:
            checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, KeyError) as exc:
                logger.warning("Could not restore optimizer state: %s", exc)
        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except (ValueError, KeyError) as exc:
                logger.warning("Could not restore scheduler state: %s", exc)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("loss", float("inf")))
        extra = checkpoint.get("extra", {}) or {}
        history = list(extra.get("history", []))

    logger.info(
        "Cached training: %d train sequences, %d val sequences, %d epochs, batch=%d",
        len(train_dataset), len(val_dataset), epochs, batch_size,
    )
    logger.info("Trainable parameters: %d", sum(p.numel() for p in trainable))

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch_cached(
            model, train_loader, optimizer, device, amp=amp, grad_clip=grad_clip,
            scaler=scaler, limit_batches=limit_batches, epoch=epoch,
        )
        val_result = validate_cached(
            model,
            [Path(path) for path in val_dataset.cache_paths],
            device=device,
            sequence_length=val_dataset.sequence_length,
            batch_size=val_batch_size or batch_size,
            num_workers=num_workers,
            amp=amp,
        )
        val_loss = float(val_result["val_loss"])
        scheduler.step(val_loss)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)
        logger.info(
            "Epoch %d/%d: train=%.6f val=%.6f lr=%.2e",
            epoch, epochs, train_loss, val_loss, entry["lr"],
        )

        extra = {
            "history": history,
            "scheduler_state_dict": scheduler.state_dict(),
            "num_train_sequences": len(train_dataset),
            "num_val_sequences": len(val_dataset),
        }
        model.save_checkpoint(
            str(last_path), epoch, val_loss, config_dict or {},
            optimizer_state=optimizer.state_dict(), extra=extra,
        )
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                str(best_path), epoch, val_loss, config_dict or {},
                optimizer_state=optimizer.state_dict(), extra=extra,
            )
            logger.info("New best checkpoint (val_loss=%.6f)", val_loss)

        with open(checkpoint_dir / "training_history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    if not best_path.exists():
        # A completely non-finite validation loss should still leave a usable
        # checkpoint for inspection.
        model.save_checkpoint(
            str(best_path), epochs, best_val_loss, config_dict or {},
            optimizer_state=optimizer.state_dict(), extra={"history": history},
        )

    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_val_loss": best_val_loss,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Optional online (decoded-frame) training
# ---------------------------------------------------------------------------

def _to_bgr_uint8(frames: torch.Tensor) -> List[np.ndarray]:
    """Convert a (B, H, W, 3) float tensor in [0, 1] to BGR uint8 frames."""
    array = frames.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return [frame for frame in array]


def _to_gray_uint8(frames: torch.Tensor) -> List[np.ndarray]:
    array = frames.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return [frame for frame in array]


def _online_features(extractor,
                     frames: torch.Tensor,
                     masks: Optional[Sequence[Optional[np.ndarray]]] = None) -> torch.Tensor:
    """Differentiable feature extraction for online/stage-2 training."""
    bgr_frames = _to_bgr_uint8(frames)
    batch = extractor.preprocess_batch(bgr_frames, masks)
    return extractor.extract_tensor(batch)


def _online_ir_features(extractor,
                        frames: torch.Tensor,
                        masks: Optional[Sequence[Optional[np.ndarray]]] = None) -> torch.Tensor:
    gray_frames = _to_gray_uint8(frames)
    batch = extractor.preprocess_batch(gray_frames, masks)
    return extractor.extract_tensor(batch)


def _ir_masks_from_rgb(rgb_masks, ir_frames: torch.Tensor):
    if rgb_masks is None:
        return None
    ir_np = _to_gray_uint8(ir_frames)
    masks = []
    for mask, frame in zip(rgb_masks, ir_np):
        if mask is None:
            masks.append(None)
        elif mask.shape[:2] != frame.shape[:2]:
            masks.append(cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST))
        else:
            masks.append(mask)
    return masks


def _online_epoch(model,
                  dataloader: DataLoader,
                  segmentation,
                  optimizer: Optional[torch.optim.Optimizer],
                  device: str,
                  amp: bool,
                  grad_clip: float,
                  scaler: Optional[torch.amp.GradScaler],
                  training: bool,
                  limit_batches: Optional[int] = None,
                  epoch: int = 0) -> float:
    model.train(training)
    model.fusion.train(training)
    model.temporal.train(training)
    if hasattr(model.rgb_extractor, "train"):
        model.rgb_extractor.train(training and any(p.requires_grad for p in model.rgb_extractor.extractor.parameters()))
    if hasattr(model.ir_extractor, "train"):
        model.ir_extractor.train(training and any(p.requires_grad for p in model.ir_extractor.extractor.parameters()))
    if segmentation is not None:
        segmentation.model.eval()

    total_loss = 0.0
    num_batches = 0
    progress = tqdm(dataloader, desc=("Train" if training else "Val") + f" {epoch}", leave=False)
    context = torch.enable_grad() if training else torch.inference_mode()

    for batch_index, batch in enumerate(progress):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        batch = _move_batch(batch, device)
        rgb_sequence = batch["rgb_sequence"].float()
        ir_sequence = batch["ir_sequence"].float()
        rgb_available = batch["rgb_available"]
        ir_available = batch["ir_available"]
        batch_size, sequence_length = rgb_sequence.shape[:2]

        hidden_state = model.init_hidden(batch_size, device)
        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with context:
            with _autocast(device, amp):
                batch_loss = torch.zeros((), device=device)
                steps = 0
                for t in range(sequence_length - 1):
                    rgb_current = rgb_sequence[:, t]
                    rgb_next = rgb_sequence[:, t + 1]
                    ir_current = ir_sequence[:, t] if ir_sequence is not None else None
                    ir_next = ir_sequence[:, t + 1] if ir_sequence is not None else None

                    rgb_masks = segmentation.extract_person_masks(_to_bgr_uint8(rgb_current)) if segmentation is not None else None
                    rgb_next_masks = segmentation.extract_person_masks(_to_bgr_uint8(rgb_next)) if segmentation is not None else None

                    rgb_features = _online_features(model.rgb_extractor, rgb_current, rgb_masks)
                    rgb_next_features = _online_features(model.rgb_extractor, rgb_next, rgb_next_masks)

                    ir_features = None
                    ir_next_features = None
                    if ir_current is not None and bool(ir_available.any()):
                        ir_masks = _ir_masks_from_rgb(rgb_masks, ir_current)
                        ir_features = _online_ir_features(model.ir_extractor, ir_current, ir_masks)
                    if ir_next is not None and bool(ir_available.any()):
                        ir_next_masks = _ir_masks_from_rgb(rgb_next_masks, ir_next)
                        ir_next_features = _online_ir_features(model.ir_extractor, ir_next, ir_next_masks)

                    fused = model.fusion(rgb_features, ir_features, rgb_available, ir_available)
                    fused_next = model.fusion(rgb_next_features, ir_next_features, rgb_available, ir_available)
                    loss, hidden_state = model.training_step(fused, fused_next, hidden_state)
                    batch_loss = batch_loss + loss
                    steps += 1

                if steps:
                    batch_loss = batch_loss / steps

        if training and optimizer is not None and steps:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += float(batch_loss.item())
        num_batches += 1
        progress.set_postfix(loss=f"{batch_loss.item():.4f}")

    return total_loss / num_batches if num_batches else float("nan")


def train_online(model,
                 train_dataset: NormalVideoDataset,
                 val_dataset: NormalVideoDataset,
                 device: str,
                 segmentation=None,
                 epochs: int = 30,
                 batch_size: int = 2,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 grad_clip: float = 1.0,
                 amp: bool = True,
                 checkpoint_dir: Union[str, Path] = "checkpoints",
                 config_dict: Optional[Dict] = None,
                 num_workers: int = 0,
                 seed: int = 42,
                 unfreeze_layer4: bool = False,
                 limit_batches: Optional[int] = None) -> Dict:
    """Optional online training path (slow; supports ResNet stage-4 tuning)."""
    seed_everything(seed)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pth"

    if unfreeze_layer4:
        model.rgb_extractor.extractor.set_layer4_trainable(True)
        model.ir_extractor.extractor.set_layer4_trainable(True)
        logger.info("Unfroze ResNet layer4 for both encoders")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if unfreeze_layer4:
        trainable += model.rgb_extractor.extractor.trainable_parameters()
        trainable += model.ir_extractor.extractor.trainable_parameters()
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and str(device).startswith("cuda")))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _online_epoch(
            model, train_loader, segmentation, optimizer, device, amp, grad_clip,
            scaler, training=True, limit_batches=limit_batches, epoch=epoch,
        )
        val_loss = _online_epoch(
            model, val_loader, segmentation, None, device, amp, grad_clip,
            scaler, training=False, limit_batches=limit_batches, epoch=epoch,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("Epoch %d/%d: train=%.6f val=%.6f", epoch, epochs, train_loss, val_loss)
        model.save_checkpoint(
            str(best_path if val_loss < best_val else checkpoint_dir / "last_model.pth"),
            epoch, val_loss, config_dict or {},
            optimizer_state=optimizer.state_dict(), extra={"history": history},
            include_encoders=unfreeze_layer4,
        )
        if val_loss < best_val:
            best_val = val_loss
            model.save_checkpoint(
                str(best_path), epoch, val_loss, config_dict or {},
                optimizer_state=optimizer.state_dict(), extra={"history": history},
                include_encoders=unfreeze_layer4,
            )
    return {"best_checkpoint": str(best_path), "best_val_loss": best_val, "history": history}


# ---------------------------------------------------------------------------
# Dataset construction / compatibility helpers
# ---------------------------------------------------------------------------

def discover_caches(cache_dir: Union[str, Path]) -> List[Path]:
    directory = Path(cache_dir)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*.pt") if path.is_file())


def build_cached_datasets(cache_dir: Union[str, Path],
                          train_cache: Optional[Union[str, Path]] = None,
                          val_cache: Optional[Union[str, Path]] = None,
                          sequence_length: int = 16,
                          sequence_stride: int = 1,
                          rgb_dropout_prob: float = 0.15,
                          ir_dropout_prob: float = 0.15,
                          max_cached_videos: int = 4) -> Tuple[CachedFeatureDataset, CachedFeatureDataset]:
    cache_dir = Path(cache_dir)
    train_dir = Path(train_cache) if train_cache else (cache_dir / "train" if (cache_dir / "train").exists() else cache_dir)
    val_dir = Path(val_cache) if val_cache else (cache_dir / "val" if (cache_dir / "val").exists() else cache_dir)

    train_paths = discover_caches(train_dir)
    if not train_paths:
        raise FileNotFoundError(
            f"No .pt feature caches found in {train_dir}. Run "
            "`python -m training.precompute --data-dir Data --output cache` first."
        )
    val_paths = discover_caches(val_dir)
    if not val_paths:
        logger.warning("No validation caches found; using training caches for validation")
        val_paths = train_paths

    train_dataset = CachedFeatureDataset(
        train_paths, sequence_length=sequence_length, sequence_stride=sequence_stride,
        rgb_dropout_prob=rgb_dropout_prob, ir_dropout_prob=ir_dropout_prob,
        max_cached_videos=max_cached_videos,
    )
    val_dataset = CachedFeatureDataset(
        val_paths, sequence_length=sequence_length, sequence_stride=1,
        rgb_dropout_prob=0.0, ir_dropout_prob=0.0,
        max_cached_videos=max_cached_videos,
    )
    return train_dataset, val_dataset


def validate_dataset(train_rgb: List[Path],
                     train_ir: List[Path],
                     val_rgb: List[Path],
                     val_ir: List[Path]) -> None:
    """Check readability and pairing of RGB/IR videos."""
    train_pairs = pair_videos_by_sequence_id(train_rgb, train_ir)
    val_pairs = pair_videos_by_sequence_id(val_rgb, val_ir)

    for name, videos in (("train RGB", train_rgb), ("train IR", train_ir),
                         ("val RGB", val_rgb), ("val IR", val_ir)):
        for video in videos:
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f"Cannot open {name} video: {video}")
            capture.release()

    for split_name, pairs in (("train", train_pairs), ("validation", val_pairs)):
        for rgb_path, ir_path in pairs:
            rgb_cap = cv2.VideoCapture(str(rgb_path))
            ir_cap = cv2.VideoCapture(str(ir_path))
            rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
            ir_fps = ir_cap.get(cv2.CAP_PROP_FPS)
            rgb_frames = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ir_frames = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            rgb_cap.release()
            ir_cap.release()
            if abs(rgb_fps - ir_fps) > 0.1:
                logger.warning("%s: FPS mismatch %s vs %s", split_name, rgb_path.name, ir_path.name)
            if rgb_frames != ir_frames:
                logger.warning("%s: frame count mismatch %s vs %s", split_name, rgb_path.name, ir_path.name)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def train(config,
          train_videos: List[Path],
          val_videos: List[Path],
          train_ir_videos: Optional[List[Path]] = None,
          val_ir_videos: Optional[List[Path]] = None,
          num_epochs: int = 30,
          batch_size: int = 2,
          learning_rate: float = 1e-4,
          checkpoint_dir: str = "checkpoints") -> Dict:
    """Backward-compatible online-video training entry point."""
    device = config.model.device
    segmentation = None
    if train_ir_videos is not None:
        from models.segmentation import YOLOSegmentation

        segmentation = YOLOSegmentation(device=device)

    model = VigiLensModel(device=device)
    train_dataset = NormalVideoDataset(
        train_videos, train_ir_videos,
        sequence_length=config.training.sequence_length,
        sequence_stride=config.training.sequence_stride,
        rgb_dropout_prob=config.training.rgb_dropout_prob,
        ir_dropout_prob=config.training.ir_dropout_prob,
    )
    val_dataset = NormalVideoDataset(
        val_videos, val_ir_videos,
        sequence_length=config.training.sequence_length,
        rgb_dropout_prob=0.0, ir_dropout_prob=0.0,
    )
    return train_online(
        model, train_dataset, val_dataset, device, segmentation=segmentation,
        epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
        amp=config.training.amp, checkpoint_dir=checkpoint_dir,
        config_dict=config.to_dict(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train VigiLens fusion + LSTM")
    parser.add_argument("--cache-dir", default="cache", help="Root containing train/ and val/ caches")
    parser.add_argument("--train-cache", default=None)
    parser.add_argument("--val-cache", default=None)
    parser.add_argument("--data-dir", default="Data", help="Video dataset root for --online")
    parser.add_argument("--online", action="store_true", help="Decode frames during training (slow)")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seq-stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--unfreeze-layer4", action="store_true")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate after training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = get_config()
    config.model.device = args.device
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.sequence_length = args.seq_len
    config.training.sequence_stride = args.seq_stride
    config.training.amp = args.amp
    config.training.num_workers = args.workers

    # Cached-feature training only needs fusion + temporal weights; avoid
    # downloading/instantiating two ResNet50 encoders in that path.
    model = VigiLensModel(device=args.device, load_extractors=args.online)
    model.to(args.device)

    if args.online:
        data_dir = Path(args.data_dir)
        train_rgb = discover_videos(data_dir, "train", "rgb", required=True)
        train_ir = discover_videos(data_dir, "train", "ir", required=False)
        val_rgb = discover_videos(data_dir, "val", "rgb", required=False)
        val_ir = discover_videos(data_dir, "val", "ir", required=False)
        if train_ir:
            pairs = pair_videos_by_sequence_id(train_rgb, train_ir)
            train_rgb = [pair[0] for pair in pairs]
            train_ir = [pair[1] for pair in pairs]
        else:
            train_ir = None
        if val_rgb:
            if val_ir:
                pairs = pair_videos_by_sequence_id(val_rgb, val_ir)
                val_rgb = [pair[0] for pair in pairs]
                val_ir = [pair[1] for pair in pairs]
            else:
                val_ir = None
        else:
            val_rgb = train_rgb
            val_ir = train_ir
            logger.warning("No validation videos found; using training videos for validation")

        from models.segmentation import YOLOSegmentation

        segmentation = YOLOSegmentation(device=args.device)
        train_dataset = NormalVideoDataset(
            train_rgb, train_ir,
            sequence_length=args.seq_len,
            sequence_stride=args.seq_stride,
            rgb_dropout_prob=config.training.rgb_dropout_prob,
            ir_dropout_prob=config.training.ir_dropout_prob,
        )
        val_dataset = NormalVideoDataset(
            val_rgb, val_ir,
            sequence_length=args.seq_len,
            rgb_dropout_prob=0.0, ir_dropout_prob=0.0,
        )
        result = train_online(
            model, train_dataset, val_dataset, args.device,
            segmentation=segmentation, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip, amp=args.amp,
            checkpoint_dir=args.checkpoint_dir, config_dict=config.to_dict(),
            num_workers=args.workers, seed=args.seed,
            unfreeze_layer4=args.unfreeze_layer4,
            limit_batches=args.limit_batches,
        )
    else:
        train_dataset, val_dataset = build_cached_datasets(
            args.cache_dir, args.train_cache, args.val_cache,
            sequence_length=args.seq_len, sequence_stride=args.seq_stride,
            rgb_dropout_prob=config.training.rgb_dropout_prob,
            ir_dropout_prob=config.training.ir_dropout_prob,
        )
        result = train_cached(
            model, train_dataset, val_dataset, args.device,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip, amp=args.amp,
            checkpoint_dir=args.checkpoint_dir, config_dict=config.to_dict(),
            resume=args.resume, num_workers=args.workers, seed=args.seed,
            limit_batches=args.limit_batches, val_batch_size=args.val_batch_size,
        )

    print(json.dumps(result, indent=2))

    if args.calibrate and not args.online:
        from training.calibration import AnomalyCalibrator

        caches = discover_caches(args.val_cache or (Path(args.cache_dir) / "val"))
        if caches:
            calibrator = AnomalyCalibrator(model)
            errors = calibrator.collect_errors_from_caches(caches, device=args.device)
            info = calibrator.compute_threshold(errors)
            calibrator.save_calibration(info, Path(args.checkpoint_dir) / "calibration.json")
            print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
