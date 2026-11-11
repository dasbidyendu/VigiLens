# VigiLens: Multimodal Anomaly Detection with a Causal Streaming LSTM

VigiLens is a video anomaly-detection system for pedestrian scenes. It is trained **only on normal behavior** and learns how pedestrian appearance evolves over time. At inference, a frame is flagged as anomalous when the model cannot predict its feature well.

It supports paired **visible (RGB) + infrared/thermal (IR)** video, and also RGB-only or IR-only input.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Setup](#setup)
- [Quick Start: Synthetic Smoke Test](#quick-start-synthetic-smoke-test)
- [Data Preparation](#data-preparation)
- [Feature Precomputation](#feature-precomputation)
- [Training](#training)
- [Calibration](#calibration)
- [Evaluation](#evaluation)
- [Google Colab (T4)](#google-colab-t4)
- [Streamlit App](#streamlit-app)
- [Docker](#docker)
- [Tests](#tests)
- [Implementation Notes](#implementation-notes)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## What It Does

- **Normal-only training.** No anomaly labels are needed for training or calibration.
- **Pedestrian-focused features.** YOLOv8n-seg masks restrict ResNet50 features to people when a person is detected; otherwise the full frame is used.
- **Modality-aware fusion.** RGB+IR, RGB-only, and IR-only inputs all produce a fixed 512-D embedding. Availability is tracked per sample, so mixed batches are handled correctly.
- **Causal streaming inference.** A unidirectional LSTM predicts each frame from the previous `sequence_length - 1` frames (the same window it was trained on) and never looks at future frames. Offline scoring and the app produce identical scores.
- **One-frame prediction latency.** The score for frame `t+1` is the error between the feature predicted from frame `t` and the actual feature at `t+1`. Frame 0 has no score.
- **Calibrated thresholds.** The threshold is computed from prediction errors on held-out normal videos.
- **Frame-level evaluation.** AUC, average precision, precision/recall/F1, FPR, FPR@95%TPR, per-video summaries, and CSV/JSON exports.

---

## Architecture

```
RGB frame ─► YOLOv8n-seg ─► person mask ─► ResNet50 (RGB) ─► 2048-D ─┐
                 │                                                    ├─► modality-aware
IR frame ────────┴─► resized mask ─► ResNet50 (IR) ─────► 2048-D ────┘   fusion (512-D)
                                                                              │
                                                                    causal LSTM (256)
                                                                              │
                                                                  next-feature predictor
                                                                              │
                                              MSE(predicted feature, actual next feature)
                                                                              │
                                                          causal smoothing ─► threshold
                                                                              │
                                                                      Normal / Anomaly
```

| Stage | Details |
|---|---|
| Segmentation | YOLOv8n-seg, COCO person class. All person masks are merged into one ROI; masks are resized to the frame resolution. |
| Feature extraction | ImageNet-pretrained ResNet50 per modality, 2048-D. IR frames are grayscale, replicated to 3 channels, and ImageNet-normalized. |
| Fusion | Learnable per-modality projections + gated fusion to 512-D. Missing modalities are zeroed and adapted. |
| Temporal model | 2-layer unidirectional LSTM (input 512, hidden 256, dropout 0.2). State is reset per video/stream. |
| Training objective | Smooth L1 between the prediction from `t` and the fused feature at `t+1`. |
| Scoring | Mean squared error between prediction and actual next feature, optionally smoothed causally. |

---

## Repository Layout

```
VigiLens/
├── app.py                      # Streamlit UI
├── config/config.py            # SystemConfig / get_config
├── models/
│   ├── segmentation.py         # YOLOv8n-seg person masks
│   ├── feature_extractor.py    # ResNet50 RGB/IR extractors
│   ├── fusion.py               # Modality-aware fusion
│   ├── temporal_lstm.py        # Causal LSTM + next-feature predictor
│   └── anomaly_model.py        # VigiLensModel, checkpoint I/O
├── pipeline/
│   ├── frame_source.py         # Video/directory frame sources
│   ├── synchronization.py      # Non-dropping RGB/IR pairing
│   ├── preprocessing.py        # Shared BGR/RGB/IR preprocessing
│   └── inference.py            # Stateful streaming inference
├── training/
│   ├── dataset.py              # Video and cached-feature datasets
│   ├── utils.py                # Discovery, natural sort, pairing
│   ├── precompute.py           # One-time YOLO + ResNet feature cache
│   ├── scoring.py              # Shared anomaly-score definition
│   ├── metrics.py              # Frame-level binary metrics
│   ├── train.py                # Cached and online training
│   ├── validate.py             # Cached validation
│   ├── calibration.py          # Threshold calibration
│   └── evaluate.py             # Manifest-based evaluation
├── scripts/
│   ├── make_synthetic_dataset.py
│   ├── prepare_otcbvs.py
│   ├── prepare_kaist.py
│   └── make_eval_manifest.py
├── notebooks/VigiLens_Colab.ipynb
├── tests/
├── checkpoints/                # Created locally; weights are git-ignored
├── cache/                      # Created locally; feature caches are git-ignored
└── Data/                       # Created locally; videos are git-ignored
```

---

## Setup

Python 3.10+ is recommended. A CUDA GPU is optional but strongly recommended for precomputation and online training; cached fusion/LSTM training runs comfortably on CPU for small datasets.

```bash
git clone https://github.com/Gauranga025/VigiLens.git
cd VigiLens

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first YOLO run downloads `yolov8n-seg.pt`, and the first ResNet50 run downloads ImageNet weights. Both are cached by their libraries. To run fully offline, place those weights locally and pass explicit paths.

> The repository does **not** include trained weights. Without a checkpoint the app starts an untrained model with the default threshold.

---

## Quick Start: Synthetic Smoke Test

This verifies the complete pipeline without downloading a dataset. It creates short synthetic RGB/IR videos, precomputes features, trains one epoch, calibrates, and evaluates.

```bash
python scripts/make_synthetic_dataset.py --output Data --eval-dir Evaluation --overwrite

# Use --no-segmentation because synthetic rectangles are not reliable YOLO detections.
python -m training.precompute --data-dir Data --output cache --no-segmentation --batch-size 8
python -m training.precompute --manifest Evaluation/manifest.json --output cache/eval --no-segmentation --batch-size 8

python -m training.train --cache-dir cache --epochs 1 --batch-size 2 --seq-len 8 --device cpu --no-amp
python -m training.calibration --cache-dir cache/val --checkpoint checkpoints/best_model.pth \
    --output checkpoints/calibration.json --device cpu
python -m training.evaluate --manifest Evaluation/manifest.json --cache-dir cache/eval \
    --checkpoint checkpoints/best_model.pth --calibration checkpoints/calibration.json \
    --output-dir Evaluation/results --device cpu
```

Results are written to `Evaluation/results/results.json` and `Evaluation/results/scores.csv`. The synthetic evaluation set contains both normal and anomalous clips, so AUC/AP are meaningful.

On Colab, omit `--no-segmentation` to exercise the real YOLO path. The synthetic shapes may not be detected as people, which is fine; the pipeline falls back to the full frame.

---

## Data Preparation

All training videos must contain **normal behavior only**. RGB and IR files are paired by sequence ID: `scene001_rgb.mp4` pairs with `scene001_ir.mp4` (extensions may differ). Files are discovered recursively under `Data/<split>/rgb` and `Data/<split>/ir`.

### OTCBVS Dataset 10 (smoke / preview)

OTCBVS contains short visible/thermal sequences and is useful as a first real-data check. The preparation script tolerates video pairs, paired image directories, and common visible/thermal naming conventions.

```bash
python scripts/prepare_otcbvs.py --input /path/to/OTCBVS --output Data --max-frames 300
```

Useful options: `--val-ratio 0.25`, `--all-train`, `--fps 20`, `--seed 42`, `--manifest Data/pairs.json`.

### KAIST Multispectral Pedestrian (main normal corpus)

KAIST provides synchronized `visible/` and `lwir/` image sequences:

```
set00/V000/visible/*.jpg
set00/V000/lwir/*.jpg
```

```bash
python scripts/prepare_kaist.py --input /path/to/KAIST --output Data \
    --sets 00 --val-ratio 0.2 --max-frames 300
```

Use `--val-sets 01` to hold out entire sets, or `--all-train` to use every selected sequence for training. KAIST is a **pedestrian detection** dataset, not an anomaly-detection benchmark; use it for normal-behavior training and evaluate anomaly metrics on labeled clips (for example ShanghaiTech/Avenue or your own recordings).

### Custom paired videos

Place files directly:

```
Data/
├── train/
│   ├── rgb/scene001_rgb.mp4, scene002_rgb.mp4, ...
│   └── ir/scene001_ir.mp4,  scene002_ir.mp4,  ...
└── val/
    ├── rgb/scene010_rgb.mp4, ...
    └── ir/scene010_ir.mp4,  ...
```

Rules:

- Split by **video**, never by frame, to avoid leakage.
- Keep anomalous videos out of `train/` and `val/`.
- RGB and IR pairs should have the same FPS and frame count. The precompute script truncates a pair when the IR stream ends.
- If IR is unavailable, RGB-only training and inference work; pass no IR directory or omit the IR upload.

---

## Feature Precomputation

Running YOLO and two ResNet50 encoders every epoch is the main bottleneck. Precompute once and train the fusion + LSTM on cached features:

```bash
python -m training.precompute --data-dir Data --output cache --device cuda --batch-size 16
```

For an evaluation manifest:

```bash
python -m training.precompute --manifest Evaluation/manifest.json \
    --output cache/eval --device cuda --batch-size 16
```

Each sequence produces:

- `<sequence>.pt` — `rgb_features (N, 2048)`, optional `ir_features (N, 2048)`, `has_ir`, `sequence_id`, `fps`, `frame_indices`, optional `labels`, source paths.
- `<sequence>.json` — a sidecar with frame count and metadata, so the dataset can index a large cache without loading tensors.

Useful options:

| Option | Purpose |
|---|---|
| `--no-segmentation` | Full-frame features; useful offline or when YOLO misses the subjects. |
| `--max-frames N` | Limit frames per video (fast previews). |
| `--frame-stride N` | Keep every Nth frame. |
| `--save-half` | Store float16 features (smaller cache). |
| `--limit N` | Only process the first N manifest entries. |
| `--overwrite` | Replace existing caches. |
| `--no-amp` | Disable CUDA FP16 autocast. |

Caches are ignored by git.

---

## Training

### Cached features (recommended)

```bash
python -m training.train --cache-dir cache --epochs 30 --batch-size 8 \
    --seq-len 16 --device cuda
```

`cache/train/` and `cache/val/` are used automatically when present. Otherwise every `.pt` file under `cache/` is used. Explicit directories are also supported:

```bash
python -m training.train --train-cache cache/train --val-cache cache/val \
    --epochs 30 --batch-size 8 --device cuda
```

Only `model.fusion` and `model.temporal` are trained; the ResNet50 encoders are not instantiated in this path. Checkpoints contain the fusion + LSTM state.

Common options:

| Option | Default | Purpose |
|---|---|---|
| `--epochs` | 30 | Training epochs. |
| `--batch-size` | 8 | Sequences per batch. |
| `--seq-len` | 16 | Frames per training sequence. |
| `--seq-stride` | 1 | Sliding-window stride. |
| `--lr` | 1e-4 | AdamW learning rate. |
| `--weight-decay` | 1e-5 | AdamW weight decay. |
| `--grad-clip` | 1.0 | Gradient-norm clipping. |
| `--amp/--no-amp` | AMP on | CUDA FP16 mixed precision. |
| `--resume PATH` | — | Resume from a checkpoint. |
| `--limit-batches N` | — | Stop each epoch early (debugging). |
| `--calibrate` | off | Calibrate on validation caches after training. |

Outputs: `checkpoints/best_model.pth`, `checkpoints/last_model.pth`, and `checkpoints/training_history.json`.

### Online features (optional, slow)

This decodes frames and runs YOLO + ResNet during training. It is mainly for stage-2 fine-tuning:

```bash
python -m training.train --online --data-dir Data --epochs 3 --batch-size 1 \
    --seq-len 8 --device cuda --unfreeze-layer4
```

`--unfreeze-layer4` enables gradients for ResNet stage 4 and stores encoder weights in the checkpoint. Without it, only fusion + LSTM are updated and the encoders stay frozen. Online training is memory-intensive; use a small batch size.

---

## Calibration

Calibration collects next-feature prediction errors on **held-out normal** caches and sets:

```
threshold_std = mean_error + k * std_error
threshold_percentile = percentile(errors, p)
```

The default saved threshold is `mean + 3*std`; the percentile alternative is stored alongside it. Frame 0 is excluded because it has no previous prediction.

Smoothing shrinks the spread of normal scores, so one threshold cannot serve every smoothing setting. `calibration.json` therefore also stores `thresholds.moving_average["1".."30"]` and `thresholds.exponential["<alpha>"]`, each computed on the smoothed scores the app would produce, plus the SHA-256 of the checkpoint it belongs to.

```bash
python -m training.calibration --cache-dir cache/val \
    --checkpoint checkpoints/best_model.pth \
    --output checkpoints/calibration.json --device cuda --k-std 3.0 --percentile 95
```

Training can do this automatically with `--calibrate` (it calibrates `best_model.pth`, not the last epoch). The app and `training.evaluate` pick the threshold that matches their smoothing setting (`training.calibration.threshold_for`); the app ignores a calibration file that was computed for a different checkpoint.

---

## Evaluation

Evaluation is manifest-based so it can describe any dataset. Example:

```json
{
  "name": "my_eval_set",
  "videos": [
    {
      "sequence_id": "scene001",
      "rgb": "rgb/scene001_rgb.mp4",
      "ir": "ir/scene001_ir.mp4",
      "label": "anomaly",
      "segments": [[120, 260]]
    },
    {
      "sequence_id": "scene002",
      "rgb": "rgb/scene002_rgb.mp4",
      "label": "normal"
    }
  ]
}
```

Labels are resolved in this priority order:

1. `frame_labels`: a `.txt` file (one 0/1 or `normal`/`anomaly` per line) or a `.npy` array.
2. `segments`: anomalous `[start, end)` frame intervals (a missing or empty list means "use `label`").
3. `label`: `"normal"` or `"anomaly"` for the whole video.
4. Labels stored in the precomputed cache.

Frame numbers refer to the original video and are mapped through the cache's `frame_indices`, so they stay aligned with `--frame-stride`.

Build a manifest from `Evaluation/normal/` and `Evaluation/anomaly/` folders:

```bash
python scripts/make_eval_manifest.py --root Evaluation --segments Evaluation/segments.json
```

Precompute and evaluate:

```bash
python -m training.precompute --manifest Evaluation/manifest.json \
    --output cache/eval --device cuda

python -m training.evaluate --manifest Evaluation/manifest.json \
    --cache-dir cache/eval --checkpoint checkpoints/best_model.pth \
    --calibration checkpoints/calibration.json \
    --output-dir Evaluation/results --device cuda --smoothing-window 5
```

Outputs:

- `scores.csv`: `sequence_id, frame_index, video_frame, scored, raw_score, smoothed_score, label` (`scored=0` marks the unscored first frame)
- `results.json`: threshold, metrics, per-video summaries, calibration metadata

Reported metrics are frame-level: ROC-AUC, average precision, precision/recall/F1 at the calibrated (or requested) threshold, accuracy, TPR/FNR, FPR, FPR@95%TPR, confusion counts, and the best-F1 threshold. `metrics_raw` and `metrics_smoothed` are both included. A normal-only evaluation set is valid: AUC/AP are `NaN`, while FPR remains meaningful.

**Important:** the anomaly score for frame `t+1` uses the feature prediction from frame `t`. Frame 0 of each video has no score and is excluded from all metrics. `metrics_at_best_f1_threshold` tunes the threshold on the evaluation set itself, so treat it as an optimistic upper bound, not a result.

---

## Google Colab (T4)

Open `notebooks/VigiLens_Colab.ipynb` in Colab and select a GPU runtime. Set `DATASET` to `vidllvip`, `kaist` (paired RGB+IR, normal-only, no anomaly labels) or `ped2` (UCSD Ped2: visible-only, frame-level anomaly labels, gives AUC/AP). The notebook copies the dataset archive from Google Drive, prepares it, runs an optional smoke test, precomputes features, trains, calibrates the best checkpoint, evaluates, and saves the checkpoint and calibration back to Drive. A section at the end evaluates your own labeled RGB+IR clips.

---

## Streamlit App

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Upload a visible video and, optionally, an aligned IR video. The sidebar selects the checkpoint, calibration file, and smoothing method/window.

**Webcam mode:** switch *Source* to **Webcam**, pick the camera index and a time limit, and press **Start webcam** (any other click, e.g. **Stop**, ends the run). The camera is opened on the machine running Streamlit, frames are visible-only, and a score chart against the calibrated threshold updates live. The score measures how unusual the motion is compared with the training footage, so a model trained on one scene (e.g. UCSD Ped2) flags almost everything in a different scene: train on normal footage from the same camera for meaningful results.

The app loads the checkpoint and calibration once per process, runs YOLO once per frame (the mask is reused for IR and held for up to 5 frames when a detection is missed), and displays the smoothed score, status, FPS, and optional masks. Only files in `checkpoints/` can be selected. Uploaded videos are written to `.tmp/uploads/` in the repository (override with `VIGILENS_UPLOAD_DIR`). One video is analysed at a time. If no checkpoint or matching calibration is found, it warns that the model is untrained or the threshold is uncalibrated.

---

## Docker

```bash
docker build -t vigi-lens .
docker run --rm -p 8501:8501 vigi-lens
```

The image uses Python 3.10, installs OpenCV/FFmpeg system libraries, and serves Streamlit on port 8501. Mount checkpoints and data if needed:

```bash
docker run --rm -p 8501:8501 \
    -v "$PWD/checkpoints:/app/checkpoints" \
    -v "$PWD/Data:/app/Data" \
    vigi-lens
```

---

## Tests

```bash
pytest -q
```

The test suite uses tiny synthetic tensors/videos and does not download YOLO or ResNet50 weights. It covers fusion gating, causal LSTM shapes, cached datasets, scoring, calibration, metrics, manifest evaluation, frame sources, preprocessing, and the streaming inference pipeline.

---

## Implementation Notes

These are the invariants the code is built around:

- **One score definition.** Training, validation, calibration, and evaluation all use `training/scoring.py`: score at `t+1` = MSE(prediction from `t`, fused feature at `t+1`).
- **Cached features are the source of truth.** Validation and calibration never recompute features with a different preprocessing path.
- **BGR at the boundaries.** OpenCV frame sources return BGR; `models/feature_extractor.py` performs BGR→RGB exactly once. `pipeline/preprocessing.py` mirrors the same normalization.
- **One YOLO pass per frame.** The inference pipeline computes the person mask once and resizes it for IR.
- **Per-sample modality gating.** A batch may mix RGB+IR, RGB-only, and IR-only samples; fusion applies the availability mask per sample.
- **Non-dropping synchronization.** Every visible frame produces a score; IR frames are selected by nearest timestamp/index and may be reused.
- **Checkpoint compatibility.** Normal cached-feature checkpoints contain only fusion + temporal weights. Encoder weights are included only for stage-2 online fine-tuning.
- **No hardcoded epochs/batch size.** The CLI and `SystemConfig` control them.

---

## Limitations

- **KAIST is not an anomaly benchmark.** It is useful for normal pedestrian training but does not provide frame-level anomaly labels.
- **OTCBVS is small.** It is a preview/smoke dataset, not a substitute for a full evaluation.
- **YOLO can miss people.** Synthetic shapes, tiny pedestrians, severe occlusion, and thermal-only scenes may yield no mask; the pipeline then uses the full frame.
- **The IR encoder is ImageNet-pretrained.** It was not pretrained on thermal data. Stage-2 fine-tuning can help, but it is not a substitute for a thermal-pretrained backbone.
- **Frame-level metrics only.** There is no temporal-segment metric (for example, segment-level AUC) and no tracking.
- **Calibration assumes normal validation matches deployment.** Distribution shift (camera, weather, crowd density) will change the error scale; recalibrate on target-domain normal clips.
- **One-frame latency.** A frame's score is only available after the next frame is processed. The Streamlit UI displays this causally.
- **CPU is slow.** Precomputation and online training are GPU-oriented; cached fusion/LSTM training is much lighter.

---

## Roadmap

- Adapters for ShanghaiTech, Avenue, and UBnormal frame-level labels.
- Track-level features and per-person scores.
- Segment-level evaluation metrics and latency-aware labels.
- Thermal-pretrained or contrastive IR encoders.
- ONNX/TensorRT export for edge deployment.
- Better calibration under domain shift (quantile regression, conformal thresholds).

---

## Acknowledgements

VigiLens uses PyTorch, torchvision, Ultralytics YOLOv8, OpenCV, and Streamlit. Dataset preparation scripts target OTCBVS Dataset 10 and the KAIST Multispectral Pedestrian dataset; please follow their respective licenses and citation requirements.
