"""Tests for manifest-based evaluation and result export."""

import csv
import json
import math
from pathlib import Path

from training.evaluate import evaluate_manifest


def _write_manifest(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"name": "test", "videos": entries}, handle, indent=2)
    return path


def test_evaluate_manifest_writes_scores_and_results(tiny_model, make_cache, tmp_path):
    normal = make_cache(tmp_path / "cache" / "normal.pt", num_frames=8, has_ir=True, labels=[0] * 8)
    anomaly = make_cache(
        tmp_path / "cache" / "anomaly.pt", num_frames=8, has_ir=True,
        labels=[0] * 8, seed=1,
    )
    manifest = _write_manifest(tmp_path / "manifest.json", [
        {"sequence_id": "normal", "rgb": "normal_rgb.mp4", "cache": str(normal)},
        {
            "sequence_id": "anomaly",
            "rgb": "anomaly_rgb.mp4",
            "cache": str(anomaly),
            "segments": [[4, 8]],
        },
    ])
    output_dir = tmp_path / "results"

    result = evaluate_manifest(
        tiny_model,
        manifest,
        output_dir=output_dir,
        threshold=0.5,
        smoothing_window=1,
        device="cpu",
    )

    # Frame 0 of each video has no prediction and is excluded from metrics.
    assert result["num_videos"] == 2
    assert result["num_frames"] == 14
    assert result["threshold"] == 0.5
    assert result["metrics"]["num_normal"] == 10
    assert result["metrics"]["num_anomaly"] == 4
    assert result["metrics"]["num_frames"] == 14
    assert result["score_type"] == "raw"
    assert len(result["per_video"]) == 2
    assert result["per_video"][1]["label"] == 1
    assert "metrics_raw" in result and "metrics_smoothed" in result
    assert "metrics_at_best_f1_threshold" in result

    scores_path = output_dir / "scores.csv"
    results_path = output_dir / "results.json"
    assert scores_path.exists()
    assert results_path.exists()

    with open(scores_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 16
    assert set(rows[0]) == {"sequence_id", "frame_index", "video_frame", "scored",
                            "raw_score", "smoothed_score", "label"}
    assert sum(int(row["label"]) for row in rows) == 4
    assert sum(int(row["scored"]) for row in rows) == 14

    with open(results_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["num_frames"] == 14


def test_evaluate_manifest_uses_calibration_threshold(tiny_model, make_cache, tmp_path):
    cache = make_cache(tmp_path / "cache" / "normal.pt", num_frames=6, has_ir=False, labels=[0] * 6)
    manifest = _write_manifest(tmp_path / "manifest.json", [
        {"sequence_id": "normal", "rgb": "normal_rgb.mp4", "cache": str(cache)},
    ])
    calibration = tmp_path / "calibration.json"
    with open(calibration, "w", encoding="utf-8") as handle:
        json.dump({"threshold": 0.123, "threshold_std": 0.999}, handle)

    result = evaluate_manifest(
        tiny_model,
        manifest,
        output_dir=tmp_path / "results",
        calibration_path=calibration,
        device="cpu",
    )

    assert result["threshold"] == 0.123
    assert result["calibration"]["threshold"] == 0.123
    assert math.isnan(result["metrics"]["auc"])
    assert math.isfinite(result["metrics"]["fpr"])


def test_evaluate_manifest_reports_missing_cache(tiny_model, tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.json", [
        {"sequence_id": "missing", "rgb": "missing_rgb.mp4", "cache": "does_not_exist.pt"},
    ])
    try:
        evaluate_manifest(tiny_model, manifest, output_dir=tmp_path / "results", device="cpu")
    except FileNotFoundError as exc:
        assert "Could not find a feature cache" in str(exc)
    else:
        raise AssertionError("Missing cache should raise FileNotFoundError")


def test_evaluate_manifest_frame_labels_file(tiny_model, make_cache, tmp_path):
    cache = make_cache(tmp_path / "cache" / "labeled.pt", num_frames=5, has_ir=False)
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("0\n0\n1\n1\n0\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path / "manifest.json", [
        {
            "sequence_id": "labeled",
            "rgb": "labeled_rgb.mp4",
            "cache": str(cache),
            "frame_labels": str(labels_path),
        },
    ])

    result = evaluate_manifest(
        tiny_model, manifest, output_dir=tmp_path / "results", device="cpu"
    )
    assert result["metrics"]["num_anomaly"] == 2
    assert result["metrics"]["num_normal"] == 2  # frame 0 is unscored


def test_segments_follow_frame_stride(tmp_path):
    from training.evaluate import labels_for_entry

    # Cached frames are video frames 0, 2, 4, ..., 14 (frame_stride=2).
    labels = labels_for_entry(
        {"segments": [[4, 9]]}, 8, tmp_path, cache_labels=None,
        frame_indices=list(range(0, 16, 2)),
    )
    assert labels.tolist() == [0, 0, 1, 1, 1, 0, 0, 0]


def test_anomaly_label_without_segments_marks_every_frame(tmp_path):
    from training.evaluate import labels_for_entry

    # make_eval_manifest --anomaly-all used to write "segments": [], which
    # labelled whole anomaly videos as normal.
    for entry in ({"label": "anomaly"}, {"label": "anomaly", "segments": []}):
        assert labels_for_entry(entry, 4, tmp_path).tolist() == [1, 1, 1, 1]
