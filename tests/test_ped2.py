"""Tests for the UCSD Ped2 converter on a tiny fake dataset."""

import json
import sys

import cv2
import numpy as np


def _frames(directory, count, value=60):
    directory.mkdir(parents=True)
    for index in range(1, count + 1):
        cv2.imwrite(str(directory / f"{index:03d}.tif"), np.full((24, 32), value + index, np.uint8))


def test_prepare_ped2_from_m_file_and_masks(tmp_path, monkeypatch):
    from scripts import prepare_ucsd_ped2 as ped2

    root = tmp_path / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2"
    for i in range(1, 4):
        _frames(root / "Train" / f"Train{i:03d}", 5)
    _frames(root / "Test" / "Test001", 6)
    _frames(root / "Test" / "Test002", 4)
    # Test001 labelled by the .m file (frames 3..5, 1-based inclusive).
    (root / "UCSDped2.m").write_text(
        "TestVideoFile{end+1}.gt_frame = [3:5];\n", encoding="utf-8")
    # Test002 has no .m entry -> falls back to masks (frame 2 anomalous).
    gt = root / "Test" / "Test002_gt"
    gt.mkdir()
    for index in range(1, 5):
        mask = np.zeros((24, 32), np.uint8)
        if index == 2:
            mask[5:10, 5:10] = 255
        cv2.imwrite(str(gt / f"{index:03d}.bmp"), mask)

    out, eval_root = tmp_path / "Data_ped2", tmp_path / "Eval"
    monkeypatch.setattr(sys, "argv", ["prep", "--input", str(tmp_path), "--output", str(out),
                                      "--eval-root", str(eval_root), "--val-videos", "1"])
    ped2.main()

    assert len(list((out / "train" / "rgb").glob("*.mp4"))) == 2
    assert len(list((out / "val" / "rgb").glob("*.mp4"))) == 1
    manifest = json.loads((eval_root / "manifest.json").read_text(encoding="utf-8"))
    labels = {entry["sequence_id"]: open(entry["frame_labels"]).read().split()
              for entry in manifest["videos"]}
    assert labels["ped2_test001"] == ["0", "0", "1", "1", "1", "0"]
    assert labels["ped2_test002"] == ["0", "1", "0", "0"]
