Trained checkpoints, calibration.json and training_history.json are written here
by training/train.py and training/calibration.py. They are git-ignored: a
calibration file only makes sense next to the exact checkpoint it was computed
for (calibration.json records that checkpoint's SHA-256).
