# Verification — 8 November 2026

Tested on macOS 15.8.1 Intel, Python 3.11.6, torch 2.2.2 / torchvision 0.17.2, NumPy 1.26.4, OpenCV 4.10.0.84, Ultralytics 8.4.204.

## Installation and data

- Created a new venv and installed the committed requirements.txt: PASS.
- `python -m pip check`: no broken requirements.
- Pinned CUB parquet cache passed SHA256 validation. Re-created the subset with the new environment: 998 train / 200 val / 100 test, two invalid source boxes excluded.
- Dataset download originally used the public, no-login Hugging Face URLs. No Roboflow key is required.
- Local training images and caches are excluded from the repository.

## Model and training

- Bundled best.pt SHA256: `0d53dc154ea6e7bd489d8c59b616409302909a10a66f3b83958d7f2927bac4ee`.
- Loaded the actual model; names are `{0: bird}`.
- Existing 50-epoch fine-tuning completed; schedule and metrics are documented in training_provenance.json and training_history.csv.
- Actual `train.py` smoke test in the fresh venv: one CPU epoch on two train and two validation images, followed by final validation; optimizer/backward and best.pt/last.pt saving completed. This smoke test does not replace the 50-epoch training results and did not overwrite bundled weights.
- Independently re-evaluated bundled best.pt on all 200 validation and 100 held-out test images on CPU. Exact settings and results: val.json and test.json. No thresholds were selected using the held-out test.

## Video and regression checks

- Actual source: Matt MacGillivray's CC BY 2.0 Blue Jay video; details in THIRD_PARTY.md.
- Included clip: 240 frames, 640×360, approximately 29.97 FPS, approximately 8.008 seconds.
- Fresh-v-env MPS video run: all 240 frames processed; zero skipped. Output decoded in full and encoded as H.264.
- Output FPS time base may be represented as 29.97 instead of 30000/1001; difference is below 0.00004 FPS. Frame count and frame dimensions match.
- Visual inspection of extracted frames at 0, 100 and 200: bird label, confidence and model-predicted bbox are visible. Localization is imperfect and can exclude part of the bird; missed frames remain visible rather than being removed.
- Duplicate CPU NMS keeps distinct separated bird boxes and suppresses overlapping copies. It does not introduce detections on frames without model output.
- Copied scripts, weights and examples into a separate checkout without datasets, runs or the old project. CPU inference there completed: 240 output frames, zero skipped, matching duration/dimensions; 170 frames with detections and 172 retained detections. Actual metrics are in the included example sidecar; speed varies by device/load.
- Full-video processing handles background frames, not just selected successful frames.
- Observed CPU/MPS discrepancy in old torch coordinate scaling. The local SafeDetectionPredictor moves detections to CPU before scale_boxes; the neural forward pass stays on the requested device. Compared CPU/MPS predictions on the same real frame: coordinates agree within 0.02 pixels. No site-packages modifications were made.

## Automated tests

`python -m unittest discover -s tests -v`: **8 tests, PASS**.

Coverage: every-frame preservation and FPS; frames without detections; atomic publishing and cleanup after simulated model failure; refusing same input/output and accidental overwrite; corrupt input; overlapping versus separate bird boxes; single-class dataset conversion; cross-split duplicate removal; invalid bbox rejection; non-square CPU/MPS coordinate restoration.

## Limitations

This demonstrates reproducible execution, not production readiness. The video has no ground-truth frame labels, so detection counts are not video recall. CUB has limited scene/domain coverage and non-commercial image-use restrictions. Fresh training uses a fixed default batch and is not a bitwise replay of the original batch schedule. Audio copying, real-time guarantees and multi-object tracking are outside this minimal deliverable.
