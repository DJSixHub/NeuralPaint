# Copilot instructions for NeuralPaint

Purpose: Give AI coding agents immediate, actionable context about this repository so they can make safe, correct edits.

- **Entry point**: `src/neural_paint.py` — CLI for calibration and interactive mode. Use this for end-to-end runs and reproducing user scenarios.
- **Main loop & orchestration**: `src/neuralpaint/app.py` — camera capture, MediaPipe processing, gesture classification, overlay, OCR calls, and stroke/canvas management.
- **Calibration**: `src/neuralpaint/calibration.py` — homography handling. The CLI flag `--calibration` and `--calibration-only` control behavior.
- **Segmentation / model wrapper**: `src/neuralpaint/segmentation.py` — `Segmenter.run_on_patch()` shells out to `Reconocimiento de Caracteres/testing.py` and expects a model checkpoint (default: `models/checkpoint_epoch_70.pth`). Output masks are saved into `masc_produced/`.
- **Recognition**: `Reconocimiento de Caracteres/testing.py` and the `Reconocimiento de Caracteres/` package contain the character-recognition model and requirements.

Quick commands (Windows examples):

- Calibrate only: `python src/neural_paint.py --camera 0 --calibration-only --calibration-warp`
- Run interactive mode: `python src/neural_paint.py --camera 0`
- Force CPU OCR: add `--easyocr-cpu` or supply `--easyocr-models` folder for EasyOCR weights.

Project-specific patterns and gotchas:

- Overlay support is Windows-specific and requires `pywin32`. If `pywin32` is not available, `app.py` exits early. See the top of `src/neuralpaint/app.py` for the runtime check.
- Camera-to-surface mapping uses a saved homography at `calibration/homography.npz`. Calibration writes this file; the homography is applied with OpenCV `warpPerspective`.
- Gesture logic: gestures are classified in `src/neuralpaint/gestures.py`; many features depend on rising-edge detection and hold-time semantics (see `SelectionGestureTracker` in `src/neuralpaint/gestures.py`). Prefer minimal, localized changes when editing gesture or selection flows.
- **Region selection flow**: Both index fingers up → enters REGION_SELECT mode. One hand makes a fist (anchor point), other hand moves to size the rectangle. When the SAME hand that made the fist raises its index → confirms immediately and sends the selected screen region to the network. Output is overlaid at the exact same screen coordinates.
- **Coordinate systems**: Camera coords (MediaPipe) → Surface coords (via homography) → Screen coords (scaled for display). Selection rectangles are in surface coords, captured/overlaid in screen coords. The overlay capture grabs the actual visible screen pixels; no rotation/flip is applied to the network input.
- Segmenter runs an external script (subprocess). Editing segmentation behavior often requires checking `Reconocimiento de Caracteres/testing.py` and ensuring the expected CLI args and produced filenames (`*_mask.png`, `*_overlay.png`) are unchanged.
- Persistent artifacts: `masc_produced/` is used to keep input/outputs for debugging; `models/` stores checkpoints. Tests or CI do not currently manage these artifacts.

Development & debugging tips for agents:

- Reproduce flows locally by running `src/neural_paint.py` with `--calibration-only` then without it. Use `--calibration-debug` to show edge maps during calibration.
- To iterate on segmentation: run `Segmenter.run_on_patch()` via a short script or simulate the screen capture branch by warping a test image with the same homography.
- When changing model invocation, update `src/neuralpaint/segmentation.py` *and* `Reconocimiento de Caracteres/testing.py` tests/args together — they are coupled by CLI contract.
- Be conservative with frame-rate-sensitive changes in `app.py`: small inefficiencies can destabilize gesture detection. Benchmark changes with a short live run.

Files to inspect for any change that touches UX/gesture/OCR pipeline:

- `src/neural_paint.py` (entry)
- `src/neuralpaint/app.py` (main loop)
- `src/neuralpaint/calibration.py` (homography)
- `src/neuralpaint/segmentation.py` (Segmenter)
- `src/neuralpaint/gestures.py` (gesture rules, SelectionGestureTracker)
- `src/neuralpaint/overlay.py` (OverlayWindow integration, screen capture)
- `Reconocimiento de Caracteres/testing.py` (model CLI contract)

If you modify core behavior, leave small, focused changes and add a short usage snippet in a repository README or this file explaining how to exercise the change (example CLI flags and expected artifact paths). Ask maintainers before renaming produced files or changing the segmentation CLI contract.

If anything above is unclear or you'd like more examples (e.g., exact lines to change for a specific task), tell me which area and I will expand with code pointers.
