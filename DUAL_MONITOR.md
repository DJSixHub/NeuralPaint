# Dual-Monitor Setup for NeuralPaint

## Overview

NeuralPaint now supports dual-monitor configurations, allowing you to:
- **Main screen**: Show full UI (camera preview, instructions, status messages, cursor, canvas)
- **Projector/secondary screen**: Show ONLY the canvas and drawings (no UI elements)

This is perfect for presentations and art installations where you want the audience to see only the drawing surface while you monitor the camera feed and controls.

## Requirements

- Windows OS with `pywin32` installed
- Two monitors connected (extended desktop mode recommended)
- Calibration completed for the projection surface

## Usage

### Single-Monitor Mode (Default)

The default behavior is unchanged. Just run NeuralPaint normally:

```bash
python src/neural_paint.py --camera 0
```

This shows everything (canvas + UI) on your primary monitor.

### Dual-Monitor Mode

Enable dual-monitor mode with the `--dual-monitor` flag:

```bash
python src/neural_paint.py --camera 0 --dual-monitor
```

By default:
- Monitor 0 (primary): Shows full UI
- Monitor 1 (secondary): Shows canvas only

### Custom Monitor Selection

If your monitors are in a different order, specify them explicitly:

```bash
python src/neural_paint.py --camera 0 --dual-monitor --main-monitor 0 --projector-monitor 1
```

To find which monitor is which, the application prints monitor information at startup:

```
Modo dual-monitor:
  Monitor principal (0): 1920x1080 @ (0, 0)
  Monitor proyector (1): 1920x1080 @ (1920, 0)
```

### Windows Display Settings

1. Press `Win + P` to open projection settings
2. Select **"Extend"** mode (NOT Duplicate or Second screen only)
3. Open Display Settings (`Win + I` → System → Display)
4. Arrange monitors in the correct physical order
5. Note the monitor numbers (1, 2, etc.)
6. Use `--main-monitor` and `--projector-monitor` to match your setup

## Examples

### Example 1: Standard Setup (Left main, right projector)
```bash
python src/neural_paint.py --camera 0 --dual-monitor
```

### Example 2: Reversed Setup (Left projector, right main)
```bash
python src/neural_paint.py --camera 0 --dual-monitor --main-monitor 1 --projector-monitor 0
```

### Example 3: Triple Monitor (main on center, projector on right)
```bash
python src/neural_paint.py --camera 0 --dual-monitor --main-monitor 1 --projector-monitor 2
```

### Example 4: Calibration with dual monitors
```bash
# First, calibrate (uses single-monitor mode for calibration UI)
python src/neural_paint.py --camera 0 --calibration-only --calibration-mode apriltag

# Then run with dual monitors
python src/neural_paint.py --camera 0 --dual-monitor
```

## Projector Content

In dual-monitor mode, the projector shows:
- ✅ Canvas with all drawings
- ✅ Applied segmentation masks and OCR results
- ✅ Semi-transparent overlay (alpha=150 for subtle visibility)
- ❌ No camera preview
- ❌ No status text or instructions
- ❌ No gesture cursor/pointer
- ❌ No selection rectangles

The main screen shows everything for your monitoring and control.

## Troubleshooting

### "Monitor principal X no existe"
Your specified monitor index is invalid. Check how many monitors are detected at startup.

### Both monitors show the same content
You're likely in "Duplicate" mode. Switch to "Extend" mode in Windows display settings.

### Projector shows nothing
1. Ensure the projector is detected by Windows (Settings → System → Display)
2. Check the monitor index with `--projector-monitor`
3. Try running without `--dual-monitor` first to verify single-monitor mode works

### Overlay appears offset on projector
If monitors have different resolutions or DPI scaling:
1. Set both monitors to the same DPI scaling (100%)
2. Ensure Windows display arrangement matches physical setup
3. The application calls `SetProcessDPIAware()` to prevent scaling issues

## Technical Details

- Uses Windows `EnumDisplayMonitors` API to detect all screens
- Creates separate `OverlayWindow` instances per monitor
- Renders canvas-only overlay for projector (no UI compositing)
- Falls back gracefully to single-monitor if detection fails
- Fully backward compatible - existing workflows unchanged

## Calibration Notes

Calibration always uses single-monitor mode to show the calibration UI clearly. After calibration is complete, launch with `--dual-monitor` for the interactive drawing session.

The calibration homography maps camera coordinates to canvas coordinates, independent of monitor layout. The canvas is then scaled to fit each monitor's resolution separately.
