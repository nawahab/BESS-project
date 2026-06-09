# IMU-Based Gravity Alignment for Gait Video

Per-frame camera roll correction using concurrent smartphone IMU data. Developed as part of a research project on automated Edinburgh Visual Gait Score (EVGS) scoring from handheld smartphone video.

---

## Background & Motivation

When clinicians or caregivers record gait videos on handheld smartphones, camera roll (tilt relative to gravity) is inevitably introduced. This is a problem for automated scoring systems such as the EVGS pipeline, because several gait parameters are implicitly measured relative to the image vertical axis (e.g., trunk position, pelvic obliquity, hindfoot alignment). Even a few degrees of unintended roll introduces systematic bias into those angle calculations.

This pipeline reads the IMU data that most modern smartphones record simultaneously with video, estimates per-frame camera roll using a complementary filter, and rotates each frame back to gravity-aligned upright before any pose estimation occurs. On a clinical dataset of children with cerebral palsy, this correction improved automated EVGS accuracy from **48.3% to 62.4%**.

---

## Repository Contents

```
├── roll_correction.py          # Minimal version - corrected video output only
├── roll_correction_verbose.py  # Full version - adds plots, per-frame logs, timing, IMU sign controls
└── README.md
```

### Choosing a version

| Feature | `roll_correction.py` | `roll_correction_verbose.py` |
|---|---|---|
| Corrected video output | YES | YES |
| IMU + roll plots saved | NO | YES |
| Per-frame angle log file | NO | YES |
| Frame timing measurements | NO | YES |
| IMU axis sign controls | NO | YES |
| IMU sampling rate report | NO | YES |

Use `roll_correction.py` if you just want the corrected video. Use `roll_correction_verbose.py` if you are debugging, validating IMU axis orientation, or profiling performance.

---

## Dependencies

```bash
pip install numpy opencv-python matplotlib
```

Python 3.8+ is recommended. No other packages are required.

---

## IMU Data Format

The IMU file must be a JSON object where each key is a **Unix timestamp in milliseconds** (as a string or integer), and the value is a dictionary of sensor readings at that timestamp. Example:

```json
{
  "1700000000000": {
    "accelerometerX": 0.12,
    "accelerometerY": 9.78,
    "accelerometerZ": 0.45,
    "gyroX": 0.001,
    "gyroY": -0.003,
    "gyroZ": 0.021
  },
  "1700000000020": { ... }
}
```

The file is loaded with a robust decoder that handles UTF-8 BOM and extracts valid JSON even if the file has leading/trailing garbage bytes.

---

## How It Works

### 1. IMU Parsing & Time Alignment

IMU samples are sorted by timestamp and converted to a relative time axis starting at 0. A small initial segment is skipped (controlled by `SKIP_S`, see [Configuration](#configuration)) to discard the first IMU sample, which is typically zero-valued. This prevents the complementary filter from starting with a false flat reading that could take many frames to converge away from.

### 2. Orientation Auto-Detection

The script checks the mean accelerometer readings from the first few samples:

- If `|ax|` dominates → **landscape** orientation (gravity falls mostly on the X axis)
- If `|ay|` dominates → **portrait** orientation (gravity falls mostly on the Y axis)

This determines how the output frame is sized.

### 3. Roll Estimation - Complementary Filter

For each video frame, the camera roll angle is estimated using a complementary filter that fuses:

- **Accelerometer-derived roll** - `atan2(ax, ay)` gives an absolute roll angle from the gravity vector. It is accurate over long timescales but noisy during movement.
- **Gyroscope integration** - integrating `gz` (rotation around the camera's optical axis) gives a smooth short-term estimate, but drifts over time.

The filter combines both:

```
φ_k = α · (φ_{k-1} + gz_k · Δt)  +  (1 − α) · atan2(ax_k, ay_k)
```

With `α = 0.98`, the gyroscope dominates frame-to-frame (smooth), while the accelerometer slowly corrects drift (stable long-term). The fused signal is then unwrapped to avoid ±180° discontinuities.

### 4. Frame Rotation

Each video frame is rotated by `-φ_k` around its own centre using an affine warp (`cv2.warpAffine`). The output canvas is sized to avoid cropping - portrait frames stay portrait, landscape frames are transposed to portrait. Border pixels are filled by replicating the edge rather than padding with black.

---

## Configuration

All tunable parameters are at the top of each script under the `CONFIG` section.

### `roll_correction.py`

| Variable | Default | Description |
|---|---|---|
| `IMU_PATH` | - | Path to the IMU `.txt` (JSON) file |
| `VIDEO_PATH` | - | Path to the input video |
| `OUT_DIR` | `./corrected_video` | Output directory |
| `OUT_VIDEO` | `OUT_DIR/Corrected.mp4` | Output video filename |
| `SKIP_S` | `0.1` | Seconds to skip at the start of both IMU and video. This discards the first IMU sample (always `0`) so the filter does not begin from a false flat reading. See note below. |
| `ALPHA` | `0.98` | Complementary filter weight. Higher → smoother but slower drift correction. |
| `ROT_SIGN` | `+1.0` | Set to `-1.0` if the correction rotates the wrong way. |

### `roll_correction_verbose.py` - additional options

| Variable | Default | Description |
|---|---|---|
| `PLOT_BASENAME` | `"corrected"` | Prefix for the saved plot and video filenames |
| `LOG_FILENAME` | `"logs.txt"` | Name of the per-frame log file saved in `OUT_DIR` |
| `LOG_PER_FRAME_TIMINGS` | `True` | If `True`, logs warp time per frame (verbose). Set `False` for shorter logs. |
| `AX_SIGN` | `+1.0` | Flip to `-1.0` if the X accelerometer axis is inverted on your device |
| `AY_SIGN` | `+1.0` | Flip to `-1.0` if the Y accelerometer axis is inverted |
| `AZ_SIGN` | `+1.0` | Flip to `-1.0` if the Z accelerometer axis is inverted |
| `GZ_SIGN` | `+1.0` | Flip to `-1.0` if roll direction is wrong and `ROT_SIGN` alone does not fix it |
| `IMU_X_DESC` / `IMU_Y_DESC` / `IMU_Z_DESC` | strings | Human-readable descriptions of your device's IMU axes, written to the log for traceability |

#### Note on `SKIP_S`

The first sample in a smartphone IMU recording is often `0` for all channels (a hardware/buffering artefact). If this zero sample is used as the filter's initial state, the filter needs several frames to pull away from that incorrect starting angle - during which correction is wrong. Setting `SKIP_S = 0.1` (100 ms) discards those early samples. The video is trimmed by the same amount so the IMU and video stay in sync. You can reduce this to `0.0` if your IMU device does not exhibit this artefact.

---

## Usage

### Minimal version

```bash
# Edit IMU_PATH and VIDEO_PATH at the top of the script, then:
python roll_correction.py
```

Output is saved to `./corrected_video/Corrected.mp4`.

### Full version

```bash
# Edit config section at the top of the script, then:
python roll_correction_verbose.py
```

Outputs saved to `./corrected_video/`:
- `corrected.mp4` - gravity-aligned video
- `corrected__plots.png` - 4-panel plot (accelerometer, gyroZ, raw vs fused roll)
- `logs.txt` - per-frame angles, timings, and a run summary

---

## Troubleshooting

**Correction rotates the wrong way**
→ Flip `ROT_SIGN` to `-1.0`.

**Roll angle has the right shape but wrong scale / axis**
→ In `roll_correction_verbose.py`, try flipping `AX_SIGN`, `AY_SIGN`, or `GZ_SIGN` one at a time. Check the `corrected__plots.png` roll plot to see what the filter produces.

**Output video is landscape instead of portrait (or vice versa)**
→ The auto-detection uses the first 10 accelerometer samples. If your device was not in its final recording orientation during those first samples, the detection may be wrong. Check the log for the detected orientation and verify it matches your recording.

**IMU recording is shorter than the video**
→ The video is automatically trimmed to match the IMU duration. Any frames beyond the last IMU sample are discarded.

**Filter starts from a large incorrect angle and takes time to converge**
→ Increase `SKIP_S` slightly (e.g., `0.2`) to skip more of the initialisation period.

---
