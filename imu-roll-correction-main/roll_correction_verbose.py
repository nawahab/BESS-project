"""
IMU-roll video stabilizer - verbose / debug version
- Corrects every frame to gravity level using its own IMU sample
- Auto-detects portrait / landscape and outputs upright portrait video
- Saves diagnostic plots (accelerometer, gyroZ, raw vs fused roll)
- Writes a per-frame log with applied angle and optional timing data
- Exposes IMU axis sign controls for easy adaptation to different devices
"""

import os
import json
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ============================================================
# CONFIG (edit these)
# ============================================================
IMU_PATH   = r"./imu_data.txt"
VIDEO_PATH = r"./video.mp4"

OUT_DIR       = r".\corrected_video"
PLOT_BASENAME = "corrected"       # prefix for output video and plot filenames
LOG_FILENAME  = "logs.txt"        # saved inside OUT_DIR

OUT_VIDEO = os.path.join(OUT_DIR, f"{PLOT_BASENAME}.mp4")

# SKIP_S: How many seconds to discard from the start of both the IMU stream and the video.
# The very first IMU sample from most smartphone recording apps is always zero for all channels
# (a hardware/buffer initialisation artefact). If the complementary filter starts from zero, it
# takes several frames to converge to the true roll angle, during which the correction is wrong.
# Skipping 0.1 s (a handful of samples) avoids this cold-start problem entirely.
# The same duration is cut from the video so both clocks remain in sync.
SKIP_S = 0.1

# ALPHA: Blending weight of the complementary filter (see roll estimation section below).
# Higher value → trusts the gyroscope more (smoother output, slower drift correction).
# Lower value → trusts the accelerometer more (faster correction, but noisier frame-to-frame).
# 0.98 is a common starting point for video-rate fusion (~30 fps).
ALPHA  = 0.98

# ROT_SIGN: Set to -1.0 if the corrected video tilts in the wrong direction.
# Different smartphone apps may report the IMU axes with opposite sign conventions,
# which flips the computed roll angle. This variable corrects for that without
# touching the filter math. You only ever need +1.0 or -1.0.
ROT_SIGN = +1.0

# LOG_PER_FRAME_TIMINGS: When True, the log file records the cumulative warp time
# and per-loop-iteration time alongside the applied angle for every frame.
# Useful for profiling on a new machine or identifying slow frames.
# Set False for shorter logs when timing is not of interest.
LOG_PER_FRAME_TIMINGS = True

###############  IMU axis sign controls #####################################
# Different smartphone IMU recording apps may define accelerometer/gyroscope axes
# with opposite sign conventions. Instead of rearranging the filter equations,
# we apply sign multipliers to the raw data so that the rest of the code always
# sees a consistent physical convention.
#
# The DESC strings document what each axis represents on YOUR device. They are
# written to the log at the top of every run, making the log self-documenting -
# someone reading it later can immediately see which physical direction each axis
# referred to without having to look up the code.
#
# How to determine if a sign needs flipping:
#   1. Run the script and open the diagnostic plot (corrected__plots.png).
#   2. On the Accelerometer panel, tilt the phone left. If ax goes negative instead
#      of positive, set AX_SIGN = -1.0. Repeat for ay (tilt forward) and az (press
#      the back of the phone).
#   3. On the Roll panel, if the fused roll angle goes the wrong way compared to
#      your physical tilt, try flipping GZ_SIGN, then ROT_SIGN if still wrong.
IMU_X_DESC = "left side of screen"
IMU_Y_DESC = "top of the screen"
IMU_Z_DESC = "out of the screen (toward the user)"
AX_SIGN    = +1.0
AY_SIGN    = +1.0
AZ_SIGN    = +1.0
GX_SIGN    = +1.0   # gyroX is not used in roll computation, included for completeness
GY_SIGN    = +1.0   # gyroY is not used in roll computation, included for completeness
GZ_SIGN    = +1.0

IMU_FRAME_TEXT = f"""
IMU frame (device/screen frame):
 +x = {IMU_X_DESC}
 +y = {IMU_Y_DESC}
 +z = {IMU_Z_DESC}
""".strip()


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_imu_json_robust(path: str) -> dict:
    # Read as raw bytes first, then decode. "utf-8-sig" automatically strips the
    # byte-order mark (BOM) that some apps prepend to their output files.
    # errors="ignore" silently drops any bytes that are not valid UTF-8.
    raw = open(path, "rb").read()
    txt = raw.decode("utf-8-sig", errors="ignore")

    # Some recording apps write a text preamble or trailing newline/whitespace
    # outside the JSON object itself. Slicing from the first "{" to the last "}"
    # extracts only the valid JSON regardless of what surrounds it.
    start, end = txt.find("{"), txt.rfind("}")
    txt = txt[start:end+1]
    return json.loads(txt)


def parse_imu(data: dict):
    # Keys are Unix timestamps in milliseconds, stored as strings in JSON.
    # We convert to int before sorting to get numerical order, not lexicographic order.
    # (Lexicographic order would put "1000" before "999", which is wrong.)
    t_ms = np.array(sorted(int(k) for k in data.keys()), dtype=np.int64)

    # Defensive lookup: some JSON parsers keep keys as integers, others stringify them.
    # The conditional handles both cases so the code works regardless of the app/parser.
    rows = [data[str(k)] if str(k) in data else data[k] for k in t_ms]

    # .get(name, np.nan) gracefully returns NaN for any timestamp where a sensor
    # channel is missing (e.g., a partial record at the end of the file).
    def col(name):
        return np.array([r.get(name, np.nan) for r in rows], dtype=np.float64)

    # The sign multipliers (AX_SIGN etc.) are applied here, at the point of ingest,
    # so every downstream calculation sees a consistently signed signal.
    return (t_ms,
            (t_ms - t_ms[0]) / 1000.0,    # ms → seconds, starting at 0
            AX_SIGN * col("accelerometerX"),
            AY_SIGN * col("accelerometerY"),
            AZ_SIGN * col("accelerometerZ"),
            GX_SIGN * col("gyroX"),
            GY_SIGN * col("gyroY"),
            GZ_SIGN * col("gyroZ"))


def imu_sampling_rate_hz(t_ms: np.ndarray) -> float:
    """
    Estimate the IMU sampling rate from the median inter-sample interval.

    We use the median rather than the mean because occasional large gaps
    (caused by buffer overflows or OS scheduling delays on the smartphone)
    would inflate the mean and produce an inaccurately low rate estimate.
    The median is robust to these outliers and reflects the nominal rate.
    """
    if len(t_ms) < 3:
        return float("nan")
    dt = np.diff(t_ms).astype(np.float64) / 1000.0  # intervals in seconds
    dt = dt[dt > 0]   # discard duplicate timestamps (zero-interval artefacts)
    return 1.0 / np.median(dt) if len(dt) > 0 else float("nan")


def get_video_props(video_path: str):
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n_frames, w, h


def pick_left_imu_index(t_imu_s, t_frame_s):
    """
    For each video frame timestamp, find the index of the most recent IMU sample
    that was recorded at or before that time (the nearest sample to the left on
    the time axis). This is the best available IMU reading for that frame.

    searchsorted with side="right" returns the insertion point after any equal values,
    so subtracting 1 steps back to the last sample that is <= t_frame_s.
    The clipping handles the edge case where a frame time falls exactly at or before
    the very first IMU sample.
    """
    idx = np.searchsorted(t_imu_s, t_frame_s, side="right") - 1
    idx[idx < 0] = 0
    idx[idx >= len(t_imu_s)] = len(t_imu_s) - 1
    return idx


def rotate_frame_centered(frame, angle_deg, out_size):
    """
    Rotate the frame by angle_deg around its own centre, placing the result
    centred inside an output canvas of size out_size (width, height).

    cv2.getRotationMatrix2D builds a 2×3 affine matrix that rotates around the
    given pivot point. By default the pivot is the input frame's centre. When the
    input and output canvas sizes differ (e.g., landscape → portrait swap), the
    default translation column of M would place the rotated image off-centre.
    The two lines below correct for that: they shift the translation components so
    the input centre maps exactly to the output centre.

    BORDER_REPLICATE fills any pixels revealed at the edges after rotation by
    repeating the nearest border pixel. This avoids black padding which would
    corrupt pose estimation at frame edges.
    """
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    M[0, 2] += (out_size[0] / 2 - w / 2)   # horizontal shift: input centre → output centre
    M[1, 2] += (out_size[1] / 2 - h / 2)   # vertical shift:   input centre → output centre
    return cv2.warpAffine(frame, M, out_size, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(OUT_DIR)
    log_path = os.path.join(OUT_DIR, LOG_FILENAME)

    with open(log_path, "w", encoding="utf-8") as log:
        def logprint(s=""):
            print(s)
            log.write(str(s) + "\n")

        # ── Load data ──────────────────────────────────────────
        data = load_imu_json_robust(IMU_PATH)
        t_ms, t_imu, ax, ay, az, gx, gy, gz = parse_imu(data)
        imu_hz = imu_sampling_rate_hz(t_ms)
        fps, n_frames_total, orig_w, orig_h = get_video_props(VIDEO_PATH)

        # Log the axis convention at the top of every run so the log is
        # self-documenting - a reader months later will know which physical
        # direction each axis referred to.
        logprint("============================================================")
        logprint(IMU_FRAME_TEXT)
        logprint("------------------------------------------------------------")
        logprint(f"IMU sampling rate: {imu_hz:.3f} Hz")
        logprint(f"Video: {fps:.3f} fps, {n_frames_total} frames, {orig_w}x{orig_h}")
        logprint("============================================================")

        # ── Time alignment ─────────────────────────────────────
        # Drop all IMU samples before SKIP_S, then re-zero the time axis so that
        # t_imu2 = 0 corresponds to the first video frame we will actually process.
        imu_keep = t_imu >= SKIP_S  # boolean mask: True for samples we keep
        t_imu2   = t_imu[imu_keep] - SKIP_S
        ax2, ay2, az2, gz2 = ax[imu_keep], ay[imu_keep], az[imu_keep], gz[imu_keep]

        # ── Orientation auto-detection ─────────────────────────
        # The phone's accelerometer measures the gravity vector projected onto its own axes.
        # In portrait mode the phone is upright, so gravity falls almost entirely along the
        # Y axis → |ay| dominates. In landscape mode the phone is rotated 90°, so gravity
        # projects mostly onto the X axis → |ax| dominates. Comparing the two magnitudes
        # tells us which way the phone was held, which determines how the output canvas
        # must be sized to produce an upright portrait video.
        n_init       = 10
        avg_ax       = np.mean(ax2[:n_init])
        avg_ay       = np.mean(ay2[:n_init])
        is_landscape = np.abs(avg_ax) > np.abs(avg_ay)
        logprint(f"Orientation determined by first {n_init} samples: "
                 f"{'landscape' if is_landscape else 'portrait'}")

        # ── Frame index range ──────────────────────────────────
        start_frame = int(SKIP_S * fps)
        frame_idx   = np.arange(start_frame, n_frames_total, dtype=int)

        # Convert frame numbers to relative timestamps (seconds from SKIP_S).
        # This makes frame times directly comparable to t_imu2.
        t_frame = (frame_idx - start_frame) / fps

        # Discard any video frames whose timestamp exceeds the last available IMU sample.
        # This happens when the IMU recording stops slightly before the video ends.
        valid     = t_frame <= t_imu2[-1]
        frame_idx = frame_idx[valid]
        t_frame   = t_frame[valid]
        n_frames  = len(t_frame)
        logprint(f"Processing {n_frames} frames (frames {frame_idx[0]}–{frame_idx[-1]})")

        # ── Roll estimation ────────────────────────────────────
        # Map each frame's timestamp to the most recent preceding IMU sample.
        left_idx         = pick_left_imu_index(t_imu2, t_frame)
        ax_f, ay_f, gz_f = ax2[left_idx], ay2[left_idx], gz2[left_idx]

        # Accelerometer-only roll estimate.
        # Gravity is a constant downward vector. When the phone is tilted by roll angle φ,
        # the gravity components measured by the phone's accelerometer are approximately:
        #   ax ≈ g·sin(φ),   ay ≈ g·cos(φ)
        # Therefore atan2(ax, ay) = atan2(sin φ, cos φ) = φ.
        # This gives an absolute but noisy estimate - handheld motion adds transient
        # accelerations that corrupt individual frames.
        roll_acc = np.arctan2(ax_f, ay_f)  # radians, absolute roll per frame

        # Complementary filter: fuse gyroscope integration with accelerometer correction.
        # For each frame k:
        #   roll_pred = roll[k-1] + gz[k] * dt   ← gyro integration: smooth, but drifts over time
        #   roll_fuse = α · roll_pred + (1−α) · roll_acc[k]
        #                                         ← blend: gyro dominates short-term (smooth),
        #                                           accelerometer corrects long-term drift (stable)
        # α = 0.98 means each frame is 98% gyro prediction and only 2% accelerometer correction.
        # This suppresses the noise from handheld movement while still anchoring to gravity
        # over the course of several seconds.
        roll_fuse    = np.zeros(n_frames)
        roll_fuse[0] = roll_acc[0]  # initialise from the accelerometer (no previous gyro state)
        for k in range(1, n_frames):
            dt           = t_frame[k] - t_frame[k - 1]
            roll_pred    = roll_fuse[k - 1] + gz_f[k] * dt  # gyro integration, drifts over time
            roll_fuse[k] = ALPHA * roll_pred + (1 - ALPHA) * roll_acc[k]

        # Convert radians → degrees for cv2 (which expects degrees).
        # np.unwrap is applied first: the filter output lives in [-π, π], so a true roll that
        # crosses ±180° would cause the raw value to jump by ~360°, which would apply a
        # ~360° rotation to that one frame. np.unwrap detects such jumps and adds ±2π offsets
        # to make the signal continuous, so the rotation applied to each frame is always the
        # physically correct smooth value.
        rad2deg       = 180.0 / math.pi
        roll_fuse_deg = np.unwrap(roll_fuse) * rad2deg

        # ── Diagnostic plots ───────────────────────────────────
        # These plots are the primary debugging tool for verifying correct IMU sign
        # configuration and filter behaviour before committing to a full video run.
        #
        # What to look for:
        #   Accelerometer panel: ax and ay should respond to tilt; az should be ~9.8 at rest.
        #   GyroZ panel: should show clear spikes when the camera rolls, near-zero when still.
        #   Roll panel: "acc" (raw accelerometer roll) should be noisy but centred on the true
        #               angle; "fuse" (complementary filter) should track it smoothly.
        roll_acc_deg = np.unwrap(roll_acc) * rad2deg   # raw accelerometer roll for comparison

        fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

        axs[0, 0].plot(t_imu2, ax2, label="ax")
        axs[0, 0].plot(t_imu2, ay2, label="ay")
        axs[0, 0].plot(t_imu2, az2, label="az")
        axs[0, 0].set_title("Accelerometer (m/s²)")
        axs[0, 0].set_xlabel("Time (s)")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(t_imu2, gz2, label="gz")
        axs[0, 1].set_title("GyroZ - rotation around optical axis (rad/s)")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[1, 0].plot(t_frame, roll_acc_deg, label="acc (raw, noisy)")
        axs[1, 0].plot(t_frame, roll_fuse_deg, label="fuse (complementary filter)")
        axs[1, 0].set_title("Roll angle applied to video frames (°)")
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        axs[1, 1].axis("off")   # reserved for future use (e.g., keypoint quality metrics)

        plot_path = os.path.join(OUT_DIR, f"{PLOT_BASENAME}__plots.png")
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        logprint(f"[SAVED] plots:      {os.path.abspath(plot_path)}")

        # ── Video processing ───────────────────────────────────
        cap = cv2.VideoCapture(VIDEO_PATH)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx[0])  # jump to the first frame we need

        # For landscape-captured video the raw dimensions are wide (e.g., 1920×1080).
        # Swapping width and height produces a portrait-shaped canvas (e.g., 1080×1920),
        # and the roll rotation will bring the sideways subject upright inside it.
        out_w    = orig_h if is_landscape else orig_w
        out_h    = orig_w if is_landscape else orig_h
        out_size = (out_w, out_h)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, out_size)

        t_rotate_s     = 0.0
        t_total_s      = 0.0
        loop_frames_done = 0

        # We track current_frame ourselves rather than calling cap.set() each iteration.
        # In compressed video formats (H.264/H.265), seeking to an arbitrary frame requires
        # the decoder to find and decode back to the nearest keyframe, which is slow.
        # Sequential cap.read() calls are always fast. The while-loop below fast-forwards
        # past any frames we want to skip by reading and discarding them.
        current_frame = int(frame_idx[0])

        for i in range(n_frames):
            loop_t0 = time.perf_counter()

            # Fast-forward past skipped frames by decoding and discarding.
            target = int(frame_idx[i])
            while current_frame < target:
                cap.read()
                current_frame += 1

            ok, frame = cap.read()
            if not ok:
                break
            current_frame += 1

            # Negate the roll angle: if the camera rolled +5° clockwise, we rotate
            # the frame -5° counter-clockwise to bring it back to level.
            # ROT_SIGN handles the case where the app's IMU axis convention is inverted.
            angle = ROT_SIGN * roll_fuse_deg[i]

            t0         = time.perf_counter()
            stabilized = rotate_frame_centered(frame, angle, out_size)
            t_rotate_s += time.perf_counter() - t0   # cumulative time spent on warp calls

            out.write(stabilized)

            loop_time  = time.perf_counter() - loop_t0
            t_total_s += loop_time
            loop_frames_done += 1

            if LOG_PER_FRAME_TIMINGS:
                # t_rotate_s is cumulative at this point (used for rough trending).
                logprint(f"[frame {i:5d}] warp={t_rotate_s:.6f}s  "
                         f"loop={loop_time:.6f}s  angle={angle:+.2f}°")
            else:
                logprint(f"[frame {i:5d}] angle={angle:+.2f}°")

        cap.release()
        out.release()

        # ── Timing summary ─────────────────────────────────────
        lf = max(loop_frames_done, 1)
        logprint("============================================================")
        logprint("TIMING SUMMARY")
        logprint(f"Frames processed: {loop_frames_done}")
        logprint(f"Total time:       {t_total_s:.3f}s  ({t_total_s / lf * 1000:.2f} ms/frame)")
        logprint(f"Rotation time:    {t_rotate_s:.3f}s  ({t_rotate_s / lf * 1000:.2f} ms/frame)")
        logprint("============================================================")
        logprint(f"[SAVED] stabilized: {os.path.abspath(OUT_VIDEO)}")

    print(f"\nDONE! Logs: {log_path} | Corrected video: {OUT_VIDEO}")


if __name__ == "__main__":
    main()
