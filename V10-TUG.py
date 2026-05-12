"""
TUG Phase Segmentation (MediaPipe Pose Landmarker LITE)

Pipeline overview:
1) Extract pose landmarks per frame.
2) Build smoothed kinematic signals (posture, velocities, oscillation).
3) Segment phases using rule-based thresholds and persistence.
4) Export annotated video + per-phase timings + debug signals.

Outputs:
- annotated video: <name>_annotated.mp4
- excel: <name>_phase_times.xlsx
- debug csv: <name>_signals_debug.csv
"""

import os
import math
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# VIDEO SELECTION
# ==========================================
import tkinter as tk
from tkinter import filedialog


def select_video_file():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select TUG Video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if not video_path:
        raise RuntimeError("No video selected.")
    return video_path


# ==========================================
# CONFIG
# ==========================================
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
MODEL_PATH = "pose_landmarker_lite.task"

PHASES_ORDER = [
    "SITTING_START",
    "STAND_UP",
    "WALK_FORWARD",
    "TURN",
    "WALK_BACK",
    "SIT_DOWN",
    "SITTING_END",
]

SCALE_FACTOR = 1.75  # inference scale-up for higher landmark precision

SMOOTH_WIN_BASE = 15  # base Savitzky-Golay window (frames)
SMOOTH_POLY = 3  # Savitzky-Golay polynomial order

PERSIST_BASE = {  # minimum consecutive frames to confirm a transition
    "SITTING_TO_STAND": 10,
    "STAND_TO_WALK": 6,
    "WALK_TO_TURN": 6,
    "TURN_TO_WALKBACK": 6,
    "WALKBACK_TO_SITDOWN": 10,
    "SITDOWN_TO_SITTING_END": 10,
}
PERSIST_MIN_FRAMES = 3  # floor for any adaptive persistence

OSC_WIN_SEC = 0.6  # window for gait oscillation estimation
TRANSITION_BANNER_SEC = 0.45  # on-screen phase-change banner duration
MIN_WALKBACK_SEC = 1.2  # enforce minimum walk-back duration
MIN_WALK_FORWARD_SEC = 2.0  # enforce minimum walk-forward duration
MIN_TURN_SEC = 1.2  # enforce minimum turn duration
MIN_SITDOWN_SEC = 0.45  # enforce minimum sit-down duration

QUALITY_HOLD = True  # if True, gate decisions by landmark quality
QUALITY_THRESH = 0.30  # strict quality threshold
QUALITY_SOFT_MULT = 0.50  # soft threshold multiplier
QUALITY_WIN_SEC = 0.25  # rolling window for quality smoothing

# Turn fusion thresholds
TURN_OSC_DROP_FACTOR = 0.75
TURN_LATERAL_SPIKE_PCTL = 80

TURN_SEARCH_BACK_SEC = 3.0
TURN_SEARCH_FWD_SEC = 4.0
TURN_EARLY_MIN_GAP_SEC = 0.5

# Fast-walk adaptation
CADENCE_HZ_FAST = 2.4
CADENCE_HZ_VERY_FAST = 3.0


# ==========================================
# HELPERS
# ==========================================
def ensure_model(model_path: str = MODEL_PATH, url: str = MODEL_URL):
    """Download the pose model if missing, return local path."""
    if not os.path.exists(model_path):
        print(f"Downloading LITE model to: {model_path}")
        urllib.request.urlretrieve(url, model_path)
    return model_path


def safe_savgol(x, win, poly=SMOOTH_POLY):
    """Safely apply Savitzky-Golay smoothing with odd window bounds."""
    x = np.asarray(x, dtype=float)
    if len(x) < max(7, win):
        return x.copy()
    w = win if win % 2 == 1 else win + 1
    if w >= len(x):
        w = len(x) - 1 if (len(x) - 1) % 2 == 1 else len(x) - 2
    w = max(w, 7)
    return savgol_filter(x, window_length=w, polyorder=min(poly, w - 2))


def angle_3pt(a, b, c):
    """Return angle ABC (degrees) using 2D landmark coordinates."""
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]
    v1 = np.array([ax - bx, ay - by], dtype=float)
    v2 = np.array([cx - bx, cy - by], dtype=float)
    n1 = np.linalg.norm(v1) + 1e-9
    n2 = np.linalg.norm(v2) + 1e-9
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def draw_label(frame, text, xy=(20, 45), scale=1.0):
    """Draw high-contrast label on video frame."""
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_transition(frame, text, y=130):
    """Draw a transition banner between phases."""
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)


@dataclass
class PhaseInterval:
    phase: str
    start_frame: int
    end_frame: int

    def to_row(self, fps):
        """Serialize interval with time columns (seconds)."""
        return {
            "phase": self.phase,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": self.start_frame / fps,
            "end_time_s": self.end_frame / fps,
            "duration_s": max(0.0, (self.end_frame - self.start_frame) / fps),
        }


def get_landmark_xyz(landmarks, idx):
    """Extract (x, y, z) from a landmark."""
    lm = landmarks[idx]
    return (lm.x, lm.y, lm.z)


def get_lm_quality(lm, idxs):
    """Compute mean visibility across a set of landmark indices."""
    vis = []
    finite = []
    for i in idxs:
        p = lm[i]
        finite.append(np.isfinite(p.x) and np.isfinite(p.y) and np.isfinite(p.z))
        vis.append(float(getattr(p, "visibility", 1.0)))
    if not all(finite):
        return 0.0
    return float(np.clip(np.mean(vis), 0.0, 1.0))


def sustained_first_true(mask: np.ndarray, start: int, need: int):
    """Return first index with 'need' consecutive True values (forward)."""
    consec = 0
    for i in range(start, len(mask)):
        if mask[i]:
            consec += 1
            if consec >= need:
                return i - (need - 1)
        else:
            consec = 0
    return None


def sustained_last_true_before(mask: np.ndarray, end: int, need: int):
    """Return last index before 'end' with 'need' consecutive True values (backward)."""
    consec = 0
    for i in range(end, -1, -1):
        if mask[i]:
            consec += 1
            if consec >= need:
                return i
        else:
            consec = 0
    return None


def estimate_cadence_hz(walk_osc: np.ndarray, fps: float):
    """Estimate dominant gait cadence from oscillation signal."""
    n = len(walk_osc)
    if n < int(3 * fps):
        return None
    a = int(0.20 * n)
    b = int(0.80 * n)
    x = np.asarray(walk_osc[a:b], dtype=float)
    x = x - np.nanmean(x)
    if np.all(np.abs(x) < 1e-9):
        return None
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fps)
    spec = np.abs(np.fft.rfft(x))
    lo, hi = 0.8, 4.0
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return None
    return float(freqs[m][np.argmax(spec[m])])


def adapt_persist(persist_base: dict, cadence_hz):
    """Scale persistence thresholds for fast walkers."""
    persist = dict(persist_base)
    if cadence_hz is None:
        return persist, 1.0
    if cadence_hz >= CADENCE_HZ_VERY_FAST:
        scale = 0.55
    elif cadence_hz >= CADENCE_HZ_FAST:
        scale = 0.70
    else:
        scale = 1.0
    for k in persist:
        persist[k] = max(PERSIST_MIN_FRAMES, int(round(persist[k] * scale)))
    return persist, scale


# ==========================================
# SIGNAL EXTRACTION
# ==========================================
def extract_signals(video_path: str):
    """Run pose inference and build per-frame kinematic signals."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model_path = ensure_model()

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
        num_poses=1
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Landmark indices (MediaPipe Pose)
    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28
    key_idxs = [L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK]

    rows = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if SCALE_FACTOR != 1.0:
            frame_inf = cv2.resize(
                frame_bgr,
                (int(frame_bgr.shape[1] * SCALE_FACTOR), int(frame_bgr.shape[0] * SCALE_FACTOR)),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            frame_inf = frame_bgr

        frame_rgb = cv2.cvtColor(frame_inf, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((frame_idx / fps) * 1000.0)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm = result.pose_landmarks[0]
            q = get_lm_quality(lm, key_idxs)

            Lsh = get_landmark_xyz(lm, L_SH)
            Rsh = get_landmark_xyz(lm, R_SH)
            Lhip = get_landmark_xyz(lm, L_HIP)
            Rhip = get_landmark_xyz(lm, R_HIP)
            Lk = get_landmark_xyz(lm, L_KNEE)
            Rk = get_landmark_xyz(lm, R_KNEE)
            La = get_landmark_xyz(lm, L_ANK)
            Ra = get_landmark_xyz(lm, R_ANK)

            # hk normalized: proxy for knee extension relative to leg length
            # - high hk_norm: more extended knee (standing)
            # - low hk_norm: more flexed knee (sitting)
            hk_L = float(np.hypot(Lhip[0] - Lk[0], Lhip[1] - Lk[1]))
            hk_R = float(np.hypot(Rhip[0] - Rk[0], Rhip[1] - Rk[1]))
            hk = 0.5 * (hk_L + hk_R)

            ha_L = float(np.hypot(Lhip[0] - La[0], Lhip[1] - La[1]))
            ha_R = float(np.hypot(Rhip[0] - Ra[0], Rhip[1] - Ra[1]))
            ha = max(1e-6, 0.5 * (ha_L + ha_R))
            hk_norm = hk / ha

            # hip_y: vertical pelvis position (higher = more standing in image coords)
            hip_y = 0.5 * (Lhip[1] + Rhip[1])
            # knee_ang: average knee joint angle (larger = more extended)
            knee_ang = 0.5 * (angle_3pt(Lhip, Lk, La) + angle_3pt(Rhip, Rk, Ra))

            # centers: mid-hip and mid-ankle positions (track body translation)
            hip_cx = 0.5 * (Lhip[0] + Rhip[0])
            hip_cy = 0.5 * (Lhip[1] + Rhip[1])
            hip_z = 0.5 * (Lhip[2] + Rhip[2])

            # ankles: keep BOTH individual + center (turn detection and gait)
            Lax, Lay = La[0], La[1]
            Rax, Ray = Ra[0], Ra[1]
            ank_cx = 0.5 * (Lax + Rax)
            ank_cy = 0.5 * (Lay + Ray)

            # ank_sep_x: lateral ankle separation (gait oscillation proxy)
            ank_sep_x = (Lax - Rax)

            # turning signals: torso yaw (z vs x) and shoulder angle (2D)
            yaw_sh = math.atan2((Lsh[2] - Rsh[2]), (Lsh[0] - Rsh[0] + 1e-9))
            yaw_hip = math.atan2((Lhip[2] - Rhip[2]), (Lhip[0] - Rhip[0] + 1e-9))
            yaw_torso = 0.5 * (yaw_sh + yaw_hip)
            sh_ang_2d = math.atan2((Lsh[1] - Rsh[1]), (Lsh[0] - Rsh[0] + 1e-9))

            # facing signals: left-right ordering of shoulders/hips (sign flips with turn)
            facing_sh = (Lsh[0] - Rsh[0])
            facing_hip = (Lhip[0] - Rhip[0])

            rows.append({
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "pose_ok": 1,
                "quality": q,

                "hk_norm": hk_norm,
                "hip_y": hip_y,
                "knee_ang": knee_ang,

                "hip_cx": hip_cx,
                "hip_cy": hip_cy,
                "hip_z": hip_z,

                "L_ank_x": Lax, "L_ank_y": Lay,
                "R_ank_x": Rax, "R_ank_y": Ray,
                "ank_cx": ank_cx,
                "ank_cy": ank_cy,

                "ank_sep_x": ank_sep_x,

                "yaw_torso": yaw_torso,
                "sh_ang_2d": sh_ang_2d,

                "facing_sh": facing_sh,
                "facing_hip": facing_hip,
            })
        else:
            rows.append({
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "pose_ok": 0,
                "quality": 0.0,

                "hk_norm": np.nan,
                "hip_y": np.nan,
                "knee_ang": np.nan,

                "hip_cx": np.nan,
                "hip_cy": np.nan,
                "hip_z": np.nan,

                "L_ank_x": np.nan, "L_ank_y": np.nan,
                "R_ank_x": np.nan, "R_ank_y": np.nan,
                "ank_cx": np.nan,
                "ank_cy": np.nan,

                "ank_sep_x": np.nan,

                "yaw_torso": np.nan,
                "sh_ang_2d": np.nan,

                "facing_sh": np.nan,
                "facing_hip": np.nan,
            })

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    df.interpolate(limit_direction="both", inplace=True)

    fps_val = fps

    # quality smoothing
    qwin = max(1, int(QUALITY_WIN_SEC * fps_val))
    df["quality_s"] = df["quality"].rolling(qwin, center=True, min_periods=1).mean().values

    win = SMOOTH_WIN_BASE
    df["hk_s"] = safe_savgol(df["hk_norm"].values, win)
    df["hip_y_s"] = safe_savgol(df["hip_y"].values, win)
    df["knee_ang_s"] = safe_savgol(df["knee_ang"].values, win)
    df["hip_z_s"] = safe_savgol(df["hip_z"].values, win)

    # ankle velocities: use ankle center and each ankle for robust turning
    # ank_v_max captures the more active foot during turning
    def vel(x, y):
        return np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2) * fps_val

    df["ank_v"] = vel(df["ank_cx"].values, df["ank_cy"].values)
    df["L_ank_v"] = vel(df["L_ank_x"].values, df["L_ank_y"].values)
    df["R_ank_v"] = vel(df["R_ank_x"].values, df["R_ank_y"].values)
    df["ank_v_max"] = np.maximum(df["L_ank_v"].values, df["R_ank_v"].values)

    df["hip_v"] = vel(df["hip_cx"].values, df["hip_cy"].values)

    # lateral ankle speed (center): highlights side-to-side during turning
    df["ank_x_v"] = np.abs(np.diff(df["ank_cx"].values, prepend=df["ank_cx"].values[0])) * fps_val

    # walk oscillation amplitude from ankle separation variability
    # higher oscillation indicates steady gait; dips suggest turning or stopping
    ank_sep_s = safe_savgol(df["ank_sep_x"].values, win)
    osc_win = max(7, int(OSC_WIN_SEC * fps_val))
    kernel = np.ones(osc_win) / osc_win
    ex = np.convolve(ank_sep_s, kernel, mode="same")
    ex2 = np.convolve(ank_sep_s * ank_sep_s, kernel, mode="same")
    df["walk_osc_raw"] = np.sqrt(np.maximum(0.0, ex2 - ex * ex))

    cadence_hz = estimate_cadence_hz(df["walk_osc_raw"].values, fps_val)

    # adaptive smoothing stronger if cadence is high
    if cadence_hz is not None and cadence_hz >= CADENCE_HZ_FAST:
        win2 = min(31, max(win, 21))
    else:
        win2 = win

    df["ank_v_s"] = safe_savgol(df["ank_v"].values, win2)
    df["ank_v_max_s"] = safe_savgol(df["ank_v_max"].values, win2)
    df["hip_v_s"] = safe_savgol(df["hip_v"].values, win2)
    df["ank_x_v_s"] = safe_savgol(df["ank_x_v"].values, win2)
    df["walk_osc"] = safe_savgol(df["walk_osc_raw"].values, win2)

    # hk derivative: posture change speed (stand-up vs sit-down)
    df["hk_dot"] = np.diff(df["hk_s"].values, prepend=df["hk_s"].values[0]) * fps_val
    df["hk_dot_s"] = safe_savgol(df["hk_dot"].values, win2)

    # hip_y derivative: vertical motion speed (downward during sit-down)
    df["hip_y_dot"] = np.diff(df["hip_y_s"].values, prepend=df["hip_y_s"].values[0]) * fps_val
    df["hip_y_dot_s"] = safe_savgol(df["hip_y_dot"].values, win2)

    # hip_z derivative: depth motion speed (camera axis, forward/back movement)
    df["hip_z_dot"] = np.diff(df["hip_z_s"].values, prepend=df["hip_z_s"].values[0]) * fps_val
    df["hip_z_dot_s"] = safe_savgol(df["hip_z_dot"].values, win2)

    # yaw unwrap + rate: torso rotation speed (turning)
    yaw = np.unwrap(df["yaw_torso"].values)
    df["yaw_s"] = safe_savgol(yaw, win2)
    df["yaw_rate"] = np.abs(np.diff(df["yaw_s"].values, prepend=df["yaw_s"].values[0])) * fps_val
    df["yaw_rate_s"] = safe_savgol(df["yaw_rate"].values, win2)

    # shoulder 2D unwrap + rate: shoulder rotation speed (turning)
    sh2d = np.unwrap(df["sh_ang_2d"].values)
    df["sh2d_s"] = safe_savgol(sh2d, win2)
    df["sh2d_rate"] = np.abs(np.diff(df["sh2d_s"].values, prepend=df["sh2d_s"].values[0])) * fps_val
    df["sh2d_rate_s"] = safe_savgol(df["sh2d_rate"].values, win2)

    # facing combined + rate: left-right torso facing change (sign flips on turn)
    df["facing_sh_s"] = safe_savgol(df["facing_sh"].values, win2)
    df["facing_hip_s"] = safe_savgol(df["facing_hip"].values, win2)
    df["facing_s"] = 0.6 * df["facing_sh_s"].values + 0.4 * df["facing_hip_s"].values
    df["facing_rate"] = safe_savgol(np.abs(np.diff(df["facing_s"].values, prepend=df["facing_s"].values[0])) * fps_val, win2)

    df.attrs["cadence_hz_est"] = cadence_hz
    df.attrs["smooth_win_used"] = win2
    return df, fps, width0, height0


# ==========================================
# TURN WINDOW (ankle fusion)
# ==========================================
def compute_turn_window(df: pd.DataFrame, fps: float, debug: dict, PERSIST: dict):
    """Detect turn start/center/end from torso + ankle signals."""
    n = len(df)

    # facing_s: smoothed left-right torso orientation proxy
    # facing_rate: absolute rate of facing change (turning speed indicator)
    facing = df["facing_s"].values
    facing_rate = df["facing_rate"].values

    q = df["quality_s"].values
    good = (q >= QUALITY_THRESH) if QUALITY_HOLD else np.ones(n, dtype=bool)

    # walk_osc: gait oscillation amplitude (high during steady walking)
    # yaw_rate_s: torso yaw speed (turning)
    # sh2d_rate_s: shoulder rotation speed (turning)
    walk_osc = df["walk_osc"].values
    yaw_rate = df["yaw_rate_s"].values
    sh_rate = df["sh2d_rate_s"].values

    # ankle signals:
    # - ank_v_max: max foot speed (busy feet during turn)
    # - ank_x_v: lateral ankle speed (sideways stepping during turn)
    ank_v_max = df["ank_v_max_s"].values
    ank_x_v = df["ank_x_v_s"].values
    hip_v = df["hip_v_s"].values
    hip_z_dot = df["hip_z_dot_s"].values

    def robust_z(x):
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med))) * 1.4826
        mad = mad if mad > 1e-9 else 1.0
        return (x - med) / mad, med, mad

    # thresholds for walking/turning signals (percentile-based, per video)
    osc_walk_thr = float(np.nanpercentile(walk_osc, 70))
    yaw_turn_thr = max(float(np.nanpercentile(yaw_rate, 60)), 0.06)
    sh_turn_thr = max(float(np.nanpercentile(sh_rate, 60)), 0.06)

    lat_thr = float(np.nanpercentile(ank_x_v, TURN_LATERAL_SPIKE_PCTL))
    ankle_turn_thr = float(np.nanpercentile(ank_v_max, 75))  # “busy feet” during turn

    fr_thr = float(np.nanpercentile(facing_rate, 75))
    hip_resume_thr = float(np.nanpercentile(hip_v, 40))
    depth_resume_thr = float(np.nanpercentile(np.abs(hip_z_dot), 50))

    debug.update({
        "osc_walk_thr": osc_walk_thr,
        "yaw_turn_thr": yaw_turn_thr,
        "sh_turn_thr": sh_turn_thr,
        "ank_lat_spike_thr": lat_thr,
        "ankle_turn_thr": ankle_turn_thr,
        "facing_rate_thr": fr_thr,
        "hip_resume_thr": hip_resume_thr,
        "depth_resume_thr": depth_resume_thr,
    })

    # torso_turning: torso/shoulder rotation above threshold
    torso_turning = (yaw_rate >= yaw_turn_thr) | (sh_rate >= sh_turn_thr)

    # ankle-based turn: lateral spike AND reduced gait osc,
    # OR very high ankle activity with low osc (turning in place)
    ankle_turning = (
        ((ank_x_v >= lat_thr) & (walk_osc < TURN_OSC_DROP_FACTOR * osc_walk_thr))
        | ((ank_v_max >= ankle_turn_thr) & (walk_osc < TURN_OSC_DROP_FACTOR * osc_walk_thr))
    )

    turning_any = (torso_turning | ankle_turning) & good

    # baseline and tail of facing (standing/walking orientation)
    base_n = max(10, int(1.0 * fps))
    f0 = float(np.nanmedian(facing[:base_n]))
    f1 = float(np.nanmedian(facing[int(0.70 * n):])) if n > 10 else f0
    debug["f0_baseline"] = f0
    debug["f1_tail"] = f1

    mid = 0.5 * (f0 + f1)
    s0 = int(0.20 * n)
    s1 = int(0.80 * n)
    center = int(s0 + np.argmin(np.abs(facing[s0:s1] - mid)))
    debug["turn_center"] = center

    # signal-based center override:
    # use the first sustained turning segment in the mid-window as TURN center
    # turn_sig: normalized torso yaw + lateral foot motion
    # (yaw dominates; lateral ankle helps detect sideways stepping)
    z_yaw, yaw_med, yaw_mad = robust_z(yaw_rate)
    z_lat, lat_med, lat_mad = robust_z(ank_x_v)
    turn_sig = 0.7 * z_yaw + 0.3 * z_lat
    center_sig = int(s0 + np.nanargmax(turn_sig[s0:s1]))
    sig_med = float(np.nanmedian(turn_sig))
    sig_mad = float(np.nanmedian(np.abs(turn_sig - sig_med))) * 1.4826
    sig_mad = sig_mad if sig_mad > 1e-9 else 1.0
    sig_thr = sig_med + 0.5 * sig_mad

    # pick earliest sustained turning segment in the middle window
    turn_mask_low = (turn_sig >= sig_thr) & good
    idx_first = sustained_first_true(turn_mask_low[s0:s1], 0, max(3, PERSIST["WALK_TO_TURN"] - 2))
    if idx_first is not None:
        seg_start = s0 + idx_first
        seg_end = seg_start
        while seg_end + 1 < s1 and turn_mask_low[seg_end + 1]:
            seg_end += 1
        center = int(0.5 * (seg_start + seg_end))
        debug["turn_center_override"] = True
    else:
        center = center_sig
        debug["turn_center_override"] = True
    debug["turn_center_sig"] = int(center_sig)
    debug["turn_sig_thr"] = sig_thr

    # start: last sustained turning block near center (or facing_rate spike)
    back = int(TURN_SEARCH_BACK_SEC * fps)
    start_search = max(0, center - back)

    window_mask = turning_any[start_search:center + 1]
    idx = sustained_last_true_before(window_mask, len(window_mask) - 1, PERSIST["WALK_TO_TURN"])
    if idx is not None:
        turn_start = start_search + idx
    else:
        mask2 = (facing_rate >= fr_thr) & good
        idx2 = sustained_last_true_before(mask2[start_search:center + 1], center - start_search, PERSIST["WALK_TO_TURN"])
        turn_start = (start_search + idx2) if idx2 is not None else center

    early_need = max(3, PERSIST["WALK_TO_TURN"] - 2)
    idx_first = sustained_first_true(window_mask, 0, early_need)
    if idx_first is not None:
        early_start = start_search + idx_first
        min_gap = int(TURN_EARLY_MIN_GAP_SEC * fps)
        if early_start <= turn_start - min_gap:
            turn_start = early_start

    # end: walking resumes after turn (ankle-led)
    # walking resumes = walk_osc rises OR ankle speed stabilizes into gait
    walk_resume = (
        (walk_osc >= osc_walk_thr)
        | (ank_v_max >= float(np.nanpercentile(ank_v_max, 65)))
        | (hip_v >= hip_resume_thr)
        | (np.abs(hip_z_dot) >= depth_resume_thr)
    )
    walk_resume = walk_resume & good

    idx_end_resume = sustained_first_true(walk_resume, center, PERSIST["TURN_TO_WALKBACK"])

    not_turning = (~turning_any) & good
    idx_end_not_turning = sustained_first_true(not_turning, center, PERSIST["TURN_TO_WALKBACK"])

    if idx_end_resume is not None and idx_end_not_turning is not None:
        turn_end = min(idx_end_resume, idx_end_not_turning)
    elif idx_end_resume is not None:
        turn_end = idx_end_resume
    elif idx_end_not_turning is not None:
        turn_end = idx_end_not_turning
    else:
        turn_end = min(n - 1, center + int(1.0 * fps))

    # center-anchored symmetry: if turning signal exists on both sides, balance around center
    left_mask = turning_any[start_search:center + 1]
    right_end = min(n - 1, center + int(TURN_SEARCH_FWD_SEC * fps))
    right_mask = turning_any[center:right_end + 1]

    left_idx = sustained_last_true_before(left_mask, len(left_mask) - 1, PERSIST["WALK_TO_TURN"])
    right_idx = sustained_first_true(right_mask, 0, PERSIST["TURN_TO_WALKBACK"])

    if left_idx is not None and right_idx is not None:
        left_len = center - (start_search + left_idx)
        right_len = right_idx
        half_len = min(left_len, right_len)
        sym_start = center - half_len
        sym_end = center + half_len
        if sym_end > sym_start:
            turn_start = sym_start
            turn_end = sym_end

    if turn_end <= turn_start:
        turn_end = min(n - 1, turn_start + int(0.5 * fps))

    # change-point based turn window (movement change vs. baseline)

    # change-point: short vs long moving average difference
    short = max(3, int(0.25 * fps))
    long = max(short + 2, int(0.9 * fps))
    k_short = np.ones(short) / short
    k_long = np.ones(long) / long
    m_short = np.convolve(turn_sig, k_short, mode="same")
    m_long = np.convolve(turn_sig, k_long, mode="same")
    delta = m_short - m_long

    # delta thresholding using robust stats (MAD)
    d_med = float(np.nanmedian(delta))
    d_mad = float(np.nanmedian(np.abs(delta - d_med))) * 1.4826
    d_mad = d_mad if d_mad > 1e-9 else 1.0
    d_thr_on = d_med + 0.6 * d_mad
    d_thr_off = d_med + 0.3 * d_mad

    cp_active = (delta >= d_thr_on) & good

    cp_start = sustained_first_true(cp_active, start_search, PERSIST["WALK_TO_TURN"])

    end_search_start = center if cp_start is None else max(center, cp_start + PERSIST["WALK_TO_TURN"])
    cp_end = sustained_first_true(delta < d_thr_off, end_search_start, PERSIST["TURN_TO_WALKBACK"])
    if cp_end is None:
        cp_end = sustained_first_true((~cp_active) & good, center, PERSIST["TURN_TO_WALKBACK"])

    if cp_start is not None and cp_end is not None and cp_end > cp_start:
        turn_start = cp_start
        turn_end = cp_end

    # signal-threshold window around center (fallback/tighten)
    turn_mask = (turn_sig >= sig_thr) & good
    left_mask = turn_mask[start_search:center + 1]
    right_end = min(n - 1, center + int(TURN_SEARCH_FWD_SEC * fps))
    right_mask = turn_mask[center:right_end + 1]
    left_idx = sustained_last_true_before(left_mask, len(left_mask) - 1, PERSIST["WALK_TO_TURN"])
    right_idx = sustained_first_true(right_mask, 0, PERSIST["TURN_TO_WALKBACK"])
    if left_idx is not None and right_idx is not None:
        t_start = start_search + left_idx
        t_end = center + right_idx
        if t_end > t_start:
            turn_start = t_start
            turn_end = t_end

    debug.update({
        "cp_d_med": d_med,
        "cp_d_mad": d_mad,
        "cp_d_thr_on": d_thr_on,
        "cp_d_thr_off": d_thr_off,
        "cp_short_win": short,
        "cp_long_win": long,
    })

    debug["turn_start"] = int(turn_start)
    debug["turn_end"] = int(turn_end)
    return int(turn_start), int(center), int(turn_end)


# ==========================================
# PHASE SEGMENTATION (ankle-led after turn)
# ==========================================
def segment_phases(df: pd.DataFrame, fps: float):
    """Segment TUG phases using posture, gait, and turn cues."""
    n = len(df)

    # posture signals
    # hk_s: normalized knee extension (higher = more standing)
    # hip_y_s: vertical pelvis position (higher = more standing)
    # knee_ang_s: knee angle (higher = more extended)
    hk = df["hk_s"].values
    hipy = df["hip_y_s"].values
    knee = df["knee_ang_s"].values

    # posture derivatives (motion direction)
    # hk_dot_s: knee extension speed (positive during stand-up)
    # hip_y_dot_s: vertical pelvis speed (positive during sit-down in image coords)
    hkdot = df["hk_dot_s"].values
    hipy_dot = df["hip_y_dot_s"].values

    # gait / translation signals
    # walk_osc: step-to-step oscillation (gait strength)
    # ank_v_s: ankle center speed (overall gait motion)
    # ank_v_max_s: max ankle speed (active foot)
    # hip_v_s: pelvis speed (forward/back translation)
    walk_osc = df["walk_osc"].values
    ank_v = df["ank_v_s"].values
    ank_v_max = df["ank_v_max_s"].values
    hip_v = df["hip_v_s"].values
    hip_z_dot = df["hip_z_dot_s"].values

    q = df["quality_s"].values
    good_strict = (q >= QUALITY_THRESH) if QUALITY_HOLD else np.ones(n, dtype=bool)
    good_soft = (q >= (QUALITY_THRESH * QUALITY_SOFT_MULT)) if QUALITY_HOLD else np.ones(n, dtype=bool)

    cadence_hz = df.attrs.get("cadence_hz_est", None)
    PERSIST, persist_scale = adapt_persist(PERSIST_BASE, cadence_hz)

    debug = {
        "cadence_hz_est": cadence_hz,
        "persist_scale": persist_scale,
        "PERSIST": dict(PERSIST),
        "smooth_win_used": df.attrs.get("smooth_win_used", SMOOTH_WIN_BASE),
        "QUALITY_THRESH": QUALITY_THRESH,
        "QUALITY_SOFT_THRESH": float(QUALITY_THRESH * QUALITY_SOFT_MULT),
        "good_strict_frac": float(np.mean(good_strict)),
        "good_soft_frac": float(np.mean(good_soft)),
    }

    # posture thresholds: sitting vs standing ranges (percentiles within this clip)
    hk_low, hk_high = float(np.nanpercentile(hk, 10)), float(np.nanpercentile(hk, 90))
    hipy_low, hipy_high = float(np.nanpercentile(hipy, 10)), float(np.nanpercentile(hipy, 90))
    knee_low, knee_high = float(np.nanpercentile(knee, 10)), float(np.nanpercentile(knee, 90))

    sit_hk_thr = (hk_low + hk_high) / 2 - 0.10 * (hk_high - hk_low)
    sit_hipy_thr = (hipy_low + hipy_high) / 2 + 0.10 * (hipy_high - hipy_low)
    sit_knee_thr = (knee_low + knee_high) / 2 - 0.05 * (knee_high - knee_low)
    sit_hk_near = sit_hk_thr + 0.05 * (hk_high - hk_low)
    sit_hipy_near = sit_hipy_thr - 0.05 * (hipy_high - hipy_low)
    sit_knee_near = sit_knee_thr + 0.05 * (knee_high - knee_low)

    # walk thresholds (ankle-led)
    # strict thresholds define "definitely walking"
    # lenient thresholds define "likely walking / post-turn walk"
    osc_walk_thr = float(np.nanpercentile(walk_osc, 70))
    ank_walk_thr = float(max(np.nanpercentile(ank_v_max, 70), np.nanmean(ank_v_max) + 0.4 * np.nanstd(ank_v_max)))
    hip_walk_thr = float(max(np.nanpercentile(hip_v, 60), np.nanmean(hip_v) + 0.25 * np.nanstd(hip_v)))
    depth_walk_thr = float(np.nanpercentile(np.abs(hip_z_dot), 70))

    osc_walk_thr_len = float(np.nanpercentile(walk_osc, 60))
    ank_walk_thr_len = float(np.nanpercentile(ank_v_max, 60))
    hip_walk_thr_len = float(np.nanpercentile(hip_v, 40))
    depth_walk_thr_len = float(np.nanpercentile(np.abs(hip_z_dot), 50))

    debug.update({
        "osc_walk_thr": osc_walk_thr,
        "ank_walk_thr_max": ank_walk_thr,
        "hip_walk_thr": hip_walk_thr,
        "depth_walk_thr": depth_walk_thr,
        "osc_walk_thr_len": osc_walk_thr_len,
        "ank_walk_thr_len": ank_walk_thr_len,
        "hip_walk_thr_len": hip_walk_thr_len,
        "depth_walk_thr_len": depth_walk_thr_len,
        "sit_hk_near": sit_hk_near,
        "sit_hipy_near": sit_hipy_near,
        "sit_knee_near": sit_knee_near,
    })

    # hkdot thresholds: stand up vs sit down motion
    hkdot_std = float(np.nanstd(hkdot))
    descent_thr = -max(0.10 * hkdot_std, 0.02)
    ascent_thr = +max(0.10 * hkdot_std, 0.02)
    debug.update({"ascent_thr_hkdot": ascent_thr, "descent_thr_hkdot": descent_thr})

    # hipy_dot threshold for sit-down
    hipy_dot_std = float(np.nanstd(hipy_dot))
    down_thr = max(0.10 * hipy_dot_std, 0.02)
    debug["down_thr_hipy_dot"] = down_thr

    def is_sitting_soft(i):
        # 2-of-3 posture cues indicates sitting-ish posture
        return good_soft[i] and (((hk[i] <= sit_hk_thr) + (hipy[i] >= sit_hipy_thr) + (knee[i] <= sit_knee_thr)) >= 2)

    def is_walking_strict(i):
        # strict gait detection for WALK_FORWARD
        return good_strict[i] and (
            (ank_v_max[i] >= ank_walk_thr)
            or (walk_osc[i] >= osc_walk_thr)
            or (hip_v[i] >= hip_walk_thr)
            or (abs(hip_z_dot[i]) >= depth_walk_thr)
        )

    def is_walking_lenient(i):
        # lenient gait detection for WALK_BACK after turn
        return good_soft[i] and (
            (ank_v_max[i] >= ank_walk_thr_len)
            or (walk_osc[i] >= osc_walk_thr_len)
            or (hip_v[i] >= hip_walk_thr_len)
            or (abs(hip_z_dot[i]) >= depth_walk_thr_len)
        )

    def is_stand_up_motion(i):
        # positive hk_dot indicates knee extension (stand-up)
        return good_strict[i] and (hkdot[i] >= ascent_thr)

    def sit_down_strong(i):
        # ankle drop + posture descent
        return (good_soft[i]
                and (hkdot[i] <= descent_thr)
                and (hipy_dot[i] >= down_thr)
                and (walk_osc[i] < osc_walk_thr)
                and (ank_v_max[i] < ank_walk_thr)
                and (((hk[i] <= sit_hk_near) + (hipy[i] >= sit_hipy_near) + (knee[i] <= sit_knee_near)) >= 1))

    def sit_down_posture_low_motion(i):
        # sitting posture plus low motion (transition into sitting)
        return (good_soft[i]
                and is_sitting_soft(i)
                and (walk_osc[i] < osc_walk_thr_len)
                and (ank_v_max[i] < ank_walk_thr_len)
                and (hip_v[i] < hip_walk_thr_len))

    # turn window uses ankle fusion (torso + ankle + change-point)
    turn_start, turn_center, turn_end = compute_turn_window(df, fps, debug, PERSIST)

    # sitting start end (quick): initial stable sitting
    init_need = max(6, int(0.20 * fps))
    sit_run = 0
    init_sit_end = 0
    for t in range(n):
        if is_sitting_soft(t):
            sit_run += 1
            if sit_run >= init_need:
                init_sit_end = t
                break
        else:
            sit_run = 0
    min_start_frame = max(init_sit_end, int(0.8 * fps))

    stand_start = sustained_first_true(np.array([is_stand_up_motion(i) for i in range(n)]), min_start_frame, PERSIST["SITTING_TO_STAND"])
    if stand_start is None:
        stand_start = min_start_frame

    walk_start = sustained_first_true(np.array([is_walking_strict(i) for i in range(n)]), stand_start, PERSIST["STAND_TO_WALK"])
    if walk_start is None:
        walk_start = min(n - 1, stand_start + int(0.4 * fps))

    # enforce a minimum WALK_FORWARD duration before TURN can start
    min_wf = int(MIN_WALK_FORWARD_SEC * fps)
    if turn_start < walk_start + min_wf:
        shift = (walk_start + min_wf) - turn_start
        turn_start = min(n - 1, turn_start + shift)
        turn_end = min(n - 1, turn_end + shift)

    # enforce a minimum TURN duration
    min_turn = int(MIN_TURN_SEC * fps)
    if turn_end < turn_start + min_turn:
        turn_end = min(n - 1, turn_start + min_turn)

    walk_back_start = sustained_first_true(np.array([is_walking_lenient(i) for i in range(n)]), turn_end, PERSIST["TURN_TO_WALKBACK"])
    if walk_back_start is None:
        walk_back_start = min(n - 1, turn_end)

    # enforce a minimum WALK_BACK duration before SIT_DOWN can start
    min_walkback = max(PERSIST["TURN_TO_WALKBACK"], int(MIN_WALKBACK_SEC * fps))
    sit_down_search_start = min(n - 1, walk_back_start + min_walkback)

    sitdown_need = max(PERSIST_MIN_FRAMES, max(3, int(0.5 * PERSIST["WALKBACK_TO_SITDOWN"])))
    # stop_walking: low gait oscillation + low ankle/hip speed
    stop_walking = np.array([
        good_soft[i]
        and (walk_osc[i] < osc_walk_thr_len)
        and (ank_v_max[i] < ank_walk_thr_len)
        and (hip_v[i] < hip_walk_thr_len)
        for i in range(n)
    ])

    cand_strong = sustained_first_true(
        np.array([sit_down_strong(i) or sit_down_posture_low_motion(i) for i in range(n)]),
        sit_down_search_start,
        sitdown_need
    )
    cand_stop = sustained_first_true(stop_walking, sit_down_search_start, sitdown_need)

    candidates = [c for c in [cand_strong, cand_stop] if c is not None]
    sit_down_start = min(candidates) if candidates else None
    if sit_down_start is None:
        sit_down_start = max(sit_down_search_start, int(0.85 * n))

    # sitting_end_start: when stable sitting posture resumes
    sitting_end_start = sustained_first_true(np.array([is_sitting_soft(i) for i in range(n)]), sit_down_start, PERSIST["SITDOWN_TO_SITTING_END"])
    if sitting_end_start is None:
        sitting_end_start = min(n - 1, sit_down_start + int(0.4 * fps))
    min_sd = int(MIN_SITDOWN_SEC * fps)
    if sitting_end_start - sit_down_start < min_sd:
        # ensure SIT_DOWN duration is taken from SITTING_END by moving start earlier
        sit_down_start = max(walk_back_start + 1, sitting_end_start - min_sd)
        if sit_down_start < 0:
            sit_down_start = 0
        if sitting_end_start - sit_down_start < min_sd:
            sitting_end_start = min(n - 1, sit_down_start + min_sd)

    def clamp(a): return int(max(0, min(n - 1, a)))

    intervals = [
        PhaseInterval("SITTING_START", 0, clamp(stand_start - 1)),
        PhaseInterval("STAND_UP", clamp(stand_start), clamp(walk_start - 1)),
        PhaseInterval("WALK_FORWARD", clamp(walk_start), clamp(turn_start - 1)),
        PhaseInterval("TURN", clamp(turn_start), clamp(turn_end - 1)),
        PhaseInterval("WALK_BACK", clamp(walk_back_start), clamp(sit_down_start - 1)),
        PhaseInterval("SIT_DOWN", clamp(sit_down_start), clamp(sitting_end_start - 1)),
        PhaseInterval("SITTING_END", clamp(sitting_end_start), n - 1),
    ]
    intervals = [it for it in intervals if it.end_frame >= it.start_frame]

    labels = np.array([""] * n, dtype=object)
    for it in intervals:
        labels[it.start_frame:it.end_frame + 1] = it.phase

    return intervals, labels, debug


# ==========================================
# ANNOTATE + EXPORT
# ==========================================
def annotate_and_export(video_path: str,
                        labels: np.ndarray,
                        intervals,
                        fps, width, height,
                        df_debug: pd.DataFrame,
                        out_video: str,
                        out_excel: str):
    """Render labeled video and export phase timing table."""
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(labels) != n_frames:
        if len(labels) < n_frames:
            labels = np.concatenate([labels, np.array([labels[-1]] * (n_frames - len(labels)), dtype=object)])
        else:
            labels = labels[:n_frames]

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    last_phase = None
    transition_text = None
    transition_countdown = 0
    banner_frames = int(TRANSITION_BANNER_SEC * fps)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        phase = labels[frame_idx] if frame_idx < len(labels) else ""
        if last_phase is None:
            last_phase = phase
        if phase != last_phase:
            transition_text = f"PHASE CHANGE: {last_phase} -> {phase}"
            transition_countdown = banner_frames
            last_phase = phase

        q = float(df_debug["quality_s"].iloc[frame_idx]) if frame_idx < len(df_debug) else 0.0
        draw_label(frame, f"Phase: {phase}", (20, 45), scale=1.0)
        draw_label(frame, f"Time: {frame_idx / fps:.2f}s  Frame: {frame_idx}  Q:{q:.2f}", (20, 85), scale=0.85)

        if transition_countdown > 0 and transition_text:
            draw_transition(frame, transition_text, y=130)
            transition_countdown -= 1

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    rows = []
    for it in intervals:
        r = it.to_row(fps)
        seg = df_debug.iloc[it.start_frame:it.end_frame + 1]
        r["quality_mean"] = float(seg["quality_s"].mean())
        r["quality_min"] = float(seg["quality_s"].min())
        r["pose_ok_frac"] = float(np.mean(seg["pose_ok"].values > 0.5))
        rows.append(r)

    pd.DataFrame(rows).to_excel(out_excel, index=False)
    print(f"\n✅ Wrote annotated video: {out_video}")
    print(f"✅ Wrote phase times Excel: {out_excel}")


# ==========================================
# MAIN
# ==========================================
def main():
    video_path = select_video_file()

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_video = f"{base}_annotated.mp4"
    out_excel = f"{base}_phase_times.xlsx"
    out_debug = f"{base}_signals_debug.csv"

    df, fps, w, h = extract_signals(video_path)
    df.to_csv(out_debug, index=False)

    intervals, labels, debug = segment_phases(df, fps)

    print("\n=== Phase frames (start->end) ===")
    for it in intervals:
        print(f"{it.phase:13s}: {it.start_frame:6d} -> {it.end_frame:6d} "
              f"({(it.end_frame - it.start_frame)/fps:.2f}s)")

    print("\n=== Debug ===")
    for k, v in debug.items():
        print(f"{k}: {v}")

    annotate_and_export(video_path, labels, intervals, fps, w, h, df, out_video, out_excel)
    print(f"✅ Wrote debug signals CSV: {out_debug}")


if __name__ == "__main__":
    main()
