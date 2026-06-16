# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:10:00 2026

@author: nosha
"""

# mediapipe version: 0.10.35
# python version: 3.10.11

import os # standard library, no pip install needed
import math 
import urllib.request # standard library, no pip install needed
from dataclasses import dataclass, field # standard library, no pip install needed

import cv2 # pip install opencv-python, version = 4.13.0.92
import numpy as np # pip install numpy, version = 1.26.4
import pandas as pd # pip install pandas, version = 2.3.3
from scipy.signal import savgol_filter # pip install scipy, version = 1.15.3

import mediapipe as mp # pip install mediapipe, version = 0.10.35
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import tkinter as tk # standard library, no pip install needed
from tkinter import filedialog

from typing import List, Optional
import time
from enum import Enum

import matplotlib
matplotlib.use("Agg")  # plots directly into files rather than popping up a GUI window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# roll_correction.py from Mobile Motion Lab
import roll_correction as rc

# ==========================================
# CONFIG
# ==========================================
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
POSE_MODEL_PATH = "pose_landmarker_heavy.task"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)
FACE_MODEL_PATH = "face_landmarker.task"

def ensure_model(model_path, url):
    if not os.path.exists(model_path):
        if model_path == POSE_MODEL_PATH:
            print(f"Downloading Pose Landmarker model to: {model_path}")
            urllib.request.urlretrieve(url, model_path)
        elif model_path == FACE_MODEL_PATH:
            print(f"Downloading Face Landmarker model to: {model_path}")
            urllib.request.urlretrieve(url, model_path)
    return model_path


# ==========================================
# TIMING CONSTANTS
# ==========================================

COUNTDOWN_S      = 3.0                      # 3 second countdown before trial starts
TRIAL_DURATION_S = 20.0                     # trial is 20 seconds long
TRIAL_START_MS   = COUNTDOWN_S * 1000.0     # when the trial starts, in ms
TRIAL_END_MS     = TRIAL_START_MS + (TRIAL_DURATION_S * 1000.0)     # when the trial ends, in ms
 
# Calibrate from the last CALIB_WINDOW_S seconds *before* trial start, when the
# subject is most likely already in the reference pose (feet together, hands on hips).
CALIB_WINDOW_S = 1.0
CALIB_START_MS = TRIAL_START_MS - CALIB_WINDOW_S * 1000.0  # e.g. 2.0 s
CALIB_END_MS   = TRIAL_START_MS                            # e.g. 3.0 s
 
# An error must persist this long before it's committed (suppresses 1-frame flicker).
DEBOUNCE_S = 0.20 # 0.2 seconds
DEBOUNCE_MS = 200 # 200 ms


# ==========================================
# DETECTION THRESHOLDS
# ==========================================

ASPECT_RATIO_THRESHOLD  = 0.20      # eye AR below this means EYES OPEN
HANDS_THRESHOLD_MULT     = 1.5      # wrist-hip dist > (baseline * this) means HANDS OFF HIPS
STUMBLE_THRESHOLD        = 0.03     # per-frame ankle x jump (normalized)
SWAY_THRESHOLD           = 0.015    # per-frame mid-shoulder x jump (normalized)
HIP_ABDUCTION_THRESHOLD  = 30.0     # degrees from neutral (180) shoulder-hip-knee
FOOT_LIFT_THRESHOLD      = 0.02     # normalized rise of foot above baseline y


# ==========================================
# MEDIAPIPE CONSTANTS
# ==========================================

LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_INNER  = 133
LEFT_EYE_OUTER  = 33

RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_INNER  = 362
RIGHT_EYE_OUTER  = 263

LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

LEFT_WRIST  = 15
RIGHT_WRIST = 16

LEFT_HIP    = 23
RIGHT_HIP   = 24

LEFT_KNEE  = 25
RIGHT_KNEE = 26

LEFT_ANKLE  = 27
RIGHT_ANKLE = 28

LEFT_HEEL = 29
RIGHT_HEEL = 30

LEFT_FOOT_INDEX  = 31
RIGHT_FOOT_INDEX = 32


# ==========================================
# DATA MODELS
# ==========================================

""" There will be 5 possible errors, each corresponding to a state in the state machine:
"EYES_OPEN", "HANDS_OFF_HIPS", "STUMBLE", "HIP_ABDUCTION", "FOOT_LIFT"
"""
@dataclass
class TrialError:
    error_type: str        # error type
    timestamp: float       # the timestamp at which the error was detected
    duration: float = 0.0  # (seconds) how long the error lasted

@dataclass
class CalibrationData:
    left_wrist_hip_dist: float  = 0.0
    right_wrist_hip_dist: float = 0.0
    left_foot_y: float          = 0.0
    right_foot_y: float         = 0.0
    valid: bool                 = False

""" Contains data per frame, for saving in a csv """
@dataclass
class FrameData:
    frame_idx: int
    timestamp_ms: float
    in_trial: bool
    in_calib: bool
 
    face_detected: bool = False
    pose_detected: bool = False
 
    # average eye aspect ratio
    avg_ar: float = 0.0
 
    # distance from ipsilateral wrist and hip
    left_wrist_hip_dist:  float = 0.0
    right_wrist_hip_dist: float = 0.0
    
    # average x-coord of shoulders
    mid_shoulder_x:       float = 0.0
    
    # x and y-coords
    left_ankle_x:  float = 0.0
    right_ankle_x: float = 0.0
    left_foot_y:   float = 0.0
    right_foot_y:  float = 0.0
    
    # angle between ipsilateral shoulder, hip and knee
    left_hip_angle:  float = 180.0
    right_hip_angle: float = 180.0

""" 
DebounceState class: tracks whether an error is currently active, 
and only commits it once it's been sustained for DEBOUNCE_S.
All error detecting helper functions will use this.
"""
@dataclass
class DebounceState:
    active: bool        = False             # is the error condition currently detected? False by default.
    first_seen: float   = 0.0               # time.time() when condition first appeared
    committed: bool     = False             # has it been logged as an error yet?
    error_ref: Optional[TrialError] = None  # reference to the logged error. None by default


# ==========================================
# GEOMETRY HELPER FUNCTIONS
# ==========================================

""" Normalized distance between 2 landmarks. """
def normalized_distance(lm_a, lm_b) -> float:
    return math.sqrt((lm_a.x - lm_b.x)**2 + (lm_a.y - lm_b.y)**2)

""" Calculate aspect ratio given list of landmarks, 4 indexes, image width and height."""
def aspect_ratio(landmarks, top_idx, bottom_idx, inner_idx, outer_idx, img_w, img_h):
    top    = landmarks[top_idx]
    bottom = landmarks[bottom_idx]
    inner  = landmarks[inner_idx]
    outer  = landmarks[outer_idx]

    vertical   = abs(top.y - bottom.y) * img_h
    horizontal = abs(inner.x - outer.x) * img_w

    if horizontal == 0:
        return 0.0
    return vertical / horizontal

""" calculates the angle formed by 3 co-ordinates
    a,b,c: tuples """
def angle_3pt(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# ==========================================================================
# IMU ROLL CORRECTION
# ==========================================================================
"""
Use roll_correction.py to correct a video with its IMU data.
Returns the output path.
"""
def correct_video(video_path: str, imu_path: str) -> str:
    rc.IMU_PATH   = imu_path
    rc.VIDEO_PATH = video_path
    rc.main()
    
    if not os.path.exists(rc.OUT_VIDEO):
        raise RuntimeError(f"Roll correction did not produce: {rc.OUT_VIDEO}")
    
    print(f"Corrected video: {rc.OUT_VIDEO}")
    return rc.OUT_VIDEO


# ==========================================================================
# EXTRACT SIGNALS: run MediaPipe once, cache signals into FrameData array.
# ==========================================================================

"""
Run face + pose landmarkers over every frame,
caching the raw signals each detector will need. 
Returns (frame_data_list, fps).
"""
def extract_signals(video_path: str):
    cap0 = cv2.VideoCapture(video_path)
    if not cap0.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps   = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    img_w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap0.release()
 
    # Mediapipe Landmarkers
    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
 
    # initialize empty FrameData array
    frame_data_list: List[FrameData] = []
    frame_idx = 0
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
 
        timestamp_ms = (frame_idx / fps) * 1000.0
 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_int = int(timestamp_ms)
 
        face_res = face.detect_for_video(mp_img, ts_int)
        pose_res = pose.detect_for_video(mp_img, ts_int)
 
        in_trial = TRIAL_START_MS <= timestamp_ms <= TRIAL_END_MS
        in_calib = CALIB_START_MS <= timestamp_ms < CALIB_END_MS
        
        # initialize a new FrameData object
        fd = FrameData(frame_idx=frame_idx, timestamp_ms=timestamp_ms,
                       in_trial=in_trial, in_calib=in_calib)
 
        # fill in eye-related FrameData fields
        if face_res.face_landmarks:
            fd.face_detected = True
            lm = face_res.face_landmarks[0]
            l = aspect_ratio(lm, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_INNER, LEFT_EYE_OUTER, img_w, img_h)
            r = aspect_ratio(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, img_w, img_h)
            fd.avg_ar = (l + r) / 2
 
        # fill in pose-related FrameData fields
        if pose_res.pose_landmarks:
            fd.pose_detected = True
            lm = pose_res.pose_landmarks[0]
            fd.left_wrist_hip_dist  = normalized_distance(lm[LEFT_WRIST],  lm[LEFT_HIP])
            fd.right_wrist_hip_dist = normalized_distance(lm[RIGHT_WRIST], lm[RIGHT_HIP])
            fd.mid_shoulder_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2
            fd.left_ankle_x,  fd.right_ankle_x = lm[LEFT_ANKLE].x,  lm[RIGHT_ANKLE].x
            fd.left_foot_y,   fd.right_foot_y  = lm[LEFT_FOOT_INDEX].y, lm[RIGHT_FOOT_INDEX].y
            fd.left_hip_angle = angle_3pt(
                (lm[LEFT_SHOULDER].x, lm[LEFT_SHOULDER].y),
                (lm[LEFT_HIP].x,      lm[LEFT_HIP].y),
                (lm[LEFT_KNEE].x,     lm[LEFT_KNEE].y))
            fd.right_hip_angle = angle_3pt(
                (lm[RIGHT_SHOULDER].x, lm[RIGHT_SHOULDER].y),
                (lm[RIGHT_HIP].x,      lm[RIGHT_HIP].y),
                (lm[RIGHT_KNEE].x,     lm[RIGHT_KNEE].y))
 
        # append to FrameData array
        frame_data_list.append(fd)
        frame_idx += 1
        
        # update progress
        if frame_idx % 60 == 0:
            print(f"  extracted {frame_idx} frames")
 
    cap.release()
    face.close()
    pose.close()
    print(f"Extraction done: {len(frame_data_list)} frames.")
    return frame_data_list, fps


# ==========================================================================
# CALIBRATION
# ==========================================================================

"""
using the subarray of FrameData where in_calib == True,
calculate and return a CalibrationData object.
"""
def calibrate(frame_data_list: List[FrameData]) -> CalibrationData:
    # only where in_calib is True
    rows = [fd for fd in frame_data_list if fd.in_calib and fd.pose_detected]
    if not rows:
        print("WARNING: no pose frames in calibration window; "
              "hands/feet detection will be unreliable.")
        return CalibrationData(valid=False)
    
    c = CalibrationData(
        left_wrist_hip_dist  = float(np.mean([f.left_wrist_hip_dist  for f in rows])),
        right_wrist_hip_dist = float(np.mean([f.right_wrist_hip_dist for f in rows])),
        left_foot_y          = float(np.mean([f.left_foot_y  for f in rows])),
        right_foot_y         = float(np.mean([f.right_foot_y for f in rows])),
        valid = True,
    )
    
    print(f"Calibrated from {len(rows)} frames: "
          f"wrist-hip L={c.left_wrist_hip_dist:.3f} R={c.right_wrist_hip_dist:.3f}, "
          f"foot-y L={c.left_foot_y:.3f} R={c.right_foot_y:.3f}")
    
    return c


# ==========================================================================
# DETECT: per-frame detection over cached signal array
# ==========================================================================

""" 
Return {error_type: bool} for this frame. This happens before debouncing.
"""
def detect_per_frame(fd: FrameData, prev: Optional[FrameData], calib: CalibrationData) -> dict:
    # {error_type : bool} dictionary
    flags = {"EYES_OPEN": False, "HANDS_OFF_HIPS": False,
             "STUMBLE_SWAY": False, "HIP_ABDUCTION": False, "FOOT_LIFT": False}
 
    
    if fd.face_detected:
        flags["EYES_OPEN"] = fd.avg_ar < ASPECT_RATIO_THRESHOLD
 
    if fd.pose_detected:
        if calib.valid:
            flags["HANDS_OFF_HIPS"] = (
                fd.left_wrist_hip_dist  > calib.left_wrist_hip_dist  * HANDS_THRESHOLD_MULT or
                fd.right_wrist_hip_dist > calib.right_wrist_hip_dist * HANDS_THRESHOLD_MULT
                )
            
            flags["FOOT_LIFT"] = (
                (calib.left_foot_y  - fd.left_foot_y)  > FOOT_LIFT_THRESHOLD or
                (calib.right_foot_y - fd.right_foot_y) > FOOT_LIFT_THRESHOLD
                )
 
        flags["HIP_ABDUCTION"] = (
            abs(180.0 - fd.left_hip_angle)  > HIP_ABDUCTION_THRESHOLD or
            abs(180.0 - fd.right_hip_angle) > HIP_ABDUCTION_THRESHOLD
            )
 
        if prev is not None and prev.pose_detected:
            stumble = (abs(fd.left_ankle_x  - prev.left_ankle_x)  > STUMBLE_THRESHOLD or
                       abs(fd.right_ankle_x - prev.right_ankle_x) > STUMBLE_THRESHOLD)
            sway = abs(fd.mid_shoulder_x - prev.mid_shoulder_x) > SWAY_THRESHOLD
            flags["STUMBLE_SWAY"] = stumble or sway
 
    return flags

"""
Iterate over the cached trial frames, apply per-frame detection + debounce.
Returns (errors, per_frame_flags) where per_frame_flags aligns with the
trial frames for the signal CSV.
"""
def run_detection(frame_data_list: List[FrameData], calib: CalibrationData):
   
    # empty TrialError array
    errors: List[TrialError] = []
    
    # create a dictionatry where:
    # keys: error types
    # values: default DeboounceState objects.
    states = {k: DebounceState() for k in
              ["EYES_OPEN", "HANDS_OFF_HIPS", "STUMBLE_SWAY", "HIP_ABDUCTION", "FOOT_LIFT"]}
 
    # initialize empty per_frame_flag array.
    # each item will be a 3-tuple of (FrameData, dictionary of {error type : bool}, dictionary of {error type : bool})
    # WHAT TYPE IS PER_FRAME FLAGS????? ARRAY OF 3-TUPLES!!!!!
    per_frame_flags = []
    prev = None
 
    # only the frames where in_trial == True
    trial_frames = [fd for fd in frame_data_list if fd.in_trial]
    
    for fd in trial_frames:
        # relative time
        t_rel = (fd.timestamp_ms - TRIAL_START_MS) / 1000.0  # seconds into trial
        
        # detect the errors in that framewithout debounce
        raw = detect_per_frame(fd, prev, calib)
 
        # empty dictionry for errors committed now
        committed_now = {}
        
        # iterate over (error type : bool) pairs in the raw detected errors in that frame
        for etype, cond in raw.items():
            # access the value (DebounceState) of that error type 
            # in the states dictionary initialized above
            st = states[etype]
            
            # bool was True
            if cond:
                # if the error is not active, become active
                if not st.active:
                    st.active = True
                    st.first_seen = t_rel 
                    st.committed = False
                    st.error_ref = None
                    
                elif not st.committed and (t_rel - st.first_seen) >= DEBOUNCE_S:
                    # add TrialError to errors array
                    err = TrialError(error_type=etype, timestamp=st.first_seen)
                    errors.append(err)
                    
                    # if the error is not committed, become committed
                    st.committed = True
                    # add reference to TrialError to the DebounceState field.
                    st.error_ref = err
            # bool was False
            else:
                # if this marks the end of a pervious error
                if st.committed and st.error_ref is not None:
                    st.error_ref.duration = t_rel - st.error_ref.timestamp
                # reset fields to no error currently
                st.active, st.committed, st.error_ref = False, False, None
            
            # add key value pair to committed_now dictionary
            committed_now[etype] = st.committed

        # ensuring that committed_now is of type dictionary
        # WHAT TYPE IS PER_FRAME FLAGS????? ARRAY OF 3-TUPLES!!!!!
        per_frame_flags.append((fd, raw, dict(committed_now)))
        prev = fd
 
    # close out any errors still active at trial end
    last_t = (trial_frames[-1].timestamp_ms - TRIAL_START_MS) / 1000.0 if trial_frames else 0.0
    for st in states.values():
        if st.committed and st.error_ref is not None and st.error_ref.duration == 0.0:
            st.error_ref.duration = last_t - st.error_ref.timestamp

    return errors, per_frame_flags


# ==========================================================================
# CSV EXPORT
# ==========================================================================

"""
Export the signals collected in the (fd, raw, committed) 3-tuple of per_frame_flags
to a .csv file.
"""
def export_signals_csv(per_frame_flags, path: str):
    rows = []
    # remember that raw and committed are both {error_type (str) : (bool)} dictionary types
    for fd, raw, committed in per_frame_flags:
        t_rel = (fd.timestamp_ms - TRIAL_START_MS) / 1000.0
        
        # each row contains these columns:
        rows.append({
            "t_trial_s": round(t_rel, 3),
            "face_detected": fd.face_detected,
            "pose_detected": fd.pose_detected,
            "avg_ar": round(fd.avg_ar, 4),
            "l_wrist_hip": round(fd.left_wrist_hip_dist, 4),
            "r_wrist_hip": round(fd.right_wrist_hip_dist, 4),
            "mid_shoulder_x": round(fd.mid_shoulder_x, 4),
            "l_ankle_x": round(fd.left_ankle_x, 4),
            "r_ankle_x": round(fd.right_ankle_x, 4),
            "l_foot_y": round(fd.left_foot_y, 4),
            "r_foot_y": round(fd.right_foot_y, 4),
            "l_hip_angle": round(fd.left_hip_angle, 2),
            "r_hip_angle": round(fd.right_hip_angle, 2),
            "raw_EYES_OPEN": raw["EYES_OPEN"],
            "raw_HANDS_OFF_HIPS": raw["HANDS_OFF_HIPS"],
            "raw_STUMBLE_SWAY": raw["STUMBLE_SWAY"],
            "raw_HIP_ABDUCTION": raw["HIP_ABDUCTION"],
            "raw_FOOT_LIFT": raw["FOOT_LIFT"],
        })
        
    # pd.DataFrame != FrameData
    # export all the rows in a .csv file.
    pd.DataFrame(rows).to_csv(path, index=False)
    
    print(f"Signals saved to {path}. ({len(rows)} rows).")

"""
Export the errors from a TrialError array to a .csv file.
""" 
def export_errors_csv(errors: List[TrialError], stance: str, surface: str, path: str):
    # each row will contain these columns:
    if errors:
        rows = [{"stance": stance, 
                 "surface": surface, 
                 "error_type": e.error_type,
                 "start_s": round(e.timestamp, 3), 
                 "duration_s": round(e.duration, 3) 
                } for e in errors]
    # no errors committed; errors array empty
    else:
        rows = [{"stance": stance, 
                 "surface": surface, 
                 "error_type": "NONE",
                 "start_s": 0, 
                 "duration_s": 0
                }]
    
    # export all the rows in a .csv file.
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Errors saved to {path}. ({len(errors)} errors).")
    

# =============================================
# FILE SELECTION + ANALYSIS WRAPPER
# =============================================

"""
Prompt user to select video and IMU data from file browser.
Returns the paths.
"""
def select_video_and_imu():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select BESS Video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    if not video_path:
        raise RuntimeError("No video selected.")
    imu_path = filedialog.askopenfilename(
        title="Select IMU Data File",
        filetypes=[("Text/JSON", "*.txt *.json"), ("All files", "*.*")])
    if not imu_path:
        raise RuntimeError("No IMU file selected (required for roll correction).")
    return video_path, imu_path
 
"""
Wrapper function to call all processing and detection functions.
Returns the TrialError array.
"""
def analyze(video_path: str, imu_path: str,
            stance: str = "DOUBLE_LEG", surface: str = "FIRM"):
    print("Step 0: IMU roll-correcting video...")
    corrected_path = correct_video(video_path, imu_path)
 
    print("Step 1: extracting signals (this runs MediaPipe once)...")
    frame_data_list, fps = extract_signals(corrected_path)
 
    calib = calibrate(frame_data_list)
 
    print("Step 2: detecting errors...")
    errors, per_frame_flags = run_detection(frame_data_list, calib)
 
    # Write CSVs next to the ORIGINAL video, not the corrected one.
    base = os.path.splitext(video_path)[0]
    export_signals_csv(per_frame_flags, base + "_signals.csv")
    export_errors_csv(errors, stance, surface, base + "_errors.csv")
 
    counts = {}
    for e in errors:
        counts[e.error_type] = counts.get(e.error_type, 0) + 1
    print("Summary:", counts if counts else "no errors")
    return errors
 
"""
Main wrapper function.
Calls video selection and analysis functions.
"""
if __name__ == "__main__":
    # video path, IMU path
    vp, ip = select_video_and_imu()
    analyze(vp, ip)