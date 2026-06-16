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