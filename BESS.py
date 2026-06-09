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

SCALE_FACTOR = 1.75  # inference scale-up for higher landmark precision



                                      ########################### HELPER FUNCTIONS ###########################


""" These connection pairs are hardcoded since POSE_CONNECTIONS is gone from Mediapipe 0.10.35"""
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]

def draw_landmarks_and_connections(image, results):
    if not results.pose_landmarks:
        return

    h, w, _ = image.shape

    for pose_landmarks in results.pose_landmarks:
        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            start = pose_landmarks[start_idx]
            end = pose_landmarks[end_idx]

            if start.visibility > 0.5 and end.visibility > 0.5:
                cv2.line(image,
                    (int(start.x * w), int(start.y * h)),
                    (int(end.x * w), int(end.y * h)),
                    (255, 0, 255), 2)

        # Draw landmark dots
        for landmark in pose_landmarks:
            try:
                if landmark.visibility > 0.5:
                    cv2.circle(image,
                        (int(landmark.x * w), int(landmark.y * h)),
                        4, (255, 255, 0), -1)
            except:
                pass
    return

""" There will be 5 possible errors, each corresponding to a state in the state machine:
"EYES_OPEN", "HANDS_OFF_HIPS", "STUMBLE", "HIP_ABDUCTION", "FOOT_LIFT"
"""
@dataclass
class TrialError:
    error_type: str        # name of error type
    timestamp: float       # time.time() when error was first detected
    duration: float = 0.0  # how long the error lasted (filled in when error ends)

@dataclass
class CalibrationData:
    left_wrist_hip_dist: float  = 0.0  # baseline normalized distance, left wrist to left hip
    right_wrist_hip_dist: float = 0.0  # right wrist to right hip
    threshold_multiplier: float = 1.5  # "HANDS_OFF_HIPS" fires at baseline * this

"""
TrialResult class: describes the nature of the trial and the results of the trial. 
contains attributes: stance, surface, duration in seconds, list of errors, start time, end time.
"""
@dataclass
class TrialResult:
    stance: str                         # double leg, single leg, or tandem
    surface: str                        # firm or foam
    duration_s: float       = 20.0      # each trial is 20 seconds long by default
    errors: List[TrialError]= field(default_factory=list)       # the list of errors committed.
    start_time: float       = 0.0       # will change
    end_time: float         = 0.0       # will changw

    @property
    def error_count(self):              # a method to return the number of errors for this trial result
        return len(self.errors)
    
""" 
DebounceState class: tracks whether an error is currently active, and only commits it once it's been sustained for 200ms.
All error detecting helper functions will use this.
"""
@dataclass
class DebounceState:
    active: bool        = False             # is the error condition currently detected? False by default.
    first_seen: float   = 0.0               # time.time() when condition first appeared
    committed: bool     = False             # has it been logged as an error yet?
    error_ref: Optional[TrialError] = None  # reference to the logged error. None by default

""" global debounce duration """
DEBOUNCE_MS = 0.2  # 200ms by default

""" a method to update an instance of the debounce class """
def update_debounce(state: DebounceState, condition: bool, error_type: str, trial: TrialResult) -> DebounceState:
    now = time.time()

    if condition:
        # condition just appeared
        if not state.active:
            state.active     = True
            state.first_seen = now
            state.committed  = False
            state.error_ref  = None
            
        # sustained long enough-> log it.
        elif not state.committed and (now - state.first_seen) >= DEBOUNCE_MS:
            err = TrialError(error_type=error_type, timestamp=state.first_seen)
            trial.errors.append(err)
            state.committed = True
            state.error_ref = err
        
    else:
        # condition ended-> fill in duration.
        if state.committed and state.error_ref is not None:
            state.error_ref.duration = now - state.error_ref.timestamp
        
        # reset
        state.active    = False
        state.committed = False
        state.error_ref = None

    return state


                                        ############################# EYES OPEN #############################
                                        
                                        
""" landmark indices for MediaPipe FaceLandmarker (478-point model) """
# Left eye
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_INNER  = 133
LEFT_EYE_OUTER  = 33

# Right eye
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_INNER  = 362
RIGHT_EYE_OUTER  = 263

""" global eye aspect ratio threshold """
ASPECT_RATIO_THRESHOLD = 0.2  # calibrate with testing

""" Calculates the Eye Aspect Ratio """
def calculate_aspect_ratio(landmarks, top_idx, bottom_idx, inner_idx, outer_idx, img_w, img_h):
    top    = landmarks[top_idx]
    bottom = landmarks[bottom_idx]
    inner  = landmarks[inner_idx]
    outer  = landmarks[outer_idx]

    vertical   = abs(top.y - bottom.y) * img_h
    horizontal = abs(inner.x - outer.x) * img_w

    if horizontal == 0:
        return 0.0
    return vertical / horizontal

""" Detection for eye opening error """
def detect_eye_error(face_landmarks, img_w, img_h) -> bool:
    left_ar  = calculate_aspect_ratio(face_landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_INNER, LEFT_EYE_OUTER, img_w, img_h)
    right_ar = calculate_aspect_ratio(face_landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, img_w, img_h)
    avg_ar   = (left_ar + right_ar) / 2
    
    # are the eyes open? boolean
    eyes_open = avg_ar < ASPECT_RATIO_THRESHOLD
    return eyes_open, avg_ar



                                    ############################# HANDS OFF HIPS #############################
                                    
                                    
""" landmark indices for MediaPipe PoseLandmarker (33-point model) """
LEFT_WRIST  = 15
RIGHT_WRIST = 16
LEFT_HIP    = 23
RIGHT_HIP   = 24

""" get the normalized distance between 2 landmarks """
def get_normalized_distance(lm_a, lm_b) -> float:
    return math.sqrt((lm_a.x - lm_b.x)**2 + (lm_a.y - lm_b.y)**2)

""" calibrate each person's hands on hips distance, and store in a CalibrationData class. """
def calibrate_hands_on_hips(pose_landmarks) -> CalibrationData:
    lm = pose_landmarks[0]
    left_dist  = get_normalized_distance(lm[LEFT_WRIST],  lm[LEFT_HIP])
    right_dist = get_normalized_distance(lm[RIGHT_WRIST], lm[RIGHT_HIP])
    return CalibrationData(
        left_wrist_hip_dist  = left_dist,
        right_wrist_hip_dist = right_dist
    )
    
""" Detection for hands off hips error """
def detect_hands_off_hips(pose_landmarks, calib: CalibrationData) -> bool:
    lm = pose_landmarks[0]
    left_dist  = get_normalized_distance(lm[LEFT_WRIST],  lm[LEFT_HIP])
    right_dist = get_normalized_distance(lm[RIGHT_WRIST], lm[RIGHT_HIP])

    left_off  = left_dist  > calib.left_wrist_hip_dist  * calib.threshold_multiplier
    right_off = right_dist > calib.right_wrist_hip_dist * calib.threshold_multiplier
    return left_off or right_off



                                    ############################# STUMBLE / SWAY #############################


""" landmark indices for MediaPipe PoseLandmarker (33-point model) """
LEFT_ANKLE  = 27
RIGHT_ANKLE = 28
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

""" global stumble and sway thresholds """
STUMBLE_THRESHOLD = 0.03   # normalized units; tune with testing
SWAY_THRESHOLD    = 0.015  # subtle sway is a smaller threshold

""" a class that keeps a rolling list of the last 5 x-coordinates for: 
left ankle, right ankle, and midpoint between shoulders (center-of-mass proxy).
the idea is if they move too much between frames, flag it as an error.
"""
@dataclass
class PoseHistory:
    left_ankle_x:   List[float] = field(default_factory=list)
    right_ankle_x:  List[float] = field(default_factory=list)
    mid_shoulder_x: List[float] = field(default_factory=list)
    max_history: int = 5  # frames to keep

""" update the last 5 to include the current (kick out oldest) """
def update_pose_history(history: PoseHistory, pose_landmarks) -> PoseHistory:
    lm = pose_landmarks[0]
    mid_shoulder_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2

    history.left_ankle_x.append(lm[LEFT_ANKLE].x)
    history.right_ankle_x.append(lm[RIGHT_ANKLE].x)
    history.mid_shoulder_x.append(mid_shoulder_x)

    # keep only recent frames
    history.left_ankle_x   = history.left_ankle_x[-history.max_history:]
    history.right_ankle_x  = history.right_ankle_x[-history.max_history:]
    history.mid_shoulder_x = history.mid_shoulder_x[-history.max_history:]

    return history

""" detecting stumble with PoseHistory of the ankles """
def detect_stumble(history: PoseHistory) -> bool:
    if len(history.left_ankle_x) < 2:
        return False
    left_delta  = abs(history.left_ankle_x[-1]  - history.left_ankle_x[-2])
    right_delta = abs(history.right_ankle_x[-1] - history.right_ankle_x[-2])
    return left_delta > STUMBLE_THRESHOLD or right_delta > STUMBLE_THRESHOLD

""" detecting stumble with PoseHistory of the mid-shoulder """
def detect_sway(history: PoseHistory) -> bool:
    if len(history.mid_shoulder_x) < 2:
        return False
    delta = abs(history.mid_shoulder_x[-1] - history.mid_shoulder_x[-2])
    return delta > SWAY_THRESHOLD



                                    ############################# HIP ABDUCTION #############################


""" landmark indices for MediaPipe PoseLandmarker (33-point model) """
LEFT_KNEE  = 25
RIGHT_KNEE = 26

""" global hip abduction threshold """
HIP_ABDUCTION_THRESHOLD = 30.0  # degrees

""" calculates the angle formed by 3 co-ordinates
    a,b,c: tuples """
def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

""" Detection for hip abduction error """
def detect_hip_abduction(pose_landmarks) -> bool:
    lm = pose_landmarks[0]

    # angle between left shoulder, hip and knee
    left_angle = calculate_angle(
        (lm[LEFT_SHOULDER].x,  lm[LEFT_SHOULDER].y),
        (lm[LEFT_HIP].x,       lm[LEFT_HIP].y),
        (lm[LEFT_KNEE].x,      lm[LEFT_KNEE].y)
    )
    
    # angle between right shoulder, hip and knee
    right_angle = calculate_angle(
        (lm[RIGHT_SHOULDER].x, lm[RIGHT_SHOULDER].y),
        (lm[RIGHT_HIP].x,      lm[RIGHT_HIP].y),
        (lm[RIGHT_KNEE].x,     lm[RIGHT_KNEE].y)
    )

    # in a neutral stance, this angle is 180. 
    # abduction --> pulls it away from 180
    left_abducted  = abs(180.0 - left_angle)  > HIP_ABDUCTION_THRESHOLD
    right_abducted = abs(180.0 - right_angle) > HIP_ABDUCTION_THRESHOLD
    return left_abducted or right_abducted



                                    ############################# FOOT LIFT #############################


""" landmark indices for MediaPipe PoseLandmarker (33-point model) """
LEFT_FOOT_INDEX  = 31
RIGHT_FOOT_INDEX = 32

""" global hip abduction threshold """
FOOT_LIFT_THRESHOLD = 0.02  # tune with testing


""" class for the baseline foot lifting """
@dataclass
class FootBaseline:
    left_y:  float = 0.0
    right_y: float = 0.0
    calibrated: bool = False


""" method to calibrate the person's foot baseline """
def calibrate_foot_baseline(pose_landmarks) -> FootBaseline:
    lm = pose_landmarks[0]
    return FootBaseline(
        left_y  = lm[LEFT_FOOT_INDEX].y,
        right_y = lm[RIGHT_FOOT_INDEX].y,
        calibrated = True
    )

""" Detection for foot lift error. """
def detect_foot_lift(pose_landmarks, baseline: FootBaseline) -> bool:
    if not baseline.calibrated:
        return False
    lm = pose_landmarks[0]
    # y decreases as foot rises (normalized coords, 0=top of frame)
    left_lift  = (baseline.left_y  - lm[LEFT_FOOT_INDEX].y)  > FOOT_LIFT_THRESHOLD
    right_lift = (baseline.right_y - lm[RIGHT_FOOT_INDEX].y) > FOOT_LIFT_THRESHOLD
    return left_lift or right_lift

######## End of helper functions. ########

# ==========================================
# STATE MACHINE
# ==========================================

class TrialState(Enum):
    IDLE        = "IDLE"
    CALIBRATING = "CALIBRATING"
    COUNTDOWN   = "COUNTDOWN"
    RUNNING     = "RUNNING"
    COMPLETE    = "COMPLETE"

# ==========================================
# HUD
# ==========================================

def draw_hud(image, state: TrialState, trial: TrialResult, avg_ar: float,
             countdown_remaining: float, trial_remaining: float,
             active_errors: dict):

    h, w, _ = image.shape

    # --- state label (top center) ---
    cv2.putText(image, state.value, (w//2 - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    # --- aspect ratio (top left) ---
    cv2.putText(image, f"Avg AR: {avg_ar:.3f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- countdown or trial timer ---
    if state == TrialState.COUNTDOWN:
        cv2.putText(image, f"Starting in: {countdown_remaining:.1f}s", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if state == TrialState.RUNNING:
        cv2.putText(image, f"Time left: {trial_remaining:.1f}s", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(image, f"Errors: {trial.error_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # active error warnings
        y_offset = 160
        for error_type, is_active in active_errors.items():
            if is_active:
                cv2.putText(image, f"!! {error_type}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 35

    # --- final summary ---
    if state == TrialState.COMPLETE:
        cv2.putText(image, "TRIAL COMPLETE", (w//2 - 150, h//2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(image, f"Total Errors: {trial.error_count}", (w//2 - 120, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(image, "Press SPACE to save and exit", (w//2 - 180, h//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # list each error type and count
        counts = {}
        for e in trial.errors:
            counts[e.error_type] = counts.get(e.error_type, 0) + 1
        y_offset = h//2 + 100
        for error_type, count in counts.items():
            cv2.putText(image, f"  {error_type}: {count}", (w//2 - 120, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30

# ==========================================
# CSV EXPORT
# ==========================================

def save_results(trial: TrialResult, path: str = "bess_results.csv"):
    rows = []
    for e in trial.errors:
        rows.append({
            "stance":     trial.stance,
            "surface":    trial.surface,
            "error_type": e.error_type,
            "timestamp":  round(e.timestamp - trial.start_time, 3),  # relative to trial start
            "duration_s": round(e.duration, 3)
        })
    # if no errors, still write a row so the trial is recorded
    if not rows:
        rows.append({
            "stance":     trial.stance,
            "surface":    trial.surface,
            "error_type": "NONE",
            "timestamp":  0,
            "duration_s": 0
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")

# ==========================================
# TRIAL WRAPPER
# ==========================================

""" global durations """
COUNTDOWN_DURATION = 5.0   # seconds
TRIAL_DURATION     = 20.0  # seconds
CALIBRATION_FRAMES = 180    # ~6 seconds at 30fps

""" Run the trial. """
def run_trial():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- build face landmarker ---
    face_model_path = ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=face_model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

    # --- build pose landmarker ---
    pose_model_path = ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=pose_model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    # --- trial data, hard coded for now. ---
    trial = TrialResult(stance="DOUBLE_LEG", surface="FIRM")

    # --- debounce states, one per error type ---
    db_eyes      = DebounceState()
    db_hands     = DebounceState()
    db_stumble   = DebounceState()
    db_abduction = DebounceState()
    db_foot      = DebounceState()

    # --- pose tracking ---
    pose_history  = PoseHistory()
    calib_data    = CalibrationData()
    foot_baseline = FootBaseline()
    calib_frame_count = 0

    # --- state machine ---
    state             = TrialState.IDLE # start at idle
    countdown_start   = None
    trial_start       = None
    start_time        = time.time()
    avg_ar            = 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp_ms = int((time.time() - start_time) * 1000)

        face_results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face_detected = bool(face_results.face_landmarks)
        pose_detected = bool(pose_results.pose_landmarks)

        now = time.time()
        countdown_remaining = 0.0
        trial_remaining     = 0.0
        active_errors = {
            "EYES_OPEN":   False,
            "HANDS_OFF_HIPS": False,
            "STUMBLE_SWAY":  False,
            "HIP_ABDUCTION": False,
            "FOOT_LIFT":     False,
        }

        # ==========================================
        # STATE TRANSITIONS
        # ==========================================

        ## 1. IDLE. until face is detected. most likely immediately.
        if state == TrialState.IDLE:
            ## START CALIBRATING AS SOON AS FACE IS DETECTED
            if face_detected:
                state = TrialState.CALIBRATING
                calib_frame_count = 0
                
        ## 2. CALBIRATING. 3 seconds.
        elif state == TrialState.CALIBRATING:
            if pose_detected:
                calib_frame_count += 1
                # accumulate calibration over CALIBRATION_FRAMES, average at the end
                lm = pose_results.pose_landmarks
                left_dist  = get_normalized_distance(lm[0][LEFT_WRIST], lm[0][LEFT_HIP])
                right_dist = get_normalized_distance(lm[0][RIGHT_WRIST], lm[0][RIGHT_HIP])
                calib_data.left_wrist_hip_dist  = (calib_data.left_wrist_hip_dist  * (calib_frame_count - 1) + left_dist)  / calib_frame_count
                calib_data.right_wrist_hip_dist = (calib_data.right_wrist_hip_dist * (calib_frame_count - 1) + right_dist) / calib_frame_count
                foot_baseline = calibrate_foot_baseline(lm)

            if calib_frame_count >= CALIBRATION_FRAMES:
                state = TrialState.COUNTDOWN
                countdown_start = now

        ## 3. COUNTDOWN to start the trial. 5 seconds
        elif state == TrialState.COUNTDOWN:
            countdown_remaining = max(0.0, COUNTDOWN_DURATION - (now - countdown_start))
            if countdown_remaining <= 0.0:
                state = TrialState.RUNNING
                trial.start_time = now
                trial_start = now

        ## 4. RUNNING the trial. 20 seconds
        elif state == TrialState.RUNNING:
            trial_remaining = max(0.0, TRIAL_DURATION - (now - trial_start))

            # --- run all error detectors ---
            if face_detected:
                eyes_open, avg_ar = detect_eye_error(face_results.face_landmarks[0], img_w, img_h)
                db_eyes = update_debounce(db_eyes, eyes_open, "EYES_OPEN", trial)
                active_errors["EYES_OPEN"] = db_eyes.active


            if pose_detected:
                pose_history = update_pose_history(pose_history, pose_results.pose_landmarks)

                hands_off = detect_hands_off_hips(pose_results.pose_landmarks, calib_data)
                db_hands  = update_debounce(db_hands, hands_off, "HANDS_OFF_HIPS", trial)
                active_errors["HANDS_OFF_HIPS"] = db_hands.active

                stumble = detect_stumble(pose_history) or detect_sway(pose_history)
                db_stumble = update_debounce(db_stumble, stumble, "STUMBLE_SWAY", trial)
                active_errors["STUMBLE_SWAY"] = db_stumble.active

                abduction = detect_hip_abduction(pose_results.pose_landmarks)
                db_abduction = update_debounce(db_abduction, abduction, "HIP_ABDUCTION", trial)
                active_errors["HIP_ABDUCTION"] = db_abduction.active

                foot = detect_foot_lift(pose_results.pose_landmarks, foot_baseline)
                db_foot = update_debounce(db_foot, foot, "FOOT_LIFT", trial)
                active_errors["FOOT_LIFT"] = db_foot.active

                draw_landmarks_and_connections(image, pose_results)

            if trial_remaining <= 0.0:
                trial.end_time = now
                state = TrialState.COMPLETE

        elif state == TrialState.COMPLETE:
            pass  # just display HUD until spacebar

        # ==========================================
        # HUD + DISPLAY
        # ==========================================
        draw_hud(image, state, trial, avg_ar, countdown_remaining, trial_remaining, active_errors)

        if state == TrialState.CALIBRATING:
            cv2.putText(image, f"Calibrating... {calib_frame_count}/{CALIBRATION_FRAMES}",
                        (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if state == TrialState.IDLE:
            cv2.putText(image, "Stand in front of camera to begin",
                        (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if state == TrialState.COUNTDOWN:
            cv2.putText(image, "Get into position: feet together, hands on hips",
                        (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("BESS Trial", image)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and state == TrialState.COMPLETE:
            save_results(trial)
            break

    cap.release()
    cv2.destroyAllWindows()
    return trial

def select_video_and_imu():
    root = tk.Tk()
    root.withdraw()

    video_path = filedialog.askopenfilename(
        title="Select BESS Video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if not video_path:
        raise RuntimeError("No video selected.")

    imu_path = filedialog.askopenfilename(
        title="Select IMU Data File",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not imu_path:
        print("Warning: No IMU file selected. Stabilization will be skipped.")
        imu_path = None

    return video_path, imu_path

# ==========================================
# VIDEO SELECTION
# ==========================================

def select_video_and_imu():
    root = tk.Tk()
    root.withdraw()

    video_path = filedialog.askopenfilename(
        title="Select BESS Video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if not video_path:
        raise RuntimeError("No video selected.")

    imu_path = filedialog.askopenfilename(
        title="Select IMU Data File",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not imu_path:
        print("Warning: No IMU file selected. Stabilization will be skipped.")
        imu_path = None

    return video_path, imu_path


# ==========================================
# IMU pROCESSING
# ==========================================
import json

"""
return an array of timestamps and gyro Z values from the imu data
"""
def load_imu(imu_path: str):
    with open(imu_path, "r") as f:
        raw = json.load(f)
    timestamps = np.array([int(x) for x in raw.keys()])
    gyro_z    = np.array([raw[k]["gyroZ"] for k in raw.keys()])
    return timestamps, gyro_z

""" 
helper function to return the interpolated gyro z value given a timestamp
"""
def interpolate_gyro(timestamps, gyro_z, query_ms: float) -> float:
    return float(np.interp(query_ms, timestamps, gyro_z))

"""
integrate gyro z over time to get cumulative rotation angle (radians)
at each IMU timestamp. Returns a dictionary where:
- keys are timestamp in ms
- values are angle in radians.
"""
def build_rotation_angle(timestamps, gyro_z) -> dict:
    
    angles = {timestamps[0]: 0.0}
    cumulative = 0.0
    for i in range(1, len(timestamps)):
        dt = (timestamps[i] - timestamps[i-1]) / 1000.0  # ms to seconds
        cumulative += gyro_z[i] * dt
        angles[timestamps[i]] = cumulative
    return angles

""" Helper function to interpolate cumulative rotation angle 
at a given timestamp in ms.
"""
def get_angle_at(angles: dict, timestamps, query_ms: float) -> float:
    ts_arr    = np.array(list(angles.keys()))
    angle_arr = np.array(list(angles.values()))
    return float(np.interp(query_ms, ts_arr, angle_arr))


"""
Function to apply counter-rotation, to correct for camera tilt.
angle_rad: cumulative gyroZ rotation at this frame's timestamp.
"""
def stabilize_frame(frame, angle_rad: float):
    
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    # counter-rotate by negative of accumulated angle
    deg = -np.degrees(angle_rad)
    M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    return cv2.warpAffine(frame, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ==========================================
# NEW ARCHITECTURE
# ==========================================

TRIAL_START_MS  = 1000.0   # let's say the trial starts 1 second after recording. THIS IS A TEST VALUE!!!
TRIAL_END_MS    = TRIAL_START_MS + 20000.0

@dataclass
class FrameData:
    timestamp_ms:   float
    in_trial:       bool
    avg_ar:         float = 0.0
    shoulder_mid_x: float = 0.0
    hip_mid_x:      float = 0.0
    left_ankle_y:   float = 0.0
    right_ankle_y:  float = 0.0
    face_detected:  bool  = False
    pose_detected:  bool  = False


"""
We're going to run MediaPipe on every frame, buffer FrameData.
Returns (frames_bgr, frame_data_list).
"""
def process_video(video_path: str, imu_path: str) -> tuple:
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # flexible dimensions
    img_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # print the video information
    #print(f"Video: {img_w}x{img_h} @ {fps:.1f}fps, {total} frames")

    # load the IMU data
    imu_loaded = False
    if imu_path:
        try:
            imu_timestamps, gyro_z = load_imu(imu_path)
            rotation_angles = build_rotation_angle(imu_timestamps, gyro_z)
            imu_loaded = True
            print("IMU loaded successfully.")
        except Exception as e:
            print(f"IMU load failed: {e}. Stabilization skipped.")

    # load the mediapipe models
    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    )

    frames_bgr     = [] # array of BGR (cv2 format) frames
   