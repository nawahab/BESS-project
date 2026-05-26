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
from dataclasses import dataclass # standard library, no pip install needed

import cv2 # pip install opencv-python, version = 4.13.0.92
import numpy as np # pip install numpy, version = 1.26.4
import pandas as pd # pip install pandas, version = 2.3.3
from scipy.signal import savgol_filter # pip install scipy, version = 1.15.3

import mediapipe as mp # pip install mediapipe, version = 0.10.35
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import tkinter as tk # standard library, no pip install needed
from tkinter import filedialog

import time

# ==========================================
# CONFIG
# ==========================================
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
MODEL_PATH = "pose_landmarker_heavy.task"

SCALE_FACTOR = 1.75  # inference scale-up for higher landmark precision

## HELPER FUNCTIONS ##
""" Calculate angle between co-ordinates a, b and c. """
def calculate_angle(a, b, c):
    a = np.array(a) # first
    b = np.array(b) # mid
    c = np.array(c) # end

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0 / np.pi)
    if angle > 180.0:
        angle = 360-angle
    
    return angle

""" Calculate the ratio between the c-a x and y """
def calculate_ratio(a, b, c):
    a = np.array(a) # first
    b = np.array(b) # mid
    c = np.array(c) # end

    # ratio will be the difference between first and end y / difference between first and end x
    ratio = (c[1] - a[1]) / (c[0] - a[0])

    return np.abs(ratio)

"""Download the pose model if missing, return local path."""
def ensure_model(model_path: str = MODEL_PATH, url: str = MODEL_URL):
    if not os.path.exists(model_path):
        print(f"Downloading HEAVY model to: {model_path}")
        urllib.request.urlretrieve(url, model_path)
    return model_path

"""Extract (x, y, z) from a landmark."""
def get_landmark_xyz(landmarks, idx):
    lm = landmarks[idx]
    return (lm.x, lm.y, lm.z)

# These connection pairs are hardcoded since POSE_CONNECTIONS is gone
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
            if landmark.visibility > 0.5:
                cv2.circle(image,
                    (int(landmark.x * w), int(landmark.y * h)),
                    4, (255, 255, 0), -1)
                
# ==========================================
# SIGNAL EXTRACTION
# ==========================================

"""launch camera and detect eyes"""
def detect():
    # get the video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

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

    # Landmarker instance...
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Landmark indices (MediaPipe Pose)
    LI_idx = 1 # left eye inner
    LC_idx = 2 # left eye center
    LO_idx = 3 # left eye outer
    RI_idx = 4 # right eye inner
    RC_idx = 5 # right eye center
    RO_idx = 6 # right eye outer

    counter = 0
    stage = None
    start_time = time.time()

    while cap.isOpened():
        ok, frame = cap.read()      # the frame is in BGR color.
        if not ok:
            print("camera in use.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        # extract landmarks

        try:
            print("we're in try")
            lm = results.pose_landmarks[0]
            print("we got lm")
            
            # get the eye xyz coords
            LI = get_landmark_xyz(lm, LI_idx)
            LC = get_landmark_xyz(lm, LC_idx)
            LO = get_landmark_xyz(lm, LO_idx)
            RI = get_landmark_xyz(lm, RI_idx)
            RC = get_landmark_xyz(lm, RC_idx)
            RO = get_landmark_xyz(lm, RO_idx)
            print("we got xyz")

            left_ratio = calculate_ratio(LI, LC, LO)
            right_ratio = calculate_ratio(RI, RC, RO)
            
            print("we calculated ratios")

            # visualize the left eye ratio at the left eye center
            cv2.putText(image, str(left_ratio)[:5],
                            tuple(np.multiply(LC[:2], [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
            print("we put text 1")
            
            # visualize the right eye ratio at the right eye center
            cv2.putText(image, str(right_ratio)[:5],
                            tuple(np.multiply(RC[:2], [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
            print("we put text 2")
            # blink counter: first observe what the ratios  are when your eyes are shut vs open

            
        except Exception as e:
            print(f"Error: {e}")
        
        # draw the pose landmarks and pose connections
        draw_landmarks_and_connections(image, results)
        print("we called draw_landmarks...")
        
        """
        # first drawing spec: landmarks (points) (color is in BGR)
        # second drawing spec: connections (lines)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=2)
                                    )
        """

        # render the image
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
            

    # USE HEAVY INSTEAD OF LITE!!! for more accuracy in the smaller landmarks!!
    return

print("it ran")