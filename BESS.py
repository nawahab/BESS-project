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

""" There will be 5 possible errors, each corresponding to a state in the state machine:
"EYES_CLOSED", "HANDS_OFF_HIPS", "STUMBLE", "HIP_ABDUCTION", "FOOT_LIFT"
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

@dataclass
class TrialResult:
    stance: str                         # double leg, single leg, or tandem
    surface: str                        # firm or foam
    duration_s: float       = 20.0      # each trial is 20 seconds long
    errors: List[TrialError]= field(default_factory=list)       # the list of errors committed.
    start_time: float       = 0.0       # will change
    end_time: float         = 0.0       # will changw

    @property
    def error_count(self):              # a method to return the number of errors for this trial result
        return len(self.errors)
    
