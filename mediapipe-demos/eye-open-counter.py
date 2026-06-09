import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ==========================================
# CONFIG
# ==========================================
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"

def ensure_model(model_path=MODEL_PATH, url=MODEL_URL):
    if not os.path.exists(model_path):
        print(f"Downloading Face Landmarker model to: {model_path}")
        urllib.request.urlretrieve(url, model_path)
    return model_path

## E.A.R.: Eye Aspect Ratio
# landmark indices for MediaPipe FaceLandmarker (478-point model)
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


""" These connection pairs are hardcoded since POSE_CONNECTIONS is gone from Mediapipe 0.10.35"""
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]

""" Draw the face landmarks and connections """
def draw_landmarks_and_connections(image, results):
    if not results.face_landmarks:
        return

    h, w, _ = image.shape

    for face_landmarks in results.face_landmarks:
        """# Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            start = pose_landmarks[start_idx]
            end = pose_landmarks[end_idx]

            if start.visibility > 0.5 and end.visibility > 0.5:
                cv2.line(image,
                    (int(start.x * w), int(start.y * h)),
                    (int(end.x * w), int(end.y * h)),
                    (255, 0, 255), 2)"""

        # Draw landmark dots
        for landmark in face_landmarks:
            try:
                if landmark.visibility > 0.5:
                    cv2.circle(image,
                        (int(landmark.x * w), int(landmark.y * h)),
                        4, (255, 255, 0), -1)
            except:
                pass
    return

""" Opens live video feed from camera. Counts how many times you open your eyes from closed. """
def detect():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("camera in use.")

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model_path = ensure_model()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # initialize the face landmarker model
    landmarker = vision.FaceLandmarker.create_from_options(options)

    start_time = time.time()
    
    stage = "EYES OPEN"
    counter = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Something wrong with reading camera capture")
            break

        # change to bgr for cv2
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        # detect face landmarks
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # change to rgb for mp
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 
        if results.face_landmarks:
            lm = results.face_landmarks[0]

            left_aspect_ratio  = calculate_aspect_ratio(lm, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_INNER, LEFT_EYE_OUTER, img_w, img_h)
            right_aspect_ratio = calculate_aspect_ratio(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, img_w, img_h)
            avg_aspect_ratio   = (left_aspect_ratio + right_aspect_ratio) / 2

            
            # "reverse blink" detection
            # calibrate this threshold later...
            EAR_THRESHOLD = 0.2
            if avg_aspect_ratio < EAR_THRESHOLD:
                stage = "EYES CLOSED"
            if avg_aspect_ratio > EAR_THRESHOLD and stage == "EYES CLOSED":
                stage = "EYES OPEN"
                counter += 1
                
            # draw face landmarks
            draw_landmarks_and_connections(image, results)
                
            # Display EAR values
            # Colors are in (B, G, R) format.
            cv2.putText(image, f"Left EAR:  {left_aspect_ratio:.3f}",  (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # white
            cv2.putText(image, f"Right EAR: {right_aspect_ratio:.3f}", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Avg EAR:   {avg_aspect_ratio:.3f}",   (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Open Count: {counter}",   (10, 130),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # red
            cv2.putText(image, stage, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3) # cyan
            
            """
            # curl counter logic:
            if angle > 160:
                stage = "down"
            if angle < 30 and stage =="down":
                stage = "up"
                counter += 1
                print(counter)
            """


        cv2.imshow('Face Feed', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect()