from collections import deque

import cv2
import numpy as np
from tensorflow import keras
from utils import WINDOW_SIZE, WINDOW_STEP, extract_features

import sys

if len(sys.argv) != 2:
    print("Usage: python run_live.py <model>")
    sys.exit(1)
model_path = sys.argv[1]

CALIBRATION_FRAMES = 450
calibration_ears = []
open_eye_ear = None

ALERT_AFTER_DROWSY_CHECKS = 2

model = keras.models.load_model(model_path)
recent_features = deque(maxlen=WINDOW_SIZE)
frames_seen = 0
drowsy_streak = 0
camera = cv2.VideoCapture(0)

while True:
    
    success, frame = camera.read()
    if not success:
        break

    features = extract_features(frame)
    if features and open_eye_ear is None:
        calibration_ears.append(features[0])
        if len(calibration_ears) == CALIBRATION_FRAMES:
            open_eye_ear = np.quantile(calibration_ears, 0.9)
    elif features:
        ear, mar = features
        recent_features.append((ear / open_eye_ear, mar))
        frames_seen += 1

        if len(recent_features) == WINDOW_SIZE and frames_seen % WINDOW_STEP == 0:
            probability = model(np.array([recent_features]), training=False).numpy()[0][0]
            drowsy_streak = drowsy_streak + 1 if probability > 0.5 else 0

    if open_eye_ear is None:
        cv2.putText(frame, "Calibrating... keep eyes open", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if drowsy_streak >= ALERT_AFTER_DROWSY_CHECKS:
        cv2.putText(frame, "DROWSY", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
    cv2.imshow("Driver Drowsiness Detection (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
