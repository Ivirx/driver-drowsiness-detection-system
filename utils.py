import cv2
import numpy as np
import mediapipe as mp

WINDOW_SIZE = 30
WINDOW_STEP = 5

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 308, 14]

face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


def distance(point_a, point_b):
    return np.linalg.norm(point_a - point_b)


def eye_aspect_ratio(eye_points):
    corner_a, upper_a, upper_b, corner_b, lower_b, lower_a = eye_points
    vertical = distance(upper_a, lower_a) + distance(upper_b, lower_b)
    return vertical / (2 * distance(corner_a, corner_b))


def mouth_aspect_ratio(mouth_points):
    left_corner, upper_lip, right_corner, lower_lip = mouth_points
    return distance(upper_lip, lower_lip) / distance(left_corner, right_corner)


def extract_features(frame):
    """Returns (ear, mar) for the frame, or None if no face is found."""
    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None

    height, width = frame.shape[:2]
    landmarks = result.multi_face_landmarks[0].landmark

    def points(indices):
        return np.array([[landmarks[index].x * width, landmarks[index].y * height] for index in indices])

    ear = (eye_aspect_ratio(points(LEFT_EYE)) + eye_aspect_ratio(points(RIGHT_EYE))) / 2
    mar = mouth_aspect_ratio(points(MOUTH))
    return ear, mar
