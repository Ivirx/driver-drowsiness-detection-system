import glob
import os
import re
from collections import defaultdict

import cv2
import pandas as pd
from utils import extract_features

DATASET_FOLDER = "Driver Drowsiness Dataset (DDD)"
OUTPUT_FOLDER = "sessions_ddd"
FRAME_NAME = re.compile(r"^([A-Za-z]+)(\d+)\.\w+$")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

frames_by_person = defaultdict(list)
for image_path in glob.glob(f"{DATASET_FOLDER}/**/*.*", recursive=True):
    match = FRAME_NAME.match(os.path.basename(image_path))
    if match:
        letters, frame_number = match.groups()
        frames_by_person[letters.lower()].append((letters.isupper(), int(frame_number), image_path))

for person, frames in sorted(frames_by_person.items()):
    rows = []
    for is_drowsy, frame_number, image_path in sorted(frames):
        features = extract_features(cv2.imread(image_path))
        if features:
            rows.append([*features, int(is_drowsy)])
    pd.DataFrame(rows, columns=["ear", "mar", "label"]).to_csv(f"{OUTPUT_FOLDER}/ddd_{person}.csv", index=False)
    print(person, len(rows))
