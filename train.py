import glob

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow import keras
from utils import WINDOW_SIZE, WINDOW_STEP
import time

EAR_THRESHOLD = 0.7

GRU_UNITS = 96
EPOCHS = 30
BATCH_SIZE = 32


def load_windows(session_paths):
    windows, labels = [], []
    for session_path in session_paths:
        session = pd.read_csv(session_path)
        if session["label"].nunique() < 2:
            continue
        session["ear"] = session["ear"] / session["ear"].quantile(0.9)
        features = session[["ear", "mar"]].values
        frame_labels = session["label"].values
        for start in range(0, len(features) - WINDOW_SIZE, WINDOW_STEP):
            windows.append(features[start:start + WINDOW_SIZE])
            labels.append(
                int(frame_labels[start:start + WINDOW_SIZE].mean() > 0.5))
    return np.array(windows), np.array(labels)


session_paths = sorted(glob.glob("sessions_ddd/*.csv"))
test_count = max(1, len(session_paths) // 5)
train_windows, train_labels = load_windows(session_paths[:-test_count])
test_windows, test_labels = load_windows(session_paths[-test_count:])

# train_paths = session_paths[test_count:]
# test_paths = session_paths[:test_count]   # a, b, c, d, e as the held-out people
# train_windows, train_labels = load_windows(train_paths)
# test_windows, test_labels = load_windows(test_paths)

model = keras.Sequential([
    keras.layers.Input(shape=(WINDOW_SIZE, 2)),
    keras.layers.GRU(GRU_UNITS),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(
    train_windows,
    train_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
)

# Evaluate GRU model
print("\n--- GRU model ---")

gru_predictions = (model.predict(test_windows).ravel() > 0.5).astype(int)

gru_report = classification_report(
    test_labels,
    gru_predictions,
    target_names=["alert", "drowsy"],
    zero_division=0,
    output_dict=True
)

print(classification_report(
    test_labels,
    gru_predictions,
    target_names=["alert", "drowsy"],
    zero_division=0
))

# Evaluate rule-based baseline
print("--- Rule-based baseline (EAR threshold) ---")

baseline_predictions = (
    (test_windows[:, :, 0] < EAR_THRESHOLD).mean(axis=1) > 0.5
).astype(int)

baseline_report = classification_report(
    test_labels,
    baseline_predictions,
    target_names=["alert", "drowsy"],
    zero_division=0,
    output_dict=True
)

print(classification_report(
    test_labels,
    baseline_predictions,
    target_names=["alert", "drowsy"],
    zero_division=0
))

# Generate model filename
def class_tag(prefix, class_report):
    return (
        f"{prefix}-p{class_report['precision']:.2f}"
        f"-r{class_report['recall']:.2f}"
        f"-f1{class_report['f1-score']:.2f}"
        f"-n{int(class_report['support'])}"
    )


timestamp = int(time.time())

model_name = (
    f"{timestamp}_ddd_"
    f"ws-{WINDOW_SIZE}_step-{WINDOW_STEP}_"
    f"gru-{GRU_UNITS}_ep-{EPOCHS}_bs-{BATCH_SIZE}_"
    f"{class_tag('alert', gru_report['alert'])}_"
    f"{class_tag('drowsy', gru_report['drowsy'])}"
    f".keras"
)

model.save(model_name)
print(f"\nModel saved as: {model_name}")
