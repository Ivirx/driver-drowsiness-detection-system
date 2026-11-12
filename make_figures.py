import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from utils import WINDOW_SIZE, WINDOW_STEP
 
MODEL_PATH = "1794377878_ddd_ws-30_step-5_gru-96_ep-30_bs-32_alert-p0.87-r0.85-f10.86-n1232_drowsy-p0.85-r0.88-f10.86-n1260.keras"
EAR_THRESHOLD = 0.7
TEST_PERSON_COUNT = 5
 
 
def load_windows(session_path):
    session = pd.read_csv(session_path)
    session["ear"] = session["ear"] / session["ear"].quantile(0.9)
    features = session[["ear", "mar"]].values
    frame_labels = session["label"].values
    windows, labels = [], []
    for start in range(0, len(features) - WINDOW_SIZE, WINDOW_STEP):
        windows.append(features[start:start + WINDOW_SIZE])
        labels.append(int(frame_labels[start:start + WINDOW_SIZE].mean() > 0.5))
    return np.array(windows), np.array(labels)
 
 
def save_confusion_matrix(true_labels, predictions, title, file_name):
    matrix = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(3, 3))
    plt.imshow(matrix, cmap="Greys")
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            plt.text(column, row, matrix[row, column], ha="center", va="center", color=color)
    plt.xticks([0, 1], ["alert", "drowsy"])
    plt.yticks([0, 1], ["alert", "drowsy"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    plt.close()
 
 
model = keras.models.load_model(MODEL_PATH)
test_paths = sorted(glob.glob("sessions_ddd/*.csv"))[-TEST_PERSON_COUNT:]
loaded = [load_windows(path) for path in test_paths]
test_windows = np.concatenate([windows for windows, _ in loaded])
test_labels = np.concatenate([labels for _, labels in loaded])
 
gru_predictions = (model.predict(test_windows).ravel() > 0.5).astype(int)
baseline_predictions = ((test_windows[:, :, 0] < EAR_THRESHOLD).mean(axis=1) > 0.5).astype(int)
for name, predictions in [("GRU", gru_predictions), ("Baseline", baseline_predictions)]:
    print(name)
    print(classification_report(test_labels, predictions, target_names=["alert", "drowsy"]))
    save_confusion_matrix(test_labels, predictions, name, f"confusion_{name.lower()}.png")
 
session = pd.read_csv("sessions_ddd/ddd_a.csv")
session["ear"] = session["ear"] / session["ear"].quantile(0.9)
figure, axes = plt.subplots(2, 1, sharey=True, figsize=(7, 6))
for axis, label, title in zip(axes, [0, 1], ["Alert", "Drowsy"]):
    axis.plot(session[session["label"] == label]["ear"].values[:300], color="black")
    axis.set_title(title)
plt.savefig("ear_plot.png", dpi=200, bbox_inches="tight")
