# export_artifacts.py

import os
import json
import pickle

from config import FEATURE_COLS

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

scaler_json = {
    "feature_names": FEATURE_COLS,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}

with open(
    os.path.join(ARTIFACT_DIR, "scaler.json"),
    "w"
) as f:
    json.dump(scaler_json, f, indent=2)

print("Saved scaler.json")