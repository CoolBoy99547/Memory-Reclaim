import os
import sys
import pickle
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FEATURE_COLS

# ==========================================================
# Create ONNX output directory
# ==========================================================
ONNX_DIR = "onnx"
os.makedirs(ONNX_DIR, exist_ok=True)

# ==========================================================
# Load trained XGBoost model
# ==========================================================
with open("models/xgb_regressor.pkl", "rb") as f:
    model = pickle.load(f)

# ==========================================================
# Fix XGBoost feature names for ONNX conversion
# ONNXMLTools expects f0, f1, f2...
# ==========================================================
booster = model.get_booster()

print("Original feature names:")
print(booster.feature_names)

new_feature_names = [
    f"f{i}" for i in range(len(FEATURE_COLS))
]

booster.feature_names = new_feature_names
model._Booster = booster

print("\nRenamed feature names:")
print(model.get_booster().feature_names)

# ==========================================================
# Convert to ONNX
# ==========================================================
n_features = len(FEATURE_COLS)

input_type = [
    ("input", FloatTensorType([None, n_features]))
]

print(f"\nConverting model with {n_features} features...")

onnx_model = onnxmltools.convert_xgboost(
    model,
    initial_types=input_type
)

# ==========================================================
# Save ONNX model
# ==========================================================
onnx_path = os.path.join(
    ONNX_DIR,
    "XGBOOST.onnx"
)

onnxmltools.utils.save_model(
    onnx_model,
    onnx_path
)

print(f"\nSaved ONNX model:")
print(onnx_path)

size_kb = os.path.getsize(onnx_path) / 1024
print(f"Model size: {size_kb:.1f} KB")