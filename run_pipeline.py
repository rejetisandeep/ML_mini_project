"""
CLI script to trigger the full 12-stage Machine Learning training and evaluation pipeline.
"""
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    print(f"Launching BLE RSSI Indoor Position Prediction Pipeline in: {BASE_DIR}")
    results = run_full_pipeline(
        labeled_path=os.path.join(BASE_DIR, "data", "iBeacon_RSSI_Labeled.csv"),
        unlabeled_path=os.path.join(BASE_DIR, "data", "iBeacon_RSSI_Unlabeled.csv"),
        output_dir=os.path.join(BASE_DIR, "results"),
        model_dir=os.path.join(BASE_DIR, "models")
    )
    print("All tasks finished successfully!")
