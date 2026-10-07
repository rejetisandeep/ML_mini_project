"""
Standalone Command Line Interface (CLI) for BLE Indoor Position Prediction.
Usage:
    python predict.py --input_csv data/sample_test.csv
    python predict.py --b3002 -65 --b3004 -82
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEACON_COLS = [f"b30{i:02d}" for i in range(1, 14)]
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")

def predict_single(rssi_dict: dict):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model artifact not found at {MODEL_PATH}. Run 'python run_pipeline.py' first.")
        sys.exit(1)
        
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    pipeline = artifact["pipeline"]
    classes = artifact["classes"]
    class_to_coords = artifact["class_to_coords"]
    
    # Fill missing beacons with -200
    row_data = {}
    for col in BEACON_COLS:
        row_data[col] = float(rssi_dict.get(col, -200.0))
        
    df = pd.DataFrame([row_data])
    X, _ = pipeline.transform(df)
    
    pred_idx = model.predict(X)[0]
    pred_loc = pipeline.inverse_transform_labels(np.array([pred_idx]))[0]
    coord = class_to_coords[pred_idx]
    
    print("[LOCATION] BLE INDOOR POSITION PREDICTION")
    print(f"Predicted Location Code: {pred_loc}")
    print(f"Physical Grid Coordinates: Column X = {coord[0]} ({chr(ord('A') + coord[0])}), Row Y = {coord[1]}")
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        top3_indices = np.argsort(proba)[-3:][::-1]
        print("\nTop Candidate Locations:")
        for rank, idx in enumerate(top3_indices):
            loc_name = classes[idx]
            p = proba[idx] * 100.0
            c = class_to_coords[idx]
            print(f"  {rank+1}. Location {loc_name} (X={c[0]}, Y={c[1]}): {p:.2f}% confidence")
    print("=" * 50)

def predict_csv(csv_path: str, output_csv: str = "predictions.csv"):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model artifact not found at {MODEL_PATH}.")
        sys.exit(1)
        
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
        
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    pipeline = artifact["pipeline"]
    class_to_coords = artifact["class_to_coords"]
    
    df = pd.read_csv(csv_path)
    X, _ = pipeline.transform(df)
    
    preds = model.predict(X)
    pred_locations = pipeline.inverse_transform_labels(preds)
    pred_coords = [class_to_coords[idx] for idx in preds]
    
    df_out = df.copy()
    df_out['predicted_location'] = pred_locations
    df_out['predicted_coord_x'] = [c[0] for c in pred_coords]
    df_out['predicted_coord_y'] = [c[1] for c in pred_coords]
    
    df_out.to_csv(output_csv, index=False)
    print(f"Successfully processed {len(df)} samples and saved predictions to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict indoor location from BLE RSSI readings.")
    parser.add_argument("--input_csv", type=str, help="Path to CSV containing beacon RSSI columns")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output path for predictions CSV")
    
    for col in BEACON_COLS:
        parser.add_argument(f"--{col}", type=float, default=-200.0, help=f"RSSI for {col} (default: -200)")
        
    args = parser.parse_args()
    
    if args.input_csv:
        predict_csv(args.input_csv, args.output_csv)
    else:
        rssi_dict = {col: getattr(args, col) for col in BEACON_COLS}
        # Check if all are -200
        if all(val == -200.0 for val in rssi_dict.values()):
            print("Notice: No beacon RSSI passed; running demonstration prediction using sample fingerprint.")
            rssi_dict["b3002"] = -68.0
            rssi_dict["b3004"] = -85.0
        predict_single(rssi_dict)
