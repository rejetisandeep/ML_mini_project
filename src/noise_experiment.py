"""
Noise and Missing Signal Robustness Experiment Module for BLE RSSI Indoor Localization.
Investigates model degradation under synthetic multipath/shadowing noise and packet drop rates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from .data_loader import BEACON_COLUMNS
from .feature_engineering import engineer_features, get_feature_names

def inject_gaussian_noise(
    df: pd.DataFrame, 
    noise_std: float, 
    beacon_cols: List[str] = BEACON_COLUMNS,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Adds zero-mean Gaussian noise N(0, noise_std^2) to detected RSSI values (> -200 dBm).
    Out-of-range signals (-200) remain undetected.
    """
    if noise_std <= 0.0:
        return df.copy()
        
    rng = np.random.RandomState(random_state)
    df_noisy = df.copy()
    
    for col in beacon_cols:
        vals = df_noisy[col].values.astype(float)
        detected_mask = (vals > -200)
        noise = rng.normal(0, noise_std, size=np.sum(detected_mask))
        vals[detected_mask] = np.clip(vals[detected_mask] + noise, -120.0, -40.0)
        df_noisy[col] = vals
        
    return df_noisy

def inject_signal_drop(
    df: pd.DataFrame, 
    drop_rate: float, 
    beacon_cols: List[str] = BEACON_COLUMNS,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Randomly drops detected beacon signals (sets to -200 dBm) with probability drop_rate.
    Ensures at least 1 beacon remains detected per row to avoid empty vectors.
    """
    if drop_rate <= 0.0:
        return df.copy()
        
    rng = np.random.RandomState(random_state)
    df_dropped = df.copy()
    
    rssi_matrix = df_dropped[beacon_cols].values.copy()
    for row_idx in range(len(rssi_matrix)):
        detected_indices = np.where(rssi_matrix[row_idx] > -200)[0]
        if len(detected_indices) <= 1:
            continue
        drop_mask = rng.rand(len(detected_indices)) < drop_rate
        # Ensure at least 1 beacon is retained
        if np.all(drop_mask):
            drop_mask[0] = False
        to_drop = detected_indices[drop_mask]
        rssi_matrix[row_idx, to_drop] = -200.0
        
    for i, col in enumerate(beacon_cols):
        df_dropped[col] = rssi_matrix[:, i]
        
    return df_dropped

def evaluate_noise_robustness(
    models: Dict[str, Any],
    pipeline_fit: Any,
    test_df: pd.DataFrame,
    noise_levels: List[float] = [0.0, 2.0, 5.0, 8.0, 12.0, 15.0, 20.0],
    save_path: str = "results/figures/noise_robustness.png"
) -> pd.DataFrame:
    """
    Evaluates model accuracy and localization error across varying Gaussian noise levels.
    """
    records = []
    
    for noise in noise_levels:
        df_noisy = inject_gaussian_noise(test_df, noise_std=noise, random_state=int(42 + noise * 10))
        X_test_noisy, y_test = pipeline_fit.transform(df_noisy)
        true_coords = np.array([pipeline_fit.class_to_coords[idx] for idx in y_test])
        
        for name, model in models.items():
            preds = model.predict(X_test_noisy)
            acc = float(np.mean(preds == y_test))
            pred_coords = np.array([pipeline_fit.class_to_coords[idx] for idx in preds])
            dist_err = float(np.mean(np.sqrt(np.sum((pred_coords - true_coords) ** 2, axis=1))))
            
            records.append({
                'model': name,
                'noise_std_dbm': noise,
                'accuracy': acc,
                'mean_distance_error': dist_err
            })
            
    df_results = pd.DataFrame(records)
    
    # Plot accuracy and distance error vs noise level
    plt.figure(figsize=(14, 5))
    sns.set_theme(style="whitegrid")
    
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df_results, x='noise_std_dbm', y='accuracy', hue='model', marker='o', lw=2)
    plt.title("Model Accuracy vs. RSSI Gaussian Noise", fontsize=12, fontweight='bold')
    plt.xlabel("Injected Noise Std Dev (dBm)", fontsize=11)
    plt.ylabel("Classification Accuracy", fontsize=11)
    plt.legend(frameon=True, fontsize=9)
    
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df_results, x='noise_std_dbm', y='mean_distance_error', hue='model', marker='s', lw=2)
    plt.title("Localization Distance Error vs. RSSI Noise", fontsize=12, fontweight='bold')
    plt.xlabel("Injected Noise Std Dev (dBm)", fontsize=11)
    plt.ylabel("Mean Distance Error (grid units)", fontsize=11)
    plt.legend(frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_results
