"""
Feature Engineering module for BLE RSSI Indoor Fingerprinting.
Transforms raw beacon signals into physical propagation and statistical spatial features.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from .data_loader import BEACON_COLUMNS

def engineer_features(df: pd.DataFrame, beacon_cols: List[str] = None) -> pd.DataFrame:
    """
    Computes domain-specific radio propagation and statistical features from 13 BLE RSSI beacons.
    """
    if beacon_cols is None:
        beacon_cols = BEACON_COLUMNS
        
    features_df = df.copy()
    rssi_matrix = features_df[beacon_cols].values # shape (N, 13)
    
    # 1. Binary visibility flags (1 if detected, 0 if out-of-range -200)
    is_visible = (rssi_matrix > -200).astype(float)
    for i, col in enumerate(beacon_cols):
        features_df[f"{col}_present"] = is_visible[:, i]
        
    # 2. Number of detected beacons
    num_detected = np.sum(is_visible, axis=1)
    features_df['num_detected'] = num_detected
    
    # 3. Statistical summaries across detected beacons
    # Mask out -200 values with NaN for clean statistics
    masked_rssi = np.where(rssi_matrix > -200, rssi_matrix, np.nan)
    
    # Max RSSI (strongest signal)
    max_rssi = np.nanmax(masked_rssi, axis=1)
    features_df['max_rssi'] = np.where(np.isnan(max_rssi), -200.0, max_rssi)
    
    # Min RSSI among detected
    min_rssi = np.nanmin(masked_rssi, axis=1)
    features_df['min_rssi'] = np.where(np.isnan(min_rssi), -200.0, min_rssi)
    
    # Mean RSSI of detected
    mean_rssi = np.nanmean(masked_rssi, axis=1)
    features_df['mean_rssi'] = np.where(np.isnan(mean_rssi), -200.0, mean_rssi)
    
    # Standard deviation of detected RSSI
    std_rssi = np.nanstd(masked_rssi, axis=1)
    features_df['std_rssi'] = np.where(np.isnan(std_rssi), 0.0, std_rssi)
    
    # RSSI spread (max - min)
    features_df['rssi_spread'] = np.where(
        np.isnan(max_rssi) | np.isnan(min_rssi), 0.0, max_rssi - min_rssi
    )
    
    # 4. Strongest and 2nd Strongest Beacon Identification & Delta
    strongest_idx = np.argmax(rssi_matrix, axis=1)
    features_df['strongest_beacon_idx'] = strongest_idx
    
    # Find 2nd strongest RSSI
    sorted_rssi = np.sort(rssi_matrix, axis=1) # ascending
    second_max = sorted_rssi[:, -2] # second highest
    features_df['second_max_rssi'] = second_max
    
    # Top 2 RSSI delta (signal gap between top two beacons)
    features_df['top2_delta'] = features_df['max_rssi'] - second_max
    
    # 5. Approximate Linear Power Sum (mW scale)
    # Power = 10^(RSSI/10) for detected signals
    detected_power = np.where(rssi_matrix > -200, 10.0 ** (rssi_matrix / 10.0), 0.0)
    power_sum = np.sum(detected_power, axis=1)
    features_df['power_sum'] = power_sum
    
    # 6. Normalized RSSI (shifted positive range [0, 1])
    # -200 maps to 0.0, -50 dBm maps to 1.0
    normalized_rssi = (rssi_matrix + 200.0) / 150.0
    normalized_rssi = np.clip(normalized_rssi, 0.0, 1.0)
    for i, col in enumerate(beacon_cols):
        features_df[f"{col}_norm"] = normalized_rssi[:, i]
        
    return features_df

def get_feature_names(include_engineered: bool = True) -> List[str]:
    """
    Returns list of feature names depending on whether engineered features are included.
    """
    if not include_engineered:
        return BEACON_COLUMNS.copy()
        
    cols = BEACON_COLUMNS.copy()
    # Presence indicators
    cols += [f"{c}_present" for c in BEACON_COLUMNS]
    # Stats
    cols += ['num_detected', 'max_rssi', 'min_rssi', 'mean_rssi', 'std_rssi', 
             'rssi_spread', 'strongest_beacon_idx', 'second_max_rssi', 'top2_delta', 'power_sum']
    # Normalized
    cols += [f"{c}_norm" for c in BEACON_COLUMNS]
    return cols
