"""
Exploratory Data Analysis (EDA) module for BLE Indoor Localization.
Generates comprehensive distribution plots, spatial grid maps, and signal correlation heatmaps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from .data_loader import BEACON_COLUMNS, augment_dataframe_with_coordinates

def run_exploratory_data_analysis(
    df: pd.DataFrame, 
    save_dir: str = "results/figures"
):
    """
    Performs comprehensive Exploratory Data Analysis and generates publication-grade figures.
    """
    sns.set_theme(style="whitegrid")
    df_aug = augment_dataframe_with_coordinates(df)
    
    # 1. Beacon Detection Rates & Mean Detected RSSI
    det_rates = []
    mean_signals = []
    for col in BEACON_COLUMNS:
        detected_mask = (df[col] > -200)
        rate = np.mean(detected_mask) * 100.0
        mean_sig = np.mean(df.loc[detected_mask, col]) if np.sum(detected_mask) > 0 else np.nan
        det_rates.append(rate)
        mean_signals.append(mean_sig)
        
    df_beacon_summary = pd.DataFrame({
        'beacon': BEACON_COLUMNS,
        'detection_rate_pct': det_rates,
        'mean_detected_rssi': mean_signals
    })
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(data=df_beacon_summary, x='beacon', y='detection_rate_pct', palette='viridis')
    plt.title("BLE Beacon Detection Rates (% Samples In-Range)", fontsize=11, fontweight='bold')
    plt.xlabel("iBeacon ID", fontsize=10)
    plt.ylabel("Detection Frequency (%)", fontsize=10)
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=df_beacon_summary, x='beacon', y='mean_detected_rssi', palette='mako')
    plt.title("Mean Detected RSSI (Excluding Out-of-Range)", fontsize=11, fontweight='bold')
    plt.xlabel("iBeacon ID", fontsize=10)
    plt.ylabel("Mean RSSI (dBm)", fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eda_beacon_detection_stats.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Number of Detected Beacons per Fingerprint
    num_detected = (df[BEACON_COLUMNS] > -200).sum(axis=1)
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=num_detected, palette='Blues_d')
    plt.title("Distribution of Concurrent In-Range Beacons per Measurement", fontsize=11, fontweight='bold')
    plt.xlabel("Number of Beacons Detected Simultaneously", fontsize=10)
    plt.ylabel("Sample Count", fontsize=10)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eda_num_beacons_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Spatial Grid Map of Sample Density across Waldo Library
    plt.figure(figsize=(10, 6))
    grid_counts = df_aug.groupby(['coord_x', 'coord_y']).size().reset_index(name='sample_count')
    scatter = plt.scatter(
        grid_counts['coord_x'], grid_counts['coord_y'],
        c=grid_counts['sample_count'], cmap='plasma',
        s=130, edgecolors='black', linewidth=0.8, alpha=0.9
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Number of Recorded Fingerprints", fontsize=10)
    plt.title("Waldo Library Floor Grid: Spatial Fingerprint Density", fontsize=12, fontweight='bold')
    plt.xlabel("Column X (Grid ord)", fontsize=11)
    plt.ylabel("Row Y (Room Cell)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eda_spatial_sample_density.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Beacon Signal Correlation Heatmap
    # Filter out -200 sentinels with NaN to check correlation when co-detected
    rssi_for_corr = df[BEACON_COLUMNS].replace(-200, np.nan)
    corr_matrix = rssi_for_corr.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                cbar_kws={'label': 'Pearson Correlation (Co-detected)'})
    plt.title("BLE Beacon Cross-Signal Correlation Matrix", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eda_beacon_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()
