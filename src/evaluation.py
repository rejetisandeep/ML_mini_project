"""
Evaluation module for BLE Indoor Localization.
Computes multiclass classification metrics (Accuracy, Top-K, F1) 
and physical localization metrics (Average Euclidean Distance Error, Error CDF).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix, 
    classification_report
)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coords_lookup: Dict[int, Tuple[int, int]],
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes a comprehensive dictionary of classification and physical localization metrics.
    """
    # 1. Standard Multiclass Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 2. Top-3 Accuracy (if probabilities provided)
    top3_acc = acc
    if y_proba is not None and y_proba.shape[1] >= 3:
        top3_indices = np.argsort(y_proba, axis=1)[:, -3:]
        is_in_top3 = [y_true[i] in top3_indices[i] for i in range(len(y_true))]
        top3_acc = np.mean(is_in_top3)
        
    # 3. Physical Euclidean Distance Localization Error
    # Map class indices to (x, y) coordinates
    true_coords = np.array([coords_lookup[int(idx)] for idx in y_true])
    pred_coords = np.array([coords_lookup[int(idx)] for idx in y_pred])
    
    # Distance in grid units
    dist_errors = np.sqrt(np.sum((pred_coords - true_coords) ** 2, axis=1))
    
    mean_dist_error = float(np.mean(dist_errors))
    median_dist_error = float(np.median(dist_errors))
    p75_dist_error = float(np.percentile(dist_errors, 75))
    p90_dist_error = float(np.percentile(dist_errors, 90))
    max_dist_error = float(np.max(dist_errors))
    
    # Grid cell neighbor accuracy: fraction within 1 unit distance
    within_1_grid = float(np.mean(dist_errors <= 1.0))
    # within 2 units distance
    within_2_grid = float(np.mean(dist_errors <= 2.0))
    
    return {
        'accuracy': float(acc),
        'top3_accuracy': float(top3_acc),
        'balanced_accuracy': float(bal_acc),
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'mean_distance_error': mean_dist_error,
        'median_distance_error': median_dist_error,
        'p75_distance_error': p75_dist_error,
        'p90_distance_error': p90_dist_error,
        'max_distance_error': max_dist_error,
        'within_1_grid_acc': within_1_grid,
        'within_2_grid_acc': within_2_grid
    }

def compute_location_wise_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    coords_lookup: Dict[int, Tuple[int, int]]
) -> pd.DataFrame:
    """
    Computes location-wise accuracy and average error for each unique indoor grid location.
    """
    records = []
    true_coords = np.array([coords_lookup[int(idx)] for idx in y_true])
    pred_coords = np.array([coords_lookup[int(idx)] for idx in y_pred])
    dist_errors = np.sqrt(np.sum((pred_coords - true_coords) ** 2, axis=1))
    
    for idx, loc_name in enumerate(class_names):
        mask = (y_true == idx)
        if np.sum(mask) == 0:
            continue
        loc_acc = np.mean(y_pred[mask] == y_true[mask])
        loc_err = np.mean(dist_errors[mask])
        x, y = coords_lookup[idx]
        records.append({
            'class_idx': idx,
            'location': loc_name,
            'coord_x': x,
            'coord_y': y,
            'sample_count': int(np.sum(mask)),
            'accuracy': float(loc_acc),
            'mean_distance_error': float(loc_err)
        })
        
    return pd.DataFrame(records).sort_values(by='mean_distance_error', ascending=False).reset_index(drop=True)

def plot_error_cdf(
    model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    coords_lookup: Dict[int, Tuple[int, int]],
    save_path: str = "results/figures/error_cdf.png"
):
    """
    Plots Cumulative Distribution Function (CDF) of localization distance error for multiple models.
    """
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for (model_name, (y_true, y_pred)), color in zip(model_predictions.items(), colors):
        true_coords = np.array([coords_lookup[int(idx)] for idx in y_true])
        pred_coords = np.array([coords_lookup[int(idx)] for idx in y_pred])
        dist_errors = np.sort(np.sqrt(np.sum((pred_coords - true_coords) ** 2, axis=1)))
        cdf = np.arange(1, len(dist_errors) + 1) / len(dist_errors)
        plt.plot(dist_errors, cdf, label=f"{model_name} (Median: {np.median(dist_errors):.2f})", lw=2, color=color)
        
    plt.title("Cumulative Distribution Function (CDF) of Localization Error", fontsize=13, fontweight='bold')
    plt.xlabel("Localization Distance Error (grid units)", fontsize=11)
    plt.ylabel("Cumulative Probability", fontsize=11)
    plt.xlim(0, 10)
    plt.ylim(0, 1.02)
    plt.axvline(1.0, color='gray', linestyle='--', alpha=0.6, label='1-Grid Threshold')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
