"""
Feature Selection and Importance Analysis Module for BLE RSSI Indoor Localization.
Implements Mutual Information, Random Forest Gini Importance, RFE, and comparative ranking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

def compute_feature_importances(
    X: np.ndarray, 
    y: np.ndarray, 
    feature_names: List[str],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes Mutual Information and Random Forest Gini Importance across all features.
    """
    # 1. Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    
    # 2. Random Forest Importance
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    rf_scores = rf.feature_importances_
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'mutual_info': mi_scores,
        'rf_importance': rf_scores
    })
    
    # Compute normalized composite score
    mi_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-9)
    rf_norm = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min() + 1e-9)
    df_importance['composite_importance'] = (mi_norm + rf_norm) / 2.0
    
    df_importance = df_importance.sort_values(by='composite_importance', ascending=False).reset_index(drop=True)
    return df_importance

def plot_feature_importance(
    importance_df: pd.DataFrame, 
    top_k: int = 20, 
    save_path: str = "results/figures/feature_importance.png"
):
    """
    Plots horizontal bar charts of top influential features by Mutual Information and Random Forest Importance.
    """
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    top_df = importance_df.head(top_k)
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=top_df, y='feature', x='mutual_info', color='#2b5c8f')
    plt.title(f"Top {top_k} Features by Mutual Information", fontsize=12, fontweight='bold')
    plt.xlabel("Mutual Information Score (nats)")
    plt.ylabel("Feature")
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=top_df, y='feature', x='rf_importance', color='#d95f02')
    plt.title(f"Top {top_k} Features by Random Forest Gini Importance", fontsize=12, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.ylabel("")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
