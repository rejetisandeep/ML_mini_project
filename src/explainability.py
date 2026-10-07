"""
Explainability and Error Analysis Module for BLE Indoor Localization.
Implements SHAP explanations, Permutation Importance, and spatial error diagnosis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.inspection import permutation_importance
import shap

def run_permutation_importance(
    model: Any, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42,
    save_path: str = "results/figures/permutation_importance.png"
) -> pd.DataFrame:
    """
    Computes and plots Permutation Feature Importance on test/validation data.
    """
    result = permutation_importance(
        model, X_val, y_val, 
        n_repeats=n_repeats, 
        random_state=random_state, 
        n_jobs=-1
    )
    
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance_mean', ascending=False).reset_index(drop=True)
    
    # Plot top 15 features
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    top_df = perm_df.head(15)
    
    plt.barh(range(len(top_df)), top_df['importance_mean'][::-1], xerr=top_df['importance_std'][::-1], 
             align='center', color='#1b9e77', alpha=0.85, capsize=4)
    plt.yticks(range(len(top_df)), top_df['feature'][::-1])
    plt.xlabel("Mean Accuracy Decrease (Permutation Importance)", fontsize=11)
    plt.title("Permutation Feature Importance (Validation Set)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return perm_df

def run_shap_analysis(
    rf_model: Any, 
    X_sample: np.ndarray, 
    feature_names: List[str],
    save_path: str = "results/figures/shap_summary.png"
):
    """
    Computes Tree SHAP values on tree-based model and generates summary plot.
    """
    try:
        explainer = shap.TreeExplainer(rf_model)
        # Use a representative subset for SHAP computation speed
        sub_X = X_sample[:100]
        shap_values = explainer.shap_values(sub_X)
        
        plt.figure(figsize=(10, 6))
        # For multiclass, average absolute SHAP values across classes
        if isinstance(shap_values, list):
            # List of arrays per class: average across classes
            mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0) # shape (100, n_features)
            mean_importance = np.mean(mean_abs_shap, axis=0)
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Shape (100, n_features, n_classes)
            mean_importance = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            mean_importance = np.mean(np.abs(shap_values), axis=0)
            
        feat_df = pd.DataFrame({'feature': feature_names, 'shap_importance': mean_importance})
        feat_df = feat_df.sort_values(by='shap_importance', ascending=False).head(15)
        
        sns.set_theme(style="whitegrid")
        sns.barplot(data=feat_df, y='feature', x='shap_importance', color='#7570b3')
        plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=12, fontweight='bold')
        plt.xlabel("Mean |SHAP Value| Across Location Classes", fontsize=11)
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"SHAP analysis warning: {e}")

def analyze_error_patterns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    coords_lookup: Dict[int, Tuple[int, int]],
    save_path: str = "results/figures/error_spatial_map.png"
) -> pd.DataFrame:
    """
    Identifies high-error location clusters and physical failure modes.
    """
    true_coords = np.array([coords_lookup[int(idx)] for idx in y_true])
    pred_coords = np.array([coords_lookup[int(idx)] for idx in y_pred])
    dist_errors = np.sqrt(np.sum((pred_coords - true_coords) ** 2, axis=1))
    
    df_err = pd.DataFrame({
        'true_loc': class_names[y_true],
        'pred_loc': class_names[y_pred],
        'true_x': true_coords[:, 0],
        'true_y': true_coords[:, 1],
        'dist_error': dist_errors,
        'is_correct': (y_true == y_pred)
    })
    
    # Spatial heatmap of localization errors
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="white")
    
    # Aggregate error by true location
    agg_loc = df_err.groupby(['true_x', 'true_y'])['dist_error'].mean().reset_index()
    
    scatter = plt.scatter(
        agg_loc['true_x'], agg_loc['true_y'], 
        c=agg_loc['dist_error'], cmap='YlOrRd', 
        s=120, edgecolors='black', linewidth=0.8, alpha=0.9
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Mean Distance Error (grid units)", fontsize=10)
    plt.title("Physical Spatial Distribution of Mean Localization Error", fontsize=12, fontweight='bold')
    plt.xlabel("Grid Column X (Letter ord)", fontsize=11)
    plt.ylabel("Grid Row Y (Room index)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_err
