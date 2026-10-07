"""
End-to-End Machine Learning Pipeline for Indoor Position Prediction using BLE Signal Fingerprints.
Executes all 12 stages of the project workflow and generates comprehensive deliverables.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from .data_loader import load_datasets
from .eda import run_exploratory_data_analysis
from .preprocessing import PreprocessingPipeline, prepare_train_test_split
from .feature_engineering import get_feature_names
from .feature_selection import compute_feature_importances, plot_feature_importance
from .models import get_baseline_models, get_candidate_models, build_ensemble_models
from .evaluation import evaluate_predictions, compute_location_wise_accuracy, plot_error_cdf
from .explainability import run_permutation_importance, run_shap_analysis, analyze_error_patterns
from .noise_experiment import evaluate_noise_robustness

def run_full_pipeline(
    labeled_path: str = "data/iBeacon_RSSI_Labeled.csv",
    unlabeled_path: str = "data/iBeacon_RSSI_Unlabeled.csv",
    output_dir: str = "results",
    model_dir: str = "models",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the complete 12-stage ML pipeline.
    """
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 70)
    print("STAGE 1 & 2: Dataset Ingestion & Preprocessing Setup")
    print("=" * 70)
    df_labeled, df_unlabeled = load_datasets(labeled_path, unlabeled_path)
    print(f"Loaded labeled data: {df_labeled.shape} | Unlabeled data: {df_unlabeled.shape if df_unlabeled is not None else 'None'}")
    
    print("\n" + "=" * 70)
    print("STAGE 3: Exploratory Data Analysis (EDA)")
    print("=" * 70)
    run_exploratory_data_analysis(df_labeled, save_dir=f"{output_dir}/figures")
    print("Saved EDA plots to results/figures/")
    
    print("\n" + "=" * 70)
    print("STAGE 4: Data Partitioning & Feature Engineering")
    print("=" * 70)
    train_df, test_df = prepare_train_test_split(df_labeled, test_size=0.2, random_state=random_state)
    print(f"Train split: {train_df.shape} | Test split: {test_df.shape}")
    
    # Preprocessing & Feature Engineering Pipeline
    pipeline = PreprocessingPipeline(scaler_type="standard", include_engineered=True)
    X_train, y_train = pipeline.fit_transform(train_df)
    X_test, y_test = pipeline.transform(test_df)
    print(f"Engineered Feature Space: {X_train.shape[1]} features across {len(pipeline.classes_)} location classes.")
    
    print("\n" + "=" * 70)
    print("STAGE 5: Feature Selection & Influence Analysis")
    print("=" * 70)
    importance_df = compute_feature_importances(X_train, y_train, pipeline.feature_names, random_state=random_state)
    plot_feature_importance(importance_df, top_k=20, save_path=f"{output_dir}/figures/feature_importance.png")
    importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    print("Top 5 most influential features:")
    print(importance_df[['feature', 'mutual_info', 'rf_importance', 'composite_importance']].head(5))
    
    print("\n" + "=" * 70)
    print("STAGE 6, 7 & 8: Model Training (Baselines, Candidates, Ensembles)")
    print("=" * 70)
    
    # 1. Baseline models (Image 2 Deliverable 1)
    # Train Baselines on raw RSSI features for direct benchmark
    raw_pipeline = PreprocessingPipeline(scaler_type="standard", include_engineered=False)
    X_train_raw, y_train_raw = raw_pipeline.fit_transform(train_df)
    X_test_raw, y_test_raw = raw_pipeline.transform(test_df)
    
    baselines = get_baseline_models()
    trained_baselines = {}
    for name, model in baselines.items():
        print(f"Training Baseline: {name}...")
        model.fit(X_train_raw, y_train_raw)
        trained_baselines[name] = model
        
    # 2. Candidate models (Image 2 Deliverable 2 & 3)
    candidates = get_candidate_models(random_state=random_state)
    trained_candidates = {}
    for name, model in candidates.items():
        print(f"Training Candidate Model: {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0
        trained_candidates[name] = model
        print(f"  -> Finished in {fit_time:.2f}s")
        
    # 3. Ensemble architectures (Soft Voting & Stacking)
    print("Building Ensemble Architectures (Soft Voting & Stacking)...")
    ensembles = build_ensemble_models(trained_candidates, random_state=random_state)
    trained_ensembles = {}
    for name, model in ensembles.items():
        print(f"Training Ensemble: {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0
        trained_ensembles[name] = model
        print(f"  -> Finished in {fit_time:.2f}s")
        
    print("\n" + "=" * 70)
    print("STAGE 9: Comparative Evaluation & Performance Metrics")
    print("=" * 70)
    
    all_models = {}
    model_predictions = {}
    results_records = []
    
    # Evaluate Baselines
    for name, model in trained_baselines.items():
        preds = model.predict(X_test_raw)
        proba = model.predict_proba(X_test_raw) if hasattr(model, "predict_proba") else None
        metrics = evaluate_predictions(y_test_raw, preds, pipeline.class_to_coords, proba)
        metrics['model'] = name
        metrics['category'] = 'Baseline'
        results_records.append(metrics)
        model_predictions[name] = (y_test_raw, preds)
        
    # Evaluate Candidates & Ensembles
    for name, model in {**trained_candidates, **trained_ensembles}.items():
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        metrics = evaluate_predictions(y_test, preds, pipeline.class_to_coords, proba)
        metrics['model'] = name
        metrics['category'] = 'Proposed/Ensemble' if 'Ensemble' in name or 'Stacking' in name else 'Candidate'
        results_records.append(metrics)
        model_predictions[name] = (y_test, preds)
        all_models[name] = model
        
    df_metrics = pd.DataFrame(results_records)
    # Reorder columns
    cols_order = ['model', 'category', 'accuracy', 'top3_accuracy', 'mean_distance_error', 
                  'median_distance_error', 'within_1_grid_acc', 'f1_weighted', 'f1_macro']
    df_metrics = df_metrics[cols_order].sort_values(by='mean_distance_error', ascending=True).reset_index(drop=True)
    df_metrics.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    
    print("\n--- FINAL COMPARATIVE RESULTS ---")
    print(df_metrics.to_string(index=False))
    
    # Plot Error CDF
    plot_error_cdf(model_predictions, pipeline.class_to_coords, save_path=f"{output_dir}/figures/error_cdf.png")
    
    # Identify best performing model (lowest mean distance error / highest accuracy)
    best_row = df_metrics[df_metrics['category'] != 'Baseline'].iloc[0]
    best_model_name = best_row['model']
    best_model = all_models[best_model_name]
    print(f"\nOptimal Selected System: {best_model_name} (Mean Dist Error: {best_row['mean_distance_error']:.2f} units, Acc: {best_row['accuracy']*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("STAGE 10: Explainability & Error Diagnosis")
    print("=" * 70)
    # Permutation Importance
    print("Running Permutation Importance on validation set...")
    run_permutation_importance(
        best_model, X_test, y_test, pipeline.feature_names,
        save_path=f"{output_dir}/figures/permutation_importance.png"
    )
    
    # SHAP Tree Explainer
    print("Running SHAP Analysis on Random Forest...")
    run_shap_analysis(
        trained_candidates["Random Forest"], X_test, pipeline.feature_names,
        save_path=f"{output_dir}/figures/shap_summary.png"
    )
    
    # Error Spatial Diagnosis
    best_preds = model_predictions[best_model_name][1]
    err_df = analyze_error_patterns(
        y_test, best_preds, pipeline.classes_, pipeline.class_to_coords,
        save_path=f"{output_dir}/figures/error_spatial_map.png"
    )
    
    loc_acc_df = compute_location_wise_accuracy(y_test, best_preds, pipeline.classes_, pipeline.class_to_coords)
    loc_acc_df.to_csv(f"{output_dir}/location_wise_accuracy.csv", index=False)
    
    print("\n" + "=" * 70)
    print("STAGE 11: Noise & Packet Drop Robustness Experiments")
    print("=" * 70)
    # Evaluate noise resilience across candidate models
    noise_eval_models = {
        "Optimized KNN": trained_candidates["Optimized KNN"],
        "Random Forest": trained_candidates["Random Forest"],
        "SVM": trained_candidates["Support Vector Machine (SVM)"],
        "XGBoost": trained_candidates["XGBoost Classifier"],
        "Soft Voting": trained_ensembles["Soft Voting Ensemble"]
    }
    noise_results = evaluate_noise_robustness(
        noise_eval_models, pipeline, test_df,
        save_path=f"{output_dir}/figures/noise_robustness.png"
    )
    noise_results.to_csv(f"{output_dir}/noise_robustness_metrics.csv", index=False)
    print("Saved Noise Robustness analysis to results/figures/noise_robustness.png")
    
    print("\n" + "=" * 70)
    print("STAGE 12: System Serialization & Production Artifact Generation")
    print("=" * 70)
    model_artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "pipeline": pipeline,
        "classes": pipeline.classes_,
        "class_to_coords": pipeline.class_to_coords,
        "feature_names": pipeline.feature_names,
        "metrics": best_row.to_dict()
    }
    save_target = f"{model_dir}/best_model.joblib"
    joblib.dump(model_artifact, save_target)
    print(f"Serialized optimal model and preprocessing pipeline to {save_target}")
    
    # Run test inference on unlabeled dataset
    if df_unlabeled is not None:
        print("\nPredicting on Unlabeled Test Dataset (out-of-sample demonstration):")
        unlabeled_sample = df_unlabeled.head(5).copy()
        X_unlab, _ = pipeline.transform(unlabeled_sample)
        unlab_preds = best_model.predict(X_unlab)
        unlab_locations = pipeline.inverse_transform_labels(unlab_preds)
        for i, loc in enumerate(unlab_locations):
            x, y = pipeline.class_to_coords[unlab_preds[i]]
            print(f"  Unlabeled Sample #{i+1} [Time: {unlabeled_sample['date'].iloc[i]}]: Predicted Location = {loc} (Grid: x={x}, y={y})")
            
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return {
        "metrics": df_metrics,
        "best_model_name": best_model_name,
        "artifact_path": save_target
    }
