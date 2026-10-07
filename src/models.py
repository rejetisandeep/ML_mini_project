"""
Model factory and ensemble architectures for BLE Indoor Position Prediction.
Provides Baseline, Tuned Classifiers, and Multi-Model Ensemble Systems.
"""

from typing import Dict, Any, List
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def get_baseline_models() -> Dict[str, Any]:
    """
    Returns baseline models establishing initial performance benchmarks.
    """
    return {
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Baseline KNN (k=5, Uniform)": KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean")
    }

def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns the core standalone algorithms recommended in the project specifications:
    KNN, Random Forest, SVM, and XGBoost.
    """
    return {
        "Optimized KNN": KNeighborsClassifier(
            n_neighbors=3, 
            weights="distance", 
            metric="manhattan",
            n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, 
            max_depth=18, 
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state, 
            n_jobs=-1
        ),
        "Support Vector Machine (SVM)": SVC(
            C=10.0, 
            kernel="rbf", 
            gamma="scale", 
            probability=True, 
            random_state=random_state
        ),
        "XGBoost Classifier": xgb.XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1
        )
    }

def build_ensemble_models(base_candidates: Dict[str, Any], random_state: int = 42) -> Dict[str, Any]:
    """
    Builds Soft Voting and Stacking Ensemble classifiers combining heterogeneous algorithms.
    """
    estimators = [
        ("knn", base_candidates["Optimized KNN"]),
        ("rf", base_candidates["Random Forest"]),
        ("svm", base_candidates["Support Vector Machine (SVM)"]),
        ("xgb", base_candidates["XGBoost Classifier"])
    ]
    
    # Soft Voting Ensemble
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[1.5, 2.0, 1.5, 2.0],
        n_jobs=-1
    )
    
    # Stacking Classifier with Logistic Regression meta-learner
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500, C=1.0, random_state=random_state),
        cv=3,
        n_jobs=-1
    )
    
    return {
        "Soft Voting Ensemble": voting_clf,
        "Stacking Classifier": stacking_clf
    }
