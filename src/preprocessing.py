"""
Preprocessing and data transformation module for BLE Indoor Localization.
Handles label encoding, spatial coordinate lookup mapping, feature scaling, and train-test partitioning.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from .data_loader import parse_location_string, BEACON_COLUMNS
from .feature_engineering import engineer_features, get_feature_names

class PreprocessingPipeline:
    """
    Unified preprocessing pipeline that transforms raw BLE RSSI data into modeling-ready tensors.
    """
    def __init__(self, scaler_type: str = "standard", include_engineered: bool = True):
        self.scaler_type = scaler_type
        self.include_engineered = include_engineered
        self.label_encoder = LabelEncoder()
        
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None
            
        self.feature_names: List[str] = []
        self.classes_: np.ndarray = np.array([])
        self.class_to_coords: Dict[int, Tuple[int, int]] = {}
        self.loc_to_coords: Dict[str, Tuple[int, int]] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        """
        Fits label encoder, coordinate maps, and feature scaler on training dataframe.
        """
        # 1. Fit LabelEncoder on unique locations
        self.label_encoder.fit(df['location'].values)
        self.classes_ = self.label_encoder.classes_
        
        # 2. Build coordinate lookup dictionaries
        for idx, loc_str in enumerate(self.classes_):
            coords = parse_location_string(loc_str)
            self.class_to_coords[idx] = coords
            self.loc_to_coords[loc_str] = coords
            
        # 3. Engineer features
        if self.include_engineered:
            feat_df = engineer_features(df)
        else:
            feat_df = df.copy()
            
        self.feature_names = get_feature_names(self.include_engineered)
        X = feat_df[self.feature_names].values
        
        # 4. Fit scaler if enabled
        if self.scaler is not None:
            self.scaler.fit(X)
            
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforms input dataframe into (X, y) arrays. If 'location' column is missing or dummy, y is None.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")
            
        if self.include_engineered:
            feat_df = engineer_features(df)
        else:
            feat_df = df.copy()
            
        # Ensure all required feature columns exist
        for col in self.feature_names:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
                
        X = feat_df[self.feature_names].values
        if self.scaler is not None:
            X = self.scaler.transform(X)
            
        y = None
        if 'location' in df.columns:
            # Check if valid locations or unknown '?'
            valid_mask = df['location'].isin(self.label_encoder.classes_)
            if valid_mask.all():
                y = self.label_encoder.transform(df['location'].values)
            elif valid_mask.any():
                # Some valid, some unknown
                y = np.full(len(df), -1, dtype=int)
                y[valid_mask] = self.label_encoder.transform(df['location'].values[valid_mask])
                
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)

    def inverse_transform_labels(self, y_indices: np.ndarray) -> List[str]:
        """
        Maps class indices back to symbolic location strings (e.g. 14 -> 'K04').
        """
        return list(self.label_encoder.inverse_transform(y_indices))

    def get_coords_for_labels(self, y_indices: np.ndarray) -> np.ndarray:
        """
        Returns array of shape (N, 2) containing (x, y) coordinates for given class indices.
        """
        return np.array([self.class_to_coords[int(idx)] for idx in y_indices])

def prepare_train_test_split(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train/test split.
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['location']
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
