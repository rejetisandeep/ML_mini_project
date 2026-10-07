"""
Data loading and coordinate parsing utilities for BLE RSSI Indoor Localization.
Dataset: BLE RSSI Dataset for Indoor Localization and Navigation (UCI Machine Learning Repository).
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List

BEACON_COLUMNS = [f"b30{i:02d}" for i in range(1, 14)]

def load_datasets(
    labeled_path: str = "data/iBeacon_RSSI_Labeled.csv",
    unlabeled_path: str = "data/iBeacon_RSSI_Unlabeled.csv"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Loads labeled and unlabeled BLE RSSI datasets.
    """
    if not os.path.exists(labeled_path):
        raise FileNotFoundError(f"Labeled dataset not found at {labeled_path}")
        
    df_labeled = pd.read_csv(labeled_path)
    df_unlabeled = None
    if os.path.exists(unlabeled_path):
        df_unlabeled = pd.read_csv(unlabeled_path)
        
    return df_labeled, df_unlabeled

def parse_location_string(loc_str: str) -> Tuple[int, int]:
    """
    Parses a symbolic location string like 'K04' or 'D13' into numerical (x, y) coordinates.
    x: Column coordinate based on letter (D=3, E=4, ..., W=22)
    y: Row coordinate based on number (1 to 15)
    """
    loc_str = str(loc_str).strip()
    match = re.match(r"^([A-Za-z]+)(\d+)$", loc_str)
    if not match:
        raise ValueError(f"Invalid location format: {loc_str}")
    
    col_str, row_str = match.groups()
    x = ord(col_str.upper()) - ord('A')
    y = int(row_str)
    return x, y

def coords_to_location(x: int, y: int) -> str:
    """
    Converts numerical (x, y) coordinates back to symbolic location string.
    """
    col_str = chr(ord('A') + int(round(x)))
    return f"{col_str}{int(round(y)):02d}"

def calculate_euclidean_distance(loc1: str, loc2: str) -> float:
    """
    Calculates physical Euclidean distance in grid units between two location strings.
    """
    x1, y1 = parse_location_string(loc1)
    x2, y2 = parse_location_string(loc2)
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def augment_dataframe_with_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds numeric coord_x and coord_y columns to the dataframe based on 'location'.
    """
    df = df.copy()
    coords = [parse_location_string(loc) for loc in df['location']]
    df['coord_x'] = [c[0] for c in coords]
    df['coord_y'] = [c[1] for c in coords]
    return df
