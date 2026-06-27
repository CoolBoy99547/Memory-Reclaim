# =============================================================
#  src/preprocess.py
#  Cleans raw memory reclaim dataset.
#
#  Steps:
#    1. Load raw CSV
#    2. Sort by session + row order
#    3. Handle missing values
#    4. Physical clip on KB columns
#    5. Encode all categorical columns
#    6. Add device RAM column from device name
#
#  Input  : data/raw/memory_reclaim_60k.csv
#  Output : data/processed/memory_reclaim_cleaned.csv
# =============================================================

import os
import sys
import numpy  as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_FILE, PROCESSED_DATA_FILE, DATA_PROCESSED,
    CLIP_RULES, GRAPHICS_MAP, COMP_FILTER_MAP, ODEX_MAP,
    GAME_PRESSURE_MAP, GAME_CATEGORY_MAP, DEVICE_RAM_MAP
)


def load_raw():
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"Sessions : {df['session_id'].nunique():,}")
    return df


def sort_data(df):
    """
    Sort by session_id to keep session rows together.
    Within each session rows are already time ordered.
    """
    df = df.sort_values('session_id').reset_index(drop=True)
    print("Sorted by session_id")
    return df


def handle_missing(df):
    """
    Fill missing values within each session.
    Forward fill first, backward fill for session start gaps.
    """
    null_total = df.isnull().sum().sum()
    if null_total == 0:
        print("No missing values found")
        return df

    print(f"Found {null_total} missing values — filling...")
    df = df.groupby('session_id', group_keys=False).apply(
        lambda g: g.ffill().bfill()
    )
    print(f"Remaining: {df.isnull().sum().sum()}")
    return df


def apply_physical_clip(df):
    """
    Layer 1 outlier handling.
    Clip raw KB values to physically valid ranges.
    Based on Samsung device hardware specifications.
    """
    clipped = 0
    for col, (lo, hi) in CLIP_RULES.items():
        if col in df.columns:
            bad     = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            clipped += bad
    print(f"Physical clip: corrected {clipped} values")
    return df


def encode_categoricals(df):
    """
    Encode all categorical string columns to integers.
    Tree models need numbers not strings.
    Original columns kept for readability.
    """

    # Graphics setting
    df['graphics_encoded'] = df['graphics_setting'].map(
        GRAPHICS_MAP
    ).fillna(1).astype(int)

    # Compilation filter
    df['compilation_filter_encoded'] = df['compilation_filter'].map(
        COMP_FILTER_MAP
    ).fillna(0).astype(int)

    # Odex / vdex
    df['odex_encoded'] = df['odex_vdex'].map(
        ODEX_MAP
    ).fillna(0).astype(int)

    # Multiuser
    df['multiuser_encoded'] = (
        df['multiuser'] == 'multiuser'
    ).astype(int)

    # Online / offline
    df['online_encoded'] = (
        df['online_offline'] == 'online'
    ).astype(int)

    # Game memory pressure rank
    df['game_pressure_encoded'] = df['game'].map(
        GAME_PRESSURE_MAP
    ).fillna(5).astype(int)

    # Game category
    df['game_category_encoded'] = df['game_category'].map(
        GAME_CATEGORY_MAP
    ).fillna(0).astype(int)

    # Device RAM in GB
    df['device_ram_gb'] = df['device'].map(
        DEVICE_RAM_MAP
    ).fillna(8).astype(int)

    print(f"Encoded 8 categorical columns")
    return df


def preprocess(save=True):
    """Run full preprocessing pipeline."""
    print("\nPreprocessing")
    print("=" * 50)

    df = load_raw()
    df = sort_data(df)
    df = handle_missing(df)
    df = apply_physical_clip(df)
    df = encode_categoricals(df)

    print(f"\nOutput shape : {df.shape}")

    if save:
        os.makedirs(DATA_PROCESSED, exist_ok=True)
        df.to_csv(PROCESSED_DATA_FILE, index=False)
        print(f"Saved → {PROCESSED_DATA_FILE}")

    return df


if __name__ == '__main__':
    preprocess()