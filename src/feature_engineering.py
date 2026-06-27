# =============================================================
#  src/feature_engineering.py
#  Creates engineered features from raw memory signals.
#
#  Why engineer features?
#    Raw KB values tell you the current state.
#    Engineered features tell you the pressure, ratios,
#    and combined stress signals that actually drive
#    how much memory gets reclaimed.
#
#  IQR cap applied after feature creation.
#  Bounds computed from training sessions only.
#
#  Input  : data/processed/memory_reclaim_cleaned.csv
#  Output : data/features/memory_reclaim_final.csv
#           data/features/iqr_bounds.json
# =============================================================

import os
import sys
import json
import numpy  as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_FILE, DATA_FEATURES, FEATURES_DATA_FILE,
    IQR_BOUNDS_FILE, IQR_FEATURES, TRAIN_SPLIT
)


def add_memory_utilization(df):
    """
    How much of total RAM is actually being used?
    Utilization % is more meaningful than raw bytes
    because it normalizes across devices with different RAM.

    A 4GB used on 6GB device (67%) is more critical
    than 4GB used on 12GB device (33%).
    """
    df['memory_used_kb'] = (
        df['memtotal_kb'] - df['swapfree_kb']
    ).clip(lower=0)

    df['memory_utilization_pct'] = (
        df['memory_used_kb'] / df['memtotal_kb'] * 100
    ).clip(0, 100)

    # How much RAM is still freely available?
    df['ram_headroom_kb'] = (
        df['memtotal_kb'] - df['memory_used_kb']
    ).clip(lower=0)

    return df


def add_swap_features(df):
    """
    Swap usage signals severe memory pressure.
    When swap is heavily used, the system is under
    extreme memory stress — reclaim will be aggressive.
    """
    df['swap_used_kb'] = (
        df['swaptotal_kb'] - df['swapfree_kb']
    ).clip(lower=0)

    # Swap utilization — how full is swap?
    df['swap_utilization_pct'] = np.where(
        df['swaptotal_kb'] > 0,
        df['swap_used_kb'] / df['swaptotal_kb'] * 100,
        0
    ).clip(0, 100)

    return df


def add_heap_features(df):
    """
    Java heap metrics capture runtime memory state.
    Heap nearly full = GC pressure = reclaim likely imminent.
    """

    # What fraction of heap is being used?
    df['heap_utilization_pct'] = np.where(
        df['heapsize_kb'] > 0,
        (1 - df['heapmaxfree_kb'] / df['heapsize_kb']) * 100,
        0
    ).clip(0, 100)

    # Ratio of max free chunk to heap size
    # Low ratio = heap fragmented, GC under stress
    df['heap_free_ratio'] = np.where(
        df['heapsize_kb'] > 0,
        df['heapmaxfree_kb'] / df['heapsize_kb'],
        0
    ).clip(0, 1)

    return df


def add_dha_features(df):
    """
    Device Heap Allocator metrics.
    DHA cache ratio tells us how much of available
    heap memory is cached vs empty.
    High cache ratio = memory is reclaimable.
    """
    df['dha_total_kb'] = df['dha_empty_kb'] + df['dha_cache_kb']

    df['dha_cache_ratio'] = np.where(
        df['dha_total_kb'] > 0,
        df['dha_cache_kb'] / df['dha_total_kb'],
        0
    ).clip(0, 1)

    return df


def add_combined_stress_features(df):
    """
    Combined features that capture interaction between
    multiple subsystems under simultaneous pressure.
    These interaction terms are what separates
    good feature engineering from just raw signals.
    """

    # Memory pressure score — weighted combination
    # of memory utilization and swap utilization
    # High on both = system under severe pressure
    df['memory_pressure_score'] = (
        df['memory_utilization_pct'] * 0.60 +
        df['swap_utilization_pct']   * 0.40
    )

    # Thermal + memory interaction
    # Hot device under memory pressure = worst case for reclaim
    df['thermal_memory_interaction'] = (
        df['aptemp'] * df['memory_utilization_pct'] / 100
    )

    # CPU load × memory utilization
    # High CPU AND high memory = maximum system stress
    df['cpu_memory_stress'] = (
        df['currentcpuload'] * df['memory_utilization_pct'] / 100
    )

    return df


def compute_iqr_bounds(df_train, features):
    """
    Compute IQR bounds from training data only.
    Never computed on full dataset or test data.
    """
    bounds = {}
    for col in features:
        if col not in df_train.columns:
            continue
        Q1  = df_train[col].quantile(0.25)
        Q3  = df_train[col].quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (
            round(float(Q1 - 1.5 * IQR), 4),
            round(float(Q3 + 1.5 * IQR), 4)
        )
    return bounds


def apply_iqr_bounds(df, bounds):
    """Apply pre-computed IQR bounds to any DataFrame."""
    clipped = 0
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            bad     = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            clipped += bad
    return df, clipped


def engineer_features(df=None, save=True):
    """Run full feature engineering pipeline."""
    print("\nFeature Engineering")
    print("=" * 50)

    if df is None:
        df = pd.read_csv(PROCESSED_DATA_FILE)
        print(f"Loaded {len(df):,} rows")

    before = len(df.columns)

    df = add_memory_utilization(df)
    df = add_swap_features(df)
    df = add_heap_features(df)
    df = add_dha_features(df)
    df = add_combined_stress_features(df)

    added = len(df.columns) - before
    print(f"Added {added} engineered features")

    # IQR cap — computed on train sessions only
    sessions    = df['session_id'].unique().tolist()
    n_train     = int(len(sessions) * TRAIN_SPLIT)
    train_sess  = sessions[:n_train]
    df_train    = df[df['session_id'].isin(train_sess)]

    print(f"\nComputing IQR bounds on {len(train_sess)} train sessions...")
    iqr_bounds = compute_iqr_bounds(df_train, IQR_FEATURES)

    print("IQR bounds:")
    for col, (lo, hi) in iqr_bounds.items():
        print(f"  {col:<30} [{lo:>12.1f}, {hi:>12.1f}]")

    df, clipped = apply_iqr_bounds(df, iqr_bounds)
    print(f"\nIQR cap: corrected {clipped} values")
    print(f"Final shape: {df.shape}")

    if save:
        os.makedirs(DATA_FEATURES, exist_ok=True)
        df.to_csv(FEATURES_DATA_FILE, index=False)
        print(f"Saved → {FEATURES_DATA_FILE}")

        with open(IQR_BOUNDS_FILE, 'w') as f:
            json.dump(iqr_bounds, f, indent=2)
        print(f"Saved → {IQR_BOUNDS_FILE}")

    return df, iqr_bounds


if __name__ == '__main__':
    engineer_features()