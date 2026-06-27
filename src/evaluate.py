# =============================================================
#  src/evaluate.py
#  Evaluates model performance on test set.
#
#  Reports:
#    Global MAE, RMSE, Median Error
#    Per-game breakdown
#    Per-device breakdown
#    Per-graphics-setting breakdown
#    Baseline comparison
# =============================================================

import sys
import os
import numpy  as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET


def evaluate_global(y_true, y_pred):
    """Global regression metrics."""
    y_pred  = np.clip(y_pred, 0, None)
    mae     = mean_absolute_error(y_true, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred))
    med_ae  = np.median(np.abs(y_true - y_pred))

    print(f"MAE    : {mae:>12,.0f} KB  ({mae/1024:.1f} MB)")
    print(f"RMSE   : {rmse:>12,.0f} KB  ({rmse/1024:.1f} MB)")
    print(f"Median : {med_ae:>12,.0f} KB  ({med_ae/1024:.1f} MB)")

    for tol_mb in [50, 100, 150, 200]:
        tol_kb = tol_mb * 1024
        pct    = np.mean(np.abs(y_true - y_pred) <= tol_kb) * 100
        print(f"Within ±{tol_mb}MB : {pct:.1f}%")

    return {'mae': mae, 'rmse': rmse, 'med_ae': med_ae}


def evaluate_by_group(df_test, y_pred, group_col):
    """
    Break down MAE by a categorical column.
    Shows where model performs well and where it struggles.
    """
    print(f"\nMAE by {group_col}:")
    print(f"  {'Group':<25} {'MAE (KB)':>12}  {'MAE (MB)':>10}  {'Rows':>6}")
    print(f"  {'-'*58}")

    results = []
    for group in sorted(df_test[group_col].unique()):
        mask  = df_test[group_col] == group
        y_g   = df_test[TARGET].values[mask]
        p_g   = y_pred[mask]
        mae_g = mean_absolute_error(y_g, p_g)
        results.append((group, mae_g, mask.sum()))
        print(f"  {str(group):<25} {mae_g:>12,.0f}  "
              f"{mae_g/1024:>10.1f}  {mask.sum():>6}")

    return results


def compare_baseline(y_true, y_pred, df_test):
    """
    Naive baseline: predict mean reclaimed_kb for every row.
    ML model must beat this to be justified.
    """
    print("\nBaseline comparison:")
    mean_pred  = np.full_like(y_true, float(y_true.mean()),
                               dtype=float)
    base_mae   = mean_absolute_error(y_true, mean_pred)
    base_rmse  = np.sqrt(mean_squared_error(y_true, mean_pred))
    model_mae  = mean_absolute_error(y_true, y_pred)
    model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    improvement_mae  = (base_mae  - model_mae)  / base_mae  * 100
    improvement_rmse = (base_rmse - model_rmse) / base_rmse * 100

    print(f"  {'':25} {'MAE (KB)':>12}  {'RMSE (KB)':>12}")
    print(f"  {'Naive mean baseline':<25} "
          f"{base_mae:>12,.0f}  {base_rmse:>12,.0f}")
    print(f"  {'XGBoost model':<25} "
          f"{model_mae:>12,.0f}  {model_rmse:>12,.0f}")
    print(f"  {'Improvement':<25} "
          f"{improvement_mae:>11.1f}%  {improvement_rmse:>11.1f}%")

    return {
        'baseline_mae' : base_mae,
        'model_mae'    : model_mae,
        'improvement'  : improvement_mae
    }


def evaluate(df_test, y_pred):
    """Run full evaluation."""
    print("\nEvaluation")
    print("=" * 50)

    y_true = df_test[TARGET].values
    y_pred = np.clip(y_pred, 0, None)

    print("\nGlobal Metrics:")
    metrics = evaluate_global(y_true, y_pred)

    evaluate_by_group(df_test, y_pred, 'game')
    evaluate_by_group(df_test, y_pred, 'device')
    evaluate_by_group(df_test, y_pred, 'graphics_setting')

    baseline = compare_baseline(y_true, y_pred, df_test)

    return metrics, baseline