# =============================================================
#  src/explain.py
#  SHAP explainability for memory reclaim model.
#
#  Research questions answered:
#    1. Which features drive reclaim magnitude most?
#    2. Do feature importances differ across game types?
#    3. Does memory pressure or thermal state matter more?
# =============================================================

import sys
import os
import numpy  as np
import pandas as pd
import shap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_COLS, SHAP_SAMPLE_SIZE, RANDOM_SEED


def compute_shap(model, X, sample_size=None):
    """Compute SHAP values on a sample of test data."""
    sample_size = sample_size or SHAP_SAMPLE_SIZE
    sample      = X.sample(
        n=min(sample_size, len(X)),
        random_state=RANDOM_SEED
    )
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(sample)
    mean_abs   = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    return explainer, shap_vals, mean_abs, sample


def print_shap_importance(mean_abs, top_n=15):
    """Print SHAP feature importance table."""
    print(f"\nSHAP Feature Importance (top {top_n}):")
    print(f"  {'Rank':<5} {'Feature':<30} {'Mean |SHAP|':>12}")
    print(f"  {'-'*50}")
    for i, (feat, val) in enumerate(
            mean_abs.head(top_n).items(), 1):
        bar = "█" * int(val / mean_abs.max() * 20)
        print(f"  {i:<5} {feat:<30} {val:>12,.1f}  {bar}")


def research_findings(mean_abs):
    """
    Print the three core research findings
    from SHAP analysis.
    """
    print("\n🔬 Research Findings:")

    top1 = mean_abs.index[0]
    top2 = mean_abs.index[1]
    top3 = mean_abs.index[2]

    print(f"\n  Finding 1 — Dominant signal:")
    print(f"  '{top1}' is the strongest predictor")
    print(f"  of memory reclaim magnitude.")

    # Check if pressure score beats raw memory
    pressure_rank = list(mean_abs.index).index(
        'memory_pressure_score'
    ) + 1 if 'memory_pressure_score' in mean_abs.index else 99

    raw_rank = list(mean_abs.index).index(
        'memory_utilization_pct'
    ) + 1 if 'memory_utilization_pct' in mean_abs.index else 99

    print(f"\n  Finding 2 — Pressure vs utilization:")
    print(f"  memory_pressure_score rank : #{pressure_rank}")
    print(f"  memory_utilization_pct rank: #{raw_rank}")
    if pressure_rank < raw_rank:
        print(f"  → Combined pressure score outperforms")
        print(f"    raw utilization — interaction between")
        print(f"    memory and swap matters more than")
        print(f"    memory alone.")

    # Check thermal interaction
    thermal_rank = list(mean_abs.index).index(
        'thermal_memory_interaction'
    ) + 1 if 'thermal_memory_interaction' in mean_abs.index else 99

    print(f"\n  Finding 3 — Thermal influence:")
    print(f"  thermal_memory_interaction rank: #{thermal_rank}")
    if thermal_rank <= 10:
        print(f"  → Thermal state significantly influences")
        print(f"    reclaim magnitude — hot device under")
        print(f"    memory pressure reclaims more aggressively.")
    else:
        print(f"  → Thermal state has limited influence")
        print(f"    on reclaim magnitude compared to")
        print(f"    pure memory pressure signals.")


def explain(model, X_test):
    """Run full SHAP analysis."""
    print("\nSHAP Explainability")
    print("=" * 50)

    explainer, shap_vals, mean_abs, sample = compute_shap(
        model, X_test
    )
    print_shap_importance(mean_abs)
    research_findings(mean_abs)

    return explainer, shap_vals, mean_abs