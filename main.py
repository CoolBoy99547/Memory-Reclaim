# =============================================================
#  main.py
#  Runs complete memory reclaim prediction pipeline.
#  One command: python main.py
# =============================================================

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess          import preprocess
from src.feature_engineering import engineer_features
from src.train               import train
from src.evaluate            import evaluate
from src.explain             import explain
from config                  import FEATURE_COLS, TARGET


if __name__ == '__main__':

    print("=" * 60)
    print("  MEMORY RECLAIM PREDICTION — FULL PIPELINE")
    print("=" * 60)

    # Step 1: Preprocess
    df_clean = preprocess()

    # Step 2: Feature engineering
    df_final, iqr_bounds = engineer_features(df_clean)

    # Step 3: Train
    (model, scaler,
     df_train, df_test,
     X_train, X_test,
     y_train, y_test) = train()

    # Step 4: Evaluate
    y_pred = np.clip(model.predict(X_test), 0, None)
    metrics, baseline = evaluate(df_test, y_pred)

    # Step 5: Explain
    explainer, shap_vals, mean_abs = explain(model, X_test)

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"""
  Dataset rows   : {len(df_final):,}
  Features used  : {len(FEATURE_COLS)}
  Train sessions : {df_train['session_id'].nunique()}
  Test  sessions : {df_test['session_id'].nunique()}

  MAE            : {metrics['mae']:>10,.0f} KB
                   ({metrics['mae']/1024:.1f} MB)
  RMSE           : {metrics['rmse']:>10,.0f} KB
                   ({metrics['rmse']/1024:.1f} MB)
  Median Error   : {metrics['med_ae']:>10,.0f} KB
                   ({metrics['med_ae']/1024:.1f} MB)

  Baseline MAE   : {baseline['baseline_mae']:>10,.0f} KB
  Improvement    : {baseline['improvement']:.1f}%

  Top SHAP feature: {mean_abs.index[0]}
    """)
    print("PIPELINE COMPLETE")
    print("=" * 60)