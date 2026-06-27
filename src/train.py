# =============================================================
#  src/train.py
#  Trains XGBoost regressor for memory reclaim prediction.
#
#  Predicts: how many KB of memory will be reclaimed?
#
#  Key design decisions:
#    Session-based split — no session appears in both
#    train and test sets (prevents leakage)
#    TimeSeriesSplit cross-validation — respects time order
#    StandardScaler fitted on train only
#
#  Input  : data/features/memory_reclaim_final.csv
#  Output : models/xgb_regressor.pkl
#           models/scaler.pkl
# =============================================================

import os
import sys
import pickle
import numpy  as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import mean_absolute_error, mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FEATURES_DATA_FILE, REG_MODEL_FILE, SCALER_FILE,
    MODELS_DIR, FEATURE_COLS, TARGET,
    XGB_PARAMS, TRAIN_SPLIT, CV_FOLDS, EARLY_STOPPING,
    RANDOM_SEED
)


def load_data():
    df = pd.read_csv(FEATURES_DATA_FILE)
    print(f"Loaded {len(df):,} rows")
    print(f"Target range: {df[TARGET].min():,} → {df[TARGET].max():,} KB")
    return df


def session_split(df):
    """
    Split by sessions not rows.
    First 80% of sessions → train.
    Last 20% of sessions  → test.
    No session appears in both.
    """
    sessions   = df['session_id'].unique().tolist()
    n_train    = int(len(sessions) * TRAIN_SPLIT)
    train_sess = sessions[:n_train]
    test_sess  = sessions[n_train:]

    df_train = df[df['session_id'].isin(train_sess)].copy()
    df_test  = df[df['session_id'].isin(test_sess)].copy()

    print(f"Train: {len(train_sess)} sessions, {len(df_train):,} rows")
    print(f"Test : {len(test_sess)} sessions, {len(df_test):,} rows")

    # Verify no overlap
    assert len(set(train_sess) & set(test_sess)) == 0
    print("No session overlap confirmed")

    return df_train, df_test


def scale(df_train, df_test):
    """
    Fit StandardScaler on train only.
    Apply same scaler to test.
    """
    scaler         = StandardScaler()
    X_train        = df_train[FEATURE_COLS]
    X_test         = df_test[FEATURE_COLS]

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS
    )
    X_test_scaled  = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS
    )
    print("Features scaled (train-fitted scaler)")
    return X_train_scaled, X_test_scaled, scaler


def cross_validate(X_train, y_train):
    """
    TimeSeriesSplit cross-validation.
    Validation always comes after training in time.
    Reports MAE and RMSE per fold.
    """
    print(f"\nCross-validation ({CV_FOLDS} folds, TimeSeriesSplit)...")
    tscv      = TimeSeriesSplit(n_splits=CV_FOLDS)
    mae_list  = []
    rmse_list = []

    for fold, (tr_idx, val_idx) in enumerate(
            tscv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx],      y_train[val_idx]

        m = xgb.XGBRegressor(
        **XGB_PARAMS,
        early_stopping_rounds=EARLY_STOPPING
)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False)

        pred = np.clip(m.predict(X_val), 0, None)
        mae  = mean_absolute_error(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        mae_list.append(mae)
        rmse_list.append(rmse)
        print(f"  Fold {fold}: MAE={mae:>10,.0f} KB  "
              f"RMSE={rmse:>10,.0f} KB")

    print(f"\n  Mean MAE  : {np.mean(mae_list):>10,.0f} KB "
          f"± {np.std(mae_list):,.0f}")
    print(f"  Mean RMSE : {np.mean(rmse_list):>10,.0f} KB "
          f"± {np.std(rmse_list):,.0f}")
    return np.mean(mae_list)


def train_final(X_train, y_train):
    """
    Train final model on full training set.
    Use last 10% of train rows for early stopping only.
    """
    cutoff = int(len(X_train) * 0.90)
    model = xgb.XGBRegressor(
    **XGB_PARAMS,
    early_stopping_rounds=EARLY_STOPPING
    )
    model.fit(
        X_train.iloc[:cutoff], y_train[:cutoff],
        eval_set = [(X_train.iloc[cutoff:], y_train[cutoff:])],
        verbose  = False
    )
    print(f"Best iteration: {model.best_iteration}")
    return model


def train(save=True):
    """Run full training pipeline."""
    print("\nTraining XGBoost Regressor")
    print("=" * 50)

    df                       = load_data()
    df_train, df_test        = session_split(df)
    X_train, X_test, scaler  = scale(df_train, df_test)
    y_train                  = df_train[TARGET].values
    y_test                   = df_test[TARGET].values

    cross_validate(X_train, y_train)

    print("\nTraining final model...")
    model = train_final(X_train, y_train)
    print("Training complete")

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(REG_MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Saved → {REG_MODEL_FILE}")
        print(f"Saved → {SCALER_FILE}")

    return model, scaler, df_train, df_test, \
           X_train, X_test, y_train, y_test


if __name__ == '__main__':
    train()