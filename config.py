# =============================================================
#  config.py
#  Central configuration for memory reclaim prediction project.
#  All paths, constants, and hyperparameters live here.
# =============================================================

import os

# -------------------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------------------

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

DATA_RAW        = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED  = os.path.join(BASE_DIR, 'data', 'processed')
DATA_FEATURES   = os.path.join(BASE_DIR, 'data', 'features')
MODELS_DIR      = os.path.join(BASE_DIR, 'models')
OUTPUTS_PLOTS   = os.path.join(BASE_DIR, 'outputs', 'plots')
OUTPUTS_RESULTS = os.path.join(BASE_DIR, 'outputs', 'results')

# File paths
RAW_DATA_FILE       = os.path.join(DATA_RAW,       'memory_reclaim_60k.csv')
PROCESSED_DATA_FILE = os.path.join(DATA_PROCESSED, 'memory_reclaim_cleaned.csv')
FEATURES_DATA_FILE  = os.path.join(DATA_FEATURES,  'memory_reclaim_final.csv')
IQR_BOUNDS_FILE     = os.path.join(DATA_FEATURES,  'iqr_bounds.json')
REG_MODEL_FILE      = os.path.join(MODELS_DIR,     'xgb_regressor.pkl')
SCALER_FILE         = os.path.join(MODELS_DIR,     'scaler.pkl')

# -------------------------------------------------------------
# DATASET CONSTANTS
# -------------------------------------------------------------

RANDOM_SEED  = 42
TRAIN_SPLIT  = 0.80     # 80% sessions train, 20% test

# -------------------------------------------------------------
# PHYSICAL CLIP RULES — raw KB columns
# Values outside these ranges are sensor errors
# -------------------------------------------------------------

CLIP_RULES = {
    'memtotal_kb'    : (1_000_000,  13_000_000),
    'swaptotal_kb'   : (0,           7_000_000),
    'swapfree_kb'    : (0,           7_000_000),
    'dma_buf_kb'     : (0,           2_000_000),
    'heapsize_kb'    : (0,           2_500_000),
    'heapmaxfree_kb' : (0,           2_500_000),
    'heapminfree_kb' : (0,             500_000),
    'dha_empty_kb'   : (0,           3_000_000),
    'dha_cache_kb'   : (0,           3_000_000),
    'currentcpuload' : (0,               100.0),
    'aptemp'         : (25.0,             90.0),
    'apk_size_mb'    : (10,              5_000),
    'duration_sec'   : (60,              5_400),
    'reclaimed_kb'   : (0,           1_000_000),
}

# -------------------------------------------------------------
# CATEGORICAL COLUMNS AND THEIR VALID VALUES
# -------------------------------------------------------------

GRAPHICS_OPTIONS     = ['low', 'medium', 'high', 'ultra']
COMP_FILTER_OPTIONS  = ['speed', 'everything', 'verify',
                        'quicken', 'speed-profile']
ODEX_OPTIONS         = ['odex+vdex', 'vdex_only',
                        'odex_only', 'none']
MULTIUSER_OPTIONS    = ['multiuser', 'singleuser']
ONLINE_OPTIONS       = ['online', 'offline']

# Encoding maps — applied in preprocess.py
GRAPHICS_MAP = {'low': 0, 'medium': 1, 'high': 2, 'ultra': 3}

COMP_FILTER_MAP = {
    'verify'        : 0,
    'quicken'       : 1,
    'speed-profile' : 2,
    'speed'         : 3,
    'everything'    : 4,
}

ODEX_MAP = {
    'none'      : 0,
    'vdex_only' : 1,
    'odex_only' : 2,
    'odex+vdex' : 3,
}

# Game memory pressure category
# Higher = more memory intensive
GAME_PRESSURE_MAP = {
    'CandyCrush'     : 0,
    'AngryBird'      : 1,
    'HillClimbRacing': 2,
    'SubwaySurfer'   : 3,
    'LudoKing'       : 4,
    'EightBallPool'  : 5,
    'COC'            : 6,
    'WCC3'           : 7,
    'FreeFire'       : 8,
    'eFootball'      : 9,
    'CODM'           : 10,
    'BGMI'           : 11,
}

GAME_CATEGORY_MAP = {
    'casual'  : 0,
    'strategy': 1,
    'sports'  : 2,
    'action'  : 3,
}

DEVICE_RAM_MAP = {
    'Samsung_A_Series': 6,
    'Samsung_F_Series': 8,
    'Samsung_M_Series': 8,
    'Samsung_S_Series': 12,
}

# -------------------------------------------------------------
# ENGINEERED FEATURES — IQR cap applied to these
# -------------------------------------------------------------

IQR_FEATURES = [
    'memory_used_kb',
    'memory_utilization_pct',
    'swap_used_kb',
    'swap_utilization_pct',
    'heap_utilization_pct',
    'heap_free_ratio',
    'dha_total_kb',
    'dha_cache_ratio',
    'memory_pressure_score',
]

# -------------------------------------------------------------
# FEATURE COLUMNS — used for model training
# -------------------------------------------------------------

RAW_FEATURES = [
    'memtotal_kb',
    'swaptotal_kb',
    'swapfree_kb',
    'dma_buf_kb',
    'heapsize_kb',
    'heapmaxfree_kb',
    'heapminfree_kb',
    'dha_empty_kb',
    'dha_cache_kb',
    'currentcpuload',
    'aptemp',
    'apk_size_mb',
    'duration_sec',
]

ENCODED_FEATURES = [
    'graphics_encoded',
    'compilation_filter_encoded',
    'odex_encoded',
    'multiuser_encoded',
    'online_encoded',
    'game_pressure_encoded',
    'game_category_encoded',
    'device_ram_gb',
    'refreshrate',
]

ENGINEERED_FEATURES = [
    'memory_used_kb',
    'memory_utilization_pct',
    'swap_used_kb',
    'swap_utilization_pct',
    'heap_utilization_pct',
    'heap_free_ratio',
    'dha_total_kb',
    'dha_cache_ratio',
    'memory_pressure_score',
    'thermal_memory_interaction',
    'cpu_memory_stress',
    'ram_headroom_kb',
]

FEATURE_COLS = RAW_FEATURES + ENCODED_FEATURES + ENGINEERED_FEATURES

# Target column
TARGET = 'reclaimed_kb'

# -------------------------------------------------------------
# XGBOOST HYPERPARAMETERS
# -------------------------------------------------------------

XGB_PARAMS = {
    'n_estimators'      : 500,
    'learning_rate'     : 0.05,
    'max_depth'         : 6,
    'min_child_weight'  : 10,
    'subsample'         : 0.8,
    'colsample_bytree'  : 0.8,
    'reg_alpha'         : 0.1,
    'reg_lambda'        : 1.0,
    'objective'         : 'reg:squarederror',
    'random_state'      : RANDOM_SEED,
    'verbosity'         : 0,
    'n_jobs'            : -1,
}

EARLY_STOPPING = 50
CV_FOLDS       = 5

# -------------------------------------------------------------
# SHAP
# -------------------------------------------------------------

SHAP_SAMPLE_SIZE = 2000