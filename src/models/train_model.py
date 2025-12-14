# src/models/train_model.py
"""
Improved training script:
- Explicitly includes health-index & op-mode features
- Uses all meaningful drift/rolling/ewm statistical features
- Strengthened FD003/FD004 hyperparameters
- Clipped RUL target + sample weighting
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

PROCESSED_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\processed\cmapss"
MODEL_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLIP_RUL = 200
ALPHA_WEIGHT = 1.0
TRAIN_PER_FD = True
SEED = 42


# ===============================================================
# FEATURE SELECTION (UPDATED)
# ===============================================================
def select_features(df):
    """Select all important engineered features including HI & op-mode."""
    
    # raw sensors
    base_sensors = [f"s{i}" for i in range(1, 22) if f"s{i}" in df.columns]

    # built statistical features
    roll = [c for c in df.columns if "_rm_" in c]        # rolling mean
    std = [c for c in df.columns if "_rs_" in c]         # rolling std
    ewm = [c for c in df.columns if "_ewm_" in c]        # exponential smoothing
    drift = [c for c in df.columns if "_drift" in c]     # cumulative drift
    diffs = [c for c in df.columns if c.endswith("_diff")]

    # cycle normalized
    cycle_feats = [c for c in df.columns if c.startswith("cycle_norm")]

    # health features (critical for FD004)
    health_feats = [
        "HI", "HI_smooth", "HI_slope",
        "health_index", "health_smooth",
        "op_mode_id"
    ]
    health_feats = [h for h in health_feats if h in df.columns]

    # Combine
    feats = (
        base_sensors +
        roll + std + ewm + drift + diffs +
        cycle_feats +
        health_feats
    )

    # Deduplicate and keep order
    feats = list(dict.fromkeys([f for f in feats if f in df.columns]))

    print(f"[Feature Select] Using {len(feats)} features")
    return feats


# ===============================================================
# SAMPLE WEIGHTING
# ===============================================================
def make_weights(y_array, alpha=ALPHA_WEIGHT):
    """Higher weight for low-RUL samples."""
    max_r = float(np.max(y_array))
    scaled = 1.0 - (y_array / (max_r + 1e-9))
    return 1.0 + alpha * scaled


# ===============================================================
# TRAIN ONE DATASET
# ===============================================================
def train_one(df, suffix):
    feats = select_features(df)
    X = df[feats]
    y_true = df["RUL"].values
    y_clip = np.clip(y_true, None, CLIP_RUL)
    groups = df["unit"].values

    # train/val split (grouped by unit)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, val_idx = next(gss.split(X, y_clip, groups=groups))

    Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
    ytr, yval = y_clip[tr_idx], y_true[val_idx]

    w = make_weights(y_clip)
    wtr, wval = w[tr_idx], w[val_idx]

    # per-dataset hyperparameters
    params_map = {
        "FD001": {"objective": "reg:squarederror", "eta": 0.04, "max_depth": 6, "subsample": 0.85, "colsample_bytree": 0.85, "seed": SEED},
        "FD002": {"objective": "reg:squarederror", "eta": 0.04, "max_depth": 6, "subsample": 0.85, "colsample_bytree": 0.85, "seed": SEED},

        # stronger models for hardest datasets
        "FD003": {"objective": "reg:squarederror", "eta": 0.015, "max_depth": 10, "subsample": 0.95, "colsample_bytree": 0.95,
                  "min_child_weight": 2.0, "lambda": 2.0, "alpha": 1.0, "seed": SEED},

        "FD004": {"objective": "reg:squarederror", "eta": 0.012, "max_depth": 10, "subsample": 0.97, "colsample_bytree": 0.97,
                  "min_child_weight": 2.5, "lambda": 2.5, "alpha": 1.0, "seed": SEED},

        # global backup
        "global": {"objective": "reg:squarederror", "eta": 0.025, "max_depth": 8, "subsample": 0.9, "colsample_bytree": 0.9, "seed": SEED},
    }

    rounds_map = {
        "FD001": 450,
        "FD002": 450,
        "FD003": 1000,
        "FD004": 1200,
        "global": 800
    }

    key = suffix if suffix in params_map else "global"
    params = params_map[key]
    num_rounds = rounds_map[key]

    # TRAIN XGBOOST
    try:
        import xgboost as xgb
        dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
        dval = xgb.DMatrix(Xval, label=yval, weight=wval)

        model = xgb.train(
            params,
            dtr,
            num_boost_round=num_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=50
        )

        preds = model.predict(xgb.DMatrix(Xval))
        model_type = "xgb"

    except Exception as e:
        print(f"[WARN] XGBoost failed for {suffix}: {e}")
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=500, max_depth=20, n_jobs=-1, random_state=SEED
        )
        model.fit(Xtr, ytr, sample_weight=wtr)
        preds = model.predict(Xval)
        model_type = "rf"

    # EVALUATION
    val_mae = mean_absolute_error(yval, preds)
    out_path = os.path.join(MODEL_DIR, f"rul_model_{suffix}.joblib")
    joblib.dump((model_type, model, feats), out_path)

    print(f"[{suffix}] MAE={val_mae:.3f}  saved → {out_path}")
    return float(val_mae)


# ===============================================================
# MAIN CONTROLLER
# ===============================================================
def main():
    train_file = os.path.join(PROCESSED_DIR, "train_all_features.csv")
    df = pd.read_csv(train_file)

    results = {}

    if TRAIN_PER_FD:
        for ds in sorted(df["dataset"].unique()):
            print(f"\n=== Training {ds} ===")
            df_sub = df[df["dataset"] == ds].reset_index(drop=True)
            results[ds] = train_one(df_sub, ds)

        print("\n=== Training GLOBAL model ===")
        results["global"] = train_one(df, "global")

    print("\nFinal Validation MAEs:", results)


if __name__ == "__main__":
    main()
