# src/models/tune_fd.py
import os
import sys
import joblib
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

PROCESSED_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\processed\cmapss"
MODEL_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\models"
CLIP_RUL = 200

def select_features(df):
    SENSOR_COLS = [f"s{i}" for i in range(1,22)]
    feats = []
    for c in df.columns:
        if (
            c in SENSOR_COLS
            or c == "cycle_norm"
            or c.endswith("_rm_5") or c.endswith("_rm_10") or c.endswith("_rm_20")
            or c.endswith("_rs_5") or c.endswith("_rs_10")
            or c.endswith("_ewm_03")
        ):
            feats.append(c)
    return feats[:150]

def make_weights(y, alpha=1.0):
    max_y = float(np.max(y))
    return 1 + alpha * (1 - y / max_y)

def objective(trial, df, X, y_true, y_train, groups):
    import xgboost as xgb
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, val_idx = next(gss.split(X, y_train, groups=groups))

    Xtr = X.iloc[tr_idx]
    Xval = X.iloc[val_idx]
    ytr = y_train[tr_idx]
    yval = y_true[val_idx]

    w_all = make_weights(y_train)
    wtr = w_all[tr_idx]
    wval = w_all[val_idx]

    params = {
        "objective": "reg:squarederror",
        "eta": trial.suggest_float("eta", 0.005, 0.05),
        "max_depth": trial.suggest_int("max_depth", 5, 14),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0),
        "lambda": trial.suggest_float("lambda", 0.0, 3.0),
        "alpha": trial.suggest_float("alpha", 0.0, 3.0),
        "seed": 42,
        "tree_method": "auto"
    }

    num_rounds = trial.suggest_int("num_boost_round", 300, 1300)

    dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dval = xgb.DMatrix(Xval, label=yval, weight=wval)

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    pred = bst.predict(xgb.DMatrix(Xval))
    return mean_absolute_error(yval, pred)

def main():
    if len(sys.argv) < 2:
        print("Usage: python tune_fd.py FD003")
        sys.exit(1)

    target_fd = sys.argv[1].strip().upper()
    train_path = os.path.join(PROCESSED_DIR, "train_all_features.csv")

    df = pd.read_csv(train_path)
    df_fd = df[df["dataset"] == target_fd].reset_index(drop=True)

    if df_fd.empty:
        print("No rows found for:", target_fd)
        sys.exit(1)

    feats = select_features(df_fd)
    X = df_fd[feats]
    y_true = df_fd["RUL"].values
    y_train = np.clip(y_true, None, CLIP_RUL)
    groups = df_fd["unit"].values

    print("Starting Optuna tuning for", target_fd, "rows:", len(df_fd))

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, df_fd, X, y_true, y_train, groups), n_trials=40)

    print("Best trial:", study.best_trial.value)
    print("Best params:", study.best_params)

    # train final model with best params
    best_params = study.best_params
    num_rounds = best_params.pop("num_boost_round")
    import xgboost as xgb
    dtrain = xgb.DMatrix(X, label=y_train, weight=make_weights(y_train))
    bst = xgb.train(best_params, dtrain, num_boost_round=num_rounds)

    out = os.path.join(MODEL_DIR, f"rul_model_{target_fd}_tuned.joblib")
