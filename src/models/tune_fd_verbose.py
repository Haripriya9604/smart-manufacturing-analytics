# src/models/tune_fd_verbose.py
"""
Verbose Optuna tuner for a single FD dataset.
Safer: prints progress, exceptions, and saves a tuned model named rul_model_{FD}_tuned.joblib
Uses fewer trials by default for speed; adjust n_trials.
"""
import os, sys, traceback, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

# safe defaults (edit only if you know what you're doing)
PROCESSED_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\processed\cmapss"
MODEL_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\models"
TRAIN_FEAT = os.path.join(PROCESSED_DIR, "train_all_features.csv")
CLIP_RUL = 200
SEED = 42

def select_features(df):
    SENSOR_COLS = [f"s{i}" for i in range(1,22)]
    feats = []
    for c in df.columns:
        if (c in SENSOR_COLS or c == "cycle_norm" or c.endswith("_rm_5") or c.endswith("_rm_10") or c.endswith("_rm_20")
            or c.endswith("_rs_5") or c.endswith("_rs_10") or c.endswith("_ewm_03") or c.startswith("s")):
            feats.append(c)
    # preserve order, cap features for memory
    return [f for f in feats if f in df.columns][:200]

def make_weights(y, alpha=1.0):
    max_y = float(np.max(y))
    if max_y <= 0:
        return np.ones_like(y, dtype=float)
    return 1.0 + alpha * (1.0 - y / (max_y + 1e-9))

def train_with_params(X, y_train, y_true, groups, params, num_rounds):
    import xgboost as xgb
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, val_idx = next(gss.split(X, y_train, groups=groups))
    Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
    ytr, yval_true = y_train[tr_idx], y_true[val_idx]
    w_all = make_weights(y_train)
    dtrain = xgb.DMatrix(Xtr, label=ytr, weight=w_all[tr_idx])
    dval = xgb.DMatrix(Xval, label=yval_true, weight=w_all[val_idx])
    bst = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=[(dval,"val")], early_stopping_rounds=50, verbose_eval=False)
    pred = bst.predict(xgb.DMatrix(Xval))
    return mean_absolute_error(yval_true, pred), bst

def run_search(target_fd, n_trials=20):
    try:
        import optuna
    except Exception as e:
        print("Optuna is not installed or failed to import:", e)
        raise

    print("Loading train features:", TRAIN_FEAT)
    df = pd.read_csv(TRAIN_FEAT)
    df_fd = df[df["dataset"] == target_fd].reset_index(drop=True)
    if df_fd.shape[0] == 0:
        raise RuntimeError(f"No rows for dataset {target_fd} in {TRAIN_FEAT}")

    feats = select_features(df_fd)
    X = df_fd[feats]
    y_true = df_fd["RUL"].values
    y_train = np.clip(y_true, None, CLIP_RUL)
    groups = df_fd["unit"].values

    print(f"Starting Optuna for {target_fd}: rows={len(df_fd)}, features={len(feats)}, trials={n_trials}")

    def objective(trial):
        import xgboost as xgb
        params = {
            "objective": "reg:squarederror",
            "eta": trial.suggest_float("eta", 0.005, 0.05),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 5.0),
            "lambda": trial.suggest_float("lambda", 0.0, 3.0),
            "alpha": trial.suggest_float("alpha", 0.0, 3.0),
            "seed": SEED
        }
        num_rounds = trial.suggest_int("num_rounds", 300, 1200)
        mae, _ = train_with_params(X, y_train, y_true, groups, params, num_rounds)
        return mae

    study = optuna.create_study(direction="minimize")
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except Exception as ex:
        print("Optuna raised an exception during optimization:")
        traceback.print_exc()
        raise

    print("Optimization finished.")
    print("Best MAE:", study.best_value)
    print("Best params:", study.best_trial.params)

    # retrain final model on full FD with best params
    best = study.best_trial.params
    num_rounds = best.pop("num_rounds") if "num_rounds" in best else 600
    import xgboost as xgb
    dtrain = xgb.DMatrix(X, label=y_train, weight=make_weights(y_train))
    final_bst = xgb.train(best, dtrain, num_boost_round=num_rounds)
    out = os.path.join(MODEL_DIR, f"rul_model_{target_fd}_tuned.joblib")
    joblib.dump(("xgb", final_bst, feats), out)
    print("Saved tuned model to:", out)
    return study.best_value, study.best_trial.params, out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tune_fd_verbose.py FD003 [n_trials]")
        sys.exit(1)
    target = sys.argv[1].upper()
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    try:
        best_mae, best_params, outpath = run_search(target, n_trials=trials)
        print("Done. Best MAE:", best_mae)
    except Exception as e:
        print("Tuner failed with exception:")
        traceback.print_exc()
        sys.exit(2)
