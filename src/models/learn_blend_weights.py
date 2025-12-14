# src/models/learn_blend_weights.py
"""
Learn per-dataset blend weights between per-FD model and global model.
We split train_all_features by unit (GroupShuffleSplit) to create a holdout,
predict with saved models on the holdout, then grid-search weight w in [0,1]
to minimize MAE: blended = w*pred_fd + (1-w)*pred_global.
Saves JSON weights to models/blend_weights.json
"""
import os, joblib, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

PROCESSED_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\processed\cmapss"
MODEL_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\models"
TRAIN_FEAT = os.path.join(PROCESSED_DIR, "train_all_features.csv")
OUT_WEIGHTS = os.path.join(MODEL_DIR, "blend_weights.json")

def load_model(path):
    return joblib.load(path)

def predict_model(model_tuple, df_slice):
    model_type, model_obj, feats = model_tuple
    X = df_slice[feats]
    if model_type == "lgb":
        return model_obj.predict(X, num_iteration=getattr(model_obj, "best_iteration", None))
    elif model_type == "xgb":
        import xgboost as xgb
        return model_obj.predict(xgb.DMatrix(X))
    else:
        return model_obj.predict(X)

def grid_search_weight(y_true, pred_fd, pred_glob):
    # grid over 0..1 step 0.01
    ws = np.linspace(0,1,101)
    best_w, best_mae = 0.0, float("inf")
    for w in ws:
        blended = w*pred_fd + (1-w)*pred_glob
        mae = mean_absolute_error(y_true, blended)
        if mae < best_mae:
            best_mae = mae
            best_w = float(w)
    return best_w, float(best_mae)

def main(test_size=0.15, random_state=42):
    df = pd.read_csv(TRAIN_FEAT)
    groups = df["unit"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    _, hold_idx = next(gss.split(df, df["RUL"].values, groups=groups))
    hold = df.iloc[hold_idx].reset_index(drop=True)

    # load global model and per-FD if present
    global_path = os.path.join(MODEL_DIR, "rul_model_global.joblib")
    if not os.path.exists(global_path):
        raise FileNotFoundError("Global model not found; train global first.")
    global_model = load_model(global_path)

    weights = {}
    for ds in sorted(hold["dataset"].unique()):
        sub = hold[hold["dataset"]==ds].reset_index(drop=True)
        per_path = os.path.join(MODEL_DIR, f"rul_model_{ds}.joblib")
        if not os.path.exists(per_path):
            # if per-FD model not present, weight=0 (use global)
            weights[ds] = {"w": 0.0, "mae_hold": None}
            continue
        per_model = load_model(per_path)
        # predictions
        pred_fd = predict_model(per_model, sub)
        pred_glob = predict_model(global_model, sub)
        y = sub["RUL"].values
        best_w, best_mae = grid_search_weight(y, pred_fd, pred_glob)
        weights[ds] = {"w": best_w, "mae_hold": best_mae}
        print(f"[{ds}] best_w={best_w}, hold_MAE={best_mae:.4f}, n={len(y)}")

    # save JSON
    with open(OUT_WEIGHTS, "w") as f:
        json.dump(weights, f, indent=2)
    print("Saved blend weights to", OUT_WEIGHTS)
    return weights

if __name__ == "__main__":
    main()
