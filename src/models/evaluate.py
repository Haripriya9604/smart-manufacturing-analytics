# src/models/evaluate.py
import os, joblib, json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support

PROCESSED_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\processed\cmapss"
MODEL_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\models"
ALERT_THRESH = 30
WEIGHTS_PATH = os.path.join(MODEL_DIR, "blend_weights.json")

def load_model(path):
    return joblib.load(path)

def predict_with_model(model_tuple, X):
    model_type, model_obj, feats = model_tuple
    Xsub = X[feats]
    if model_type == "lgb":
        return model_obj.predict(Xsub, num_iteration=getattr(model_obj, "best_iteration", None))
    elif model_type == "xgb":
        import xgboost as xgb
        return model_obj.predict(xgboost_DMatrix_safe(Xsub))
    else:
        return model_obj.predict(Xsub)

def xgboost_DMatrix_safe(X):
    import xgboost as xgb
    return xgb.DMatrix(X)

def main():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "test_all_features.csv"))
    global_model = load_model(os.path.join(MODEL_DIR, "rul_model_global.joblib"))
    # load blend weights if available
    weights = {}
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH) as f:
            weights = json.load(f)

    all_preds = []
    results = {}
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"]==ds].reset_index(drop=True)
        per_path = os.path.join(MODEL_DIR, f"rul_model_{ds}.joblib")
        per_model = load_model(per_path) if os.path.exists(per_path) else None

        pred_fd = predict_with_model(per_model, sub) if per_model is not None else None
        pred_glob = predict_with_model(global_model, sub) if global_model is not None else None

        # decide blend weight
        w = 0.0
        if ds in weights and weights[ds].get("w") is not None:
            w = float(weights[ds]["w"])
        else:
            # fallback heuristic: higher trust to per-FD for short pred
            w = None

        if pred_fd is not None and pred_glob is not None:
            if w is not None:
                ypred = w*pred_fd + (1.0-w)*pred_glob
            else:
                # fallback: dynamic per-sample rule (previous heuristic)
                blended = []
                for pf, pg in zip(pred_fd, pred_glob):
                    if pf <= 40:
                        ww = 0.8
                    else:
                        ww = 0.4
                    blended.append(ww*pf + (1-ww)*pg)
                ypred = np.array(blended)
        elif pred_fd is not None:
            ypred = pred_fd
        elif pred_glob is not None:
            ypred = pred_glob
        else:
            raise FileNotFoundError("No model available for evaluation.")

        y = sub["RUL"].values
        mae = mean_absolute_error(y, ypred)
        rmse = np.sqrt(mean_squared_error(y, ypred))
        r2 = r2_score(y, ypred)
        mask = y <= ALERT_THRESH
        mae_crit = float(mean_absolute_error(y[mask], ypred[mask])) if mask.sum()>0 else None
        y_true_alert = (y <= ALERT_THRESH).astype(int)
        y_pred_alert = (ypred <= ALERT_THRESH).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_alert, y_pred_alert, average='binary', zero_division=0)
        results[ds] = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "MAE_RUL_le_30": mae_crit, "precision": float(precision), "recall": float(recall), "f1": float(f1), "n": int(len(y)), "blend_w": w}
        print(f"[{ds}] Test MAE: {mae:.3f}, MAE<=30: {mae_crit}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, blend_w: {w}")
        sub["RUL_pred"] = ypred
        all_preds.append(sub)

    df_all = pd.concat(all_preds, ignore_index=True)
    agg = {"MAE": float(mean_absolute_error(df_all["RUL"], df_all["RUL_pred"])),
           "MAE_RUL_le_30": float(mean_absolute_error(df_all[df_all["RUL"]<=ALERT_THRESH]["RUL"], df_all[df_all["RUL"]<=ALERT_THRESH]["RUL_pred"])) if (df_all["RUL"]<=ALERT_THRESH).sum()>0 else None}
    print("GLOBAL aggregate:", agg)
    out = os.path.join(MODEL_DIR, "test_with_preds.csv")
    df_all.to_csv(out, index=False)
    print("Saved preds to", out)
    print("Per-dataset summary:", results)

if __name__ == "__main__":
    main()
