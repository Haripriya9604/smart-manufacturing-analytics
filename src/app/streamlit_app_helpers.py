import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
from datetime import timedelta

# ======================================================
# Sensor definitions (CMAPSS)
# ======================================================
SENSORS = [f"s{i}" for i in range(1, 22)]

# ======================================================
# Model paths
# ======================================================
MODEL_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\models"

MODEL_MAP = {
    "FD001": "rul_model_FD001.joblib",
    "FD002": "rul_model_FD002.joblib",
    "FD003": "rul_model_FD003.joblib",
    "FD004": "rul_model_FD004.joblib",
    "global": "rul_model_global.joblib",
}

# ======================================================
# Auto model selector (SAFE)
# ======================================================
def auto_select_model(df: pd.DataFrame) -> str:
    """
    Selects the correct model based on dataset column.
    Falls back to global model if unknown.
    """
    if "dataset" in df.columns:
        ds = str(df["dataset"].iloc[0]).strip()
        return MODEL_MAP.get(ds, MODEL_MAP["global"])
    return MODEL_MAP["global"]

# ======================================================
# Load model
# ======================================================
def load_model(model_name: str):
    path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model_type, model, feats = joblib.load(path)
    return model_type, model, feats

# ======================================================
# Feature engineering (MATCHES TRAINING EXACTLY)
# ======================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -------------------------------
    # Required columns check
    # -------------------------------
    required = {"unit", "cycle"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Input data missing required columns: {missing}")

    # Ensure sensor columns exist
    for s in SENSORS:
        if s not in df.columns:
            df[s] = 0.0

    # -------------------------------
    # Rolling statistics
    # -------------------------------
    for w in [5, 10]:
        for s in SENSORS:
            df[f"{s}_rm_{w}"] = df.groupby("unit")[s].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{s}_rs_{w}"] = df.groupby("unit")[s].transform(
                lambda x: x.rolling(w, min_periods=1).std()
            ).fillna(0)

    # -------------------------------
    # Exponentially weighted means
    # -------------------------------
    for s in SENSORS:
        df[f"{s}_ewm_03"] = df.groupby("unit")[s].transform(
            lambda x: x.ewm(alpha=0.3).mean()
        )

    # -------------------------------
    # Cumulative drift
    # -------------------------------
    for s in SENSORS:
        df[f"{s}_cum_drift"] = df.groupby("unit")[s].transform(
            lambda x: x - x.iloc[0]
        )

    # -------------------------------
    # Cycle normalization
    # -------------------------------
    df["cycle_norm"] = df.groupby("unit")["cycle"].transform(
        lambda x: x / max(x.max(), 1)
    )
    df["cycle_norm2"] = df["cycle_norm"] ** 2
    df["cycle_norm3"] = df["cycle_norm"] ** 3

    # ===============================
    # Health Index (FIXED & SAFE)
    # ===============================
    sensor_min = df[SENSORS].min()
    sensor_max = df[SENSORS].max()

    hi_matrix = 1 - (df[SENSORS] - sensor_min) / (sensor_max - sensor_min + 1e-6)

    df["HI"] = hi_matrix.mean(axis=1)

    df["HI_smooth"] = df.groupby("unit")["HI"].transform(
        lambda x: x.ewm(alpha=0.2).mean()
    )

    df["HI_slope"] = df.groupby("unit")["HI"].diff().fillna(0)

    # 🔑 Training-compatible aliases
    df["health_index"] = df["HI"]
    df["health_smooth"] = df["HI_smooth"]

    # -------------------------------
    # Operational mode fallback
    # -------------------------------
    if "op_mode_id" not in df.columns:
        df["op_mode_id"] = 0

    return df

# ======================================================
# Predict RUL (ROBUST)
# ======================================================
def predict_rul(df_feat: pd.DataFrame, model_type, model, feats):
    """
    Safe prediction that never crashes due to missing features.
    """
    df = df_feat.copy()

    missing_feats = [f for f in feats if f not in df.columns]
    for f in missing_feats:
        df[f] = 0.0

    X = df[feats].fillna(0)

    if model_type == "xgb":
        return model.predict(xgb.DMatrix(X))

    return model.predict(X)

# ======================================================
# Confidence Interval (Residual-based)
# ======================================================
def add_confidence_interval(
    df: pd.DataFrame,
    pred_col: str = "RUL_pred",
    sigma: float = 12.0,   # <-- replace with validation residual std if available
    z: float = 1.96        # 95% confidence
):
    """
    Adds lower and upper confidence bounds for RUL.
    """
    df = df.copy()

    df["RUL_lower"] = df[pred_col] - z * sigma
    df["RUL_upper"] = df[pred_col] + z * sigma

    # Operator-safe: no negative lower bounds
    df["RUL_lower"] = df["RUL_lower"].clip(lower=0)

    return df

# ======================================================
# Failure date prediction (CALENDAR-BASED)
# ======================================================
def add_failure_date(
    df_latest: pd.DataFrame,
    reference_date: pd.Timestamp = None,
    cycle_to_days: int = 1
):
    """
    Converts RUL (cycles) into a calendar failure date.

    Assumptions:
    - 1 cycle ≈ 1 day (adjustable)
    - Negative RUL → failure already occurred (set to today)
    """

    df = df_latest.copy()

    if reference_date is None:
        reference_date = pd.Timestamp.today().normalize()

    # Non-negative days remaining
    df["Days_to_Failure"] = df["RUL_pred"].clip(lower=0)

    df["Failure_Date"] = df["Days_to_Failure"].apply(
        lambda d: reference_date + timedelta(days=int(d * cycle_to_days))
    )

    return df
