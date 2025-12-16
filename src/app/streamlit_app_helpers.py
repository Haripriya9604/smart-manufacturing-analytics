import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path
from datetime import timedelta

# ======================================================
# Project paths (CLOUD + LOCAL SAFE)
# ======================================================
# File location: src/app/streamlit_app_helpers.py
# PROJECT ROOT is 2 levels up → src/app → smart-manufacturing-analytics
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

# ======================================================
# Sensor definitions (CMAPSS)
# ======================================================
SENSORS = [f"s{i}" for i in range(1, 22)]

# ======================================================
# Model map
# ======================================================
MODEL_MAP = {
    "FD001": "rul_model_FD001.joblib",
    "FD002": "rul_model_FD002.joblib",
    "FD003": "rul_model_FD003.joblib",
    "FD004": "rul_model_FD004.joblib",
    "global": "rul_model_global.joblib",
}

# ======================================================
# Auto model selector
# ======================================================
def auto_select_model(df: pd.DataFrame) -> str:
    if "dataset" in df.columns:
        ds = str(df["dataset"].iloc[0]).strip()
        return MODEL_MAP.get(ds, MODEL_MAP["global"])
    return MODEL_MAP["global"]

# ======================================================
# Load model (DEPLOYMENT SAFE)
# ======================================================
def load_model(model_name: str):
    model_path = MODEL_DIR / model_name

    if not model_path.exists():
        available = [m.name for m in MODEL_DIR.glob("*.joblib")]
        raise FileNotFoundError(
            f"\n❌ Model not found: {model_path}"
            f"\n📂 MODEL_DIR resolved to: {MODEL_DIR}"
            f"\n📦 Available models: {available}\n"
        )

    model_type, model, feats = joblib.load(model_path)
    return model_type, model, feats

# ======================================================
# Feature engineering (MATCHES TRAINING)
# ======================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = {"unit", "cycle"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    for s in SENSORS:
        if s not in df.columns:
            df[s] = 0.0

    # Rolling stats
    for w in [5, 10]:
        for s in SENSORS:
            df[f"{s}_rm_{w}"] = df.groupby("unit")[s].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{s}_rs_{w}"] = df.groupby("unit")[s].transform(
                lambda x: x.rolling(w, min_periods=1).std()
            ).fillna(0)

    # EWM
    for s in SENSORS:
        df[f"{s}_ewm_03"] = df.groupby("unit")[s].transform(
            lambda x: x.ewm(alpha=0.3).mean()
        )

    # Drift
    for s in SENSORS:
        df[f"{s}_cum_drift"] = df.groupby("unit")[s].transform(
            lambda x: x - x.iloc[0]
        )

    # Cycle normalization
    df["cycle_norm"] = df.groupby("unit")["cycle"].transform(
        lambda x: x / max(x.max(), 1)
    )
    df["cycle_norm2"] = df["cycle_norm"] ** 2
    df["cycle_norm3"] = df["cycle_norm"] ** 3

    # Health Index
    smin = df[SENSORS].min()
    smax = df[SENSORS].max()
    hi = 1 - (df[SENSORS] - smin) / (smax - smin + 1e-6)

    df["HI"] = hi.mean(axis=1)
    df["HI_smooth"] = df.groupby("unit")["HI"].transform(
        lambda x: x.ewm(alpha=0.2).mean()
    )
    df["HI_slope"] = df.groupby("unit")["HI"].diff().fillna(0)

    df["health_index"] = df["HI"]
    df["health_smooth"] = df["HI_smooth"]

    if "op_mode_id" not in df.columns:
        df["op_mode_id"] = 0

    return df

# ======================================================
# Predict RUL
# ======================================================
def predict_rul(df_feat: pd.DataFrame, model_type, model, feats):
    df = df_feat.copy()

    for f in feats:
        if f not in df.columns:
            df[f] = 0.0

    X = df[feats].fillna(0)

    if model_type == "xgb":
        return model.predict(xgb.DMatrix(X))

    return model.predict(X)

# ======================================================
# Confidence Interval
# ======================================================
def add_confidence_interval(df, pred_col="RUL_pred", sigma=12.0, z=1.96):
    df = df.copy()
    df["RUL_lower"] = df[pred_col] - z * sigma
    df["RUL_upper"] = df[pred_col] + z * sigma
    df["RUL_lower"] = df["RUL_lower"].clip(lower=0)
    return df

# ======================================================
# Calendar-based failure date
# ======================================================
def add_failure_date(df_latest, reference_date=None, cycle_to_days=1):
    df = df_latest.copy()

    if reference_date is None:
        reference_date = pd.Timestamp.today().normalize()

    df["Days_to_Failure"] = df["RUL_pred"].clip(lower=0)
    df["Failure_Date"] = df["Days_to_Failure"].apply(
        lambda d: reference_date + timedelta(days=int(d * cycle_to_days))
    )

    return df
