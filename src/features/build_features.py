# src\features\build_features.py
import os
import pandas as pd

PROCESSED_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\processed\cmapss"
OUT_TRAIN = os.path.join(PROCESSED_DIR, "train_all_features.csv")
OUT_TEST = os.path.join(PROCESSED_DIR, "test_all_features.csv")

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]

def add_time_features(df, windows=(5,10)):
    df = df.sort_values(["dataset","unit","cycle"]).reset_index(drop=True)
    frames = []
    for (ds, unit), g in df.groupby(["dataset","unit"], sort=False):
        gg = g.copy()
        for w in windows:
            for s in SENSOR_COLS:
                if s in gg.columns:
                    gg[f"{s}_rm_{w}"] = gg[s].rolling(window=w, min_periods=1).mean()
                    gg[f"{s}_rs_{w}"] = gg[s].rolling(window=w, min_periods=1).std().fillna(0)
                    gg[f"{s}_r_range_{w}"] = (gg[s].rolling(window=w, min_periods=1).max() - gg[s].rolling(window=w, min_periods=1).min()).fillna(0)
        for s in SENSOR_COLS:
            if s in gg.columns:
                gg[f"{s}_d1"] = gg[s].diff().fillna(0)
                gg[f"{s}_ewm_03"] = gg[s].ewm(alpha=0.3).mean()
        if "failure_cycle" in gg.columns:
            gg["cycle_norm"] = gg["cycle"] / gg["failure_cycle"]
        else:
            gg["cycle_norm"] = gg["cycle"] / gg["cycle"].max()
        frames.append(gg)
    return pd.concat(frames, ignore_index=True)

def build(input_csv, output_csv):
    print("Reading:", input_csv)
    df = pd.read_csv(input_csv)
    df_feat = add_time_features(df, windows=(5,10))
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_feat.to_csv(output_csv, index=False)
    print("Saved:", output_csv)

def main():
    train_in = os.path.join(PROCESSED_DIR, "train_all.csv")
    test_in = os.path.join(PROCESSED_DIR, "test_all.csv")
    if not os.path.exists(train_in) or not os.path.exists(test_in):
        raise FileNotFoundError("Processed train_all.csv / test_all.csv not found. Run prepare_cmapss.py first.")
    build(train_in, OUT_TRAIN)
    build(test_in, OUT_TEST)

if __name__ == "__main__":
    main()
