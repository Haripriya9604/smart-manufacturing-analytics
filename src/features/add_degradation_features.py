import os
import pandas as pd
import numpy as np

PROCESSED_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\processed\cmapss"

FILES = [
    "train_all_features.csv",
    "test_all_features.csv"
]

SENSORS = [f"s{i}" for i in range(1,22)]

def add_deg_features(df):
    df = df.sort_values(["dataset","unit","cycle"])
    
    frames = []
    for (ds, unit), g in df.groupby(["dataset","unit"]):
        g = g.copy()
        
        # running mean slope (window=50)
        for s in SENSORS:
            if s in g.columns:
                g[f"{s}_trend50"] = g[s].rolling(50, min_periods=5).mean().diff()
        
        # cumulative drift normalised by cycle
        for s in SENSORS:
            if s in g.columns:
                g[f"{s}_cum_drift"] = (g[s] - g[s].iloc[0]) / (g["cycle"]+1)
        
        # health index = mean of selected sensors
        key_sensors = ["s2","s3","s4","s7","s8","s11","s12","s15"]
        existing = [s for s in key_sensors if s in g.columns]
        g["health_index"] = g[existing].mean(axis=1)
        
        # smoothed health index
        g["health_smooth"] = g["health_index"].rolling(30, min_periods=3).mean()

        frames.append(g)

    return pd.concat(frames, ignore_index=True)

def main():
    for fname in FILES:
        path = os.path.join(PROCESSED_DIR, fname)
        print("Loading:", path)
        df = pd.read_csv(path)
        df2 = add_deg_features(df)
        df2.to_csv(path, index=False)
        print("Saved:", path, "shape:", df2.shape)

if __name__ == "__main__":
    main()
