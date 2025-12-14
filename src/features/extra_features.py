# src/features/extra_features.py
import os
import pandas as pd
import numpy as np
from scipy.stats import skew

PROCESSED_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\processed\cmapss"
FILES = ["train_all_features.csv", "test_all_features.csv"]
SENSOR_COLS = [f"s{i}" for i in range(1,22)]

def add_extra(df):
    # cycle polys
    if "cycle_norm" in df.columns:
        df["cycle_norm2"] = df["cycle_norm"]**2
        df["cycle_norm3"] = df["cycle_norm"]**3

    # long-window medians / skew / slope (by dataset+unit)
    df = df.sort_values(["dataset","unit","cycle"]).reset_index(drop=True)
    frames = []
    for (ds, unit), g in df.groupby(["dataset","unit"], sort=False):
        gg = g.copy()
        for w in (20,40):
            for s in SENSOR_COLS:
                if s in gg.columns:
                    gg[f"{s}_med_{w}"] = gg[s].rolling(window=w, min_periods=1).median()
                    gg[f"{s}_skew_{w}"] = gg[s].rolling(window=w, min_periods=1).apply(lambda x: float(skew(x)) if len(x)>1 else 0.0)
                    # slope: linear fit over window -> slope (approx)
                    def slope(arr):
                        if len(arr) <= 1: return 0.0
                        x = np.arange(len(arr))
                        # simple least-squares slope:
                        A = np.vstack([x, np.ones_like(x)]).T
                        m, c = np.linalg.lstsq(A, arr, rcond=None)[0]
                        return float(m)
                    gg[f"{s}_slope_{w}"] = gg[s].rolling(window=w, min_periods=2).apply(slope, raw=True).fillna(0.0)

        # approximate FFT energy on 20-window (sum squared normalized FFT magnitudes)
        for s in SENSOR_COLS:
            if s in gg.columns:
                def fft_energy(arr):
                    a = np.asarray(arr)
                    if len(a) < 3: return 0.0
                    f = np.fft.rfft(a - a.mean())
                    e = np.sum((np.abs(f))**2)
                    return float(e)
                gg[f"{s}_fft20"] = gg[s].rolling(window=20, min_periods=3).apply(fft_energy, raw=True).fillna(0.0)

        frames.append(gg)
    return pd.concat(frames, ignore_index=True)

def main():
    for fname in FILES:
        p = os.path.join(PROCESSED_DIR, fname)
        print("Reading:", p)
        df = pd.read_csv(p)
        df2 = add_extra(df)
        df2.to_csv(p, index=False)
        print("Wrote:", p, "shape:", df2.shape)

if __name__ == "__main__":
    main()
