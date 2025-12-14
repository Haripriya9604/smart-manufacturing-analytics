import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

PROCESSED_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\processed\cmapss"

FILES = [
    "train_all_features.csv",
    "test_all_features.csv"
]

KEY_SENSORS = ["s2","s3","s4","s7","s8","s11","s12","s15"]

def compute_HI(df):
    """
    Computes PCA-based Health Index (HI) per dataset.
    """
    df = df.copy()

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]

        # PCA expects no NaNs
        pca = PCA(n_components=1)
        hi = pca.fit_transform(sub[KEY_SENSORS])

        df.loc[sub.index, "HI"] = hi[:, 0]

    return df


def add_smooth_HI(df):
    """
    Smoothed health index (rolling mean).
    """
    df["HI_smooth"] = (
        df.groupby(["dataset", "unit"])["HI"]
          .transform(lambda x: x.rolling(30, min_periods=5).mean())
    )
    return df


def add_HI_slope(df):
    """
    Long-window slope of smoothed HI (captures degradation trend).
    """
    def slope(series):
        if series.isnull().sum() > 0 or len(series) < 10:
            return np.nan
        y = series.values
        x = np.arange(len(y))
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m

    df["HI_slope"] = (
        df.groupby(["dataset", "unit"])["HI_smooth"]
          .transform(lambda x: x.rolling(40, min_periods=10).apply(slope, raw=False))
    )
    return df


def add_operating_mode(df):
    """
    Adds op_mode and op_mode_id to handle multi-condition FD004 behavior.
    """
    df["op_mode"] = (
        df["setting1"].round(2).astype(str) + "_" +
        df["setting2"].round(2).astype(str) + "_" +
        df["setting3"].round(2).astype(str)
    )

    df["op_mode_id"] = df["op_mode"].astype("category").cat.codes
    return df


def main():
    for fname in FILES:
        path = os.path.join(PROCESSED_DIR, fname)
        print("Loading:", path)

        df = pd.read_csv(path)

        print("→ Computing Health Index (HI)")
        df = compute_HI(df)

        print("→ Adding Smoothed Health Index")
        df = add_smooth_HI(df)

        print("→ Adding HI Slope")
        df = add_HI_slope(df)

        print("→ Encoding Operating Modes")
        df = add_operating_mode(df)

        print("→ Saving updated file...")
        df.to_csv(path, index=False)
        print("Saved:", path, "\nNew Shape:", df.shape)
        print("------------------------------------------")


if __name__ == "__main__":
    main()
