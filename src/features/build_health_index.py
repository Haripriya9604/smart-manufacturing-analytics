import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

PROCESSED_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\processed\cmapss"

FILES = ["train_all_features.csv", "test_all_features.csv"]
KEY_SENSORS = ["s2","s3","s4","s7","s8","s11","s12","s15"]

def build_HI(df):
    df = df.copy()

    for ds in df["dataset"].unique():
        sub = df[df.dataset == ds]

        # fit PCA on the training portion only
        pca = PCA(n_components=1)
        hi = pca.fit_transform(sub[KEY_SENSORS])

        df.loc[sub.index, "HI"] = hi[:,0]

    # smooth HI (critical for FD004)
    df["HI_smooth"] = df.groupby(["dataset","unit"])["HI"].transform(
        lambda x: x.rolling(30, min_periods=5).mean()
    )

    return df

def main():
    for fname in FILES:
        path = os.path.join(PROCESSED_DIR, fname)
        df = pd.read_csv(path)
        df2 = build_HI(df)
        df2.to_csv(path, index=False)
        print("Added HI fields:", path)

if __name__ == "__main__":
    main()
