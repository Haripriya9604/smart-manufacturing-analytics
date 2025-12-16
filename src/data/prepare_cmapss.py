# src/data/prepare_cmapss.py
import os
import glob
import pandas as pd

RAW_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\raw\CMaps"
OUT_DIR = r"C:\Users\vicky\smart-manufacturing-analytics\data\processed\cmapss"
os.makedirs(OUT_DIR, exist_ok=True)

# mapping of file patterns for FD001..FD004
FD_IDS = ["FD001", "FD002", "FD003", "FD004"]

COLS = [
    "unit",
    "cycle",
    "setting1",
    "setting2",
    "setting3",
] + [f"s{i}" for i in range(1, 22)]

def read_txt(path):
    # whitespace separated, no header
    return pd.read_csv(path, sep="\s+", header=None, names=COLS)

def prepare(fd_id):
    print(f"Processing {fd_id} ...")
    train_path = os.path.join(RAW_DIR, f"train_{fd_id}.txt")
    test_path = os.path.join(RAW_DIR, f"test_{fd_id}.txt")
    rul_path = os.path.join(RAW_DIR, f"RUL_{fd_id}.txt")

    train = read_txt(train_path)
    test = read_txt(test_path)
    rul = pd.read_csv(rul_path, header=None, names=["RUL"])

    # For train: compute RUL = max_cycle - cycle
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["RUL"] = max_cycle - train["cycle"]

    # For test: we have partial trajectories; append RUL from rul file to get true remaining lifetime
    # For each unit in test, the corresponding RUL is given (RUL file lines correspond to unit order)
    # We'll append failure_cycle = last_cycle + RUL_from_file
    # Then compute RUL for each row as failure_cycle - cycle
    # test units are numbered starting at 1 within each FD set
    test_units = test["unit"].unique()
    if len(rul) != len(test_units):
        print("Warning: RUL file length does not match test units count for", fd_id)
    # compute failure cycle per unit
    last_cycle_per_unit = test.groupby("unit")["cycle"].max().sort_index()
    # RUL lines correspond to unit index order 1..n
    failure_cycle = pd.Series(index=last_cycle_per_unit.index, dtype=int)
    for i, unit in enumerate(sorted(last_cycle_per_unit.index)):
        failure_cycle.loc[unit] = int(last_cycle_per_unit.loc[unit] + int(rul.iloc[i,0]))
    # map to test dataframe
    test["failure_cycle"] = test["unit"].map(failure_cycle)
    test["RUL"] = test["failure_cycle"] - test["cycle"]

    # add a dataset id column
    train["dataset"] = fd_id
    test["dataset"] = fd_id

    # Save individual processed files if needed
    train.to_csv(os.path.join(OUT_DIR, f"train_{fd_id}.csv"), index=False)
    test.to_csv(os.path.join(OUT_DIR, f"test_{fd_id}.csv"), index=False)
    print(f"Saved processed train/test for {fd_id}")
    return train, test

def main():
    all_train = []
    all_test = []
    for fd in FD_IDS:
        t, te = prepare(fd)
        all_train.append(t)
        all_test.append(te)

    df_train = pd.concat(all_train, ignore_index=True)
    df_test = pd.concat(all_test, ignore_index=True)
    df_train.to_csv(os.path.join(OUT_DIR, "train_all.csv"), index=False)
    df_test.to_csv(os.path.join(OUT_DIR, "test_all.csv"), index=False)
    print("Saved train_all.csv and test_all.csv in", OUT_DIR)

if __name__ == "__main__":
    main()
