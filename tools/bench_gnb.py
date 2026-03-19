# tools/bench_gnb.py
import argparse
import pathlib
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB


LABEL_MAP = [
    (["backward", "backwards"], "backward"),
    (["forward", "fowward"], "forward"),
    (["landing", "land"], "landing"),
    (["takeoff"], "takeoff"),
    (["turnleft"], "turnleft"),
    (["turnright"], "turnright"),
    (["steady"], "steady"),
    (["up"], "up"),
    (["down"], "down"),
    (["left"], "left"),
    (["right"], "right"),
]


def infer_label_from_path(path_str: str) -> Optional[str]:
    s2 = re.sub(r"[\s_\-]+", "", path_str.lower())
    for pats, val in LABEL_MAP:
        if any(p in s2 for p in pats):
            return val
    return None


def sanity_check() -> int:
    """Sanity test: if this fails, your environment/pipeline is broken."""
    print("\n--- Sanity Check: synthetic Gaussian data ---")
    rng = np.random.default_rng(42)

    n_classes = 4
    n_features = 32
    n_per_class = 2000

    X_list = []
    y_list = []
    for c in range(n_classes):
        mean = rng.normal(loc=0.0, scale=3.0, size=(n_features,))
        cov = np.eye(n_features) * rng.uniform(0.5, 1.5)
        Xc = rng.multivariate_normal(mean, cov, size=n_per_class).astype(np.float32)
        yc = np.full((n_per_class,), c, dtype=np.int32)
        X_list.append(Xc)
        y_list.append(yc)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = GaussianNB(var_smoothing=1e-9)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Sanity accuracy: {acc:.4f} (expected: usually > 0.90)")
    if acc < 0.85:
        print("❌ Sanity check failed: accuracy too low. Something is wrong with environment/math.")
        return 2

    print("✅ Sanity check passed.")
    return 0


def load_dataset_csv(data_root: pathlib.Path, skip_dirs: set[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Looks for *.csv files under data_root, infers labels from file/folder names.
    Expected future structure (example):
      brainwaves/<year>/<label>/*.csv
    """
    labeled_files: list[tuple[pathlib.Path, str]] = []
    for item in data_root.rglob("*.csv"):
        if any(part in skip_dirs for part in item.parts):
            continue
        label = infer_label_from_path(str(item))
        if label is None:
            continue
        labeled_files.append((item, label))

    if not labeled_files:
        raise FileNotFoundError("No labeled *.csv files found.")

    dfs: list[pd.DataFrame] = []
    for path, label in labeled_files:
        try:
            df = pd.read_csv(path, delimiter=",", header=None, on_bad_lines="skip")
            df["__label__"] = label
            dfs.append(df)
        except Exception as e:
            print(f"Skipped {path}: {e}")

    if not dfs:
        raise RuntimeError("Found labeled files but could not parse any into a dataset.")

    all_df = pd.concat(dfs, ignore_index=True)

    y_cat = all_df["__label__"].astype("category")
    label_names = list(y_cat.cat.categories)
    y = y_cat.cat.codes.to_numpy(dtype=np.int32)

    num_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise RuntimeError("No numeric columns found in parsed CSVs.")

    Xdf = all_df[num_cols].replace([np.inf, -np.inf], np.nan)
    Xdf = Xdf.dropna(axis=1, how="all")
    med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(med)

    std = Xdf.std(numeric_only=True)
    keep_cols = std[std > 0].index.tolist()
    Xdf = Xdf[keep_cols]

    X = Xdf.to_numpy(dtype=np.float32)
    return X, y, label_names


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark GaussianNB performance (sklearn baseline).")
    ap.add_argument("--data-root", type=str, required=True, help="Dataset root folder")
    ap.add_argument("--skip-dir", action="append", default=["Group1-8channels"], help="Dir name(s) to skip")
    ap.add_argument("--var-smoothing", type=float, default=1e-9)
    args = ap.parse_args()

    data_root = pathlib.Path(args.data_root)
    skip_dirs = set(args.skip_dir)

    # If dataset isn't present yet, still do something useful:
    try:
        X, y, label_names = load_dataset_csv(data_root, skip_dirs)
    except FileNotFoundError:
        print(
            f"\nDataset not found under: {data_root}\n"
            "This is expected right now because file-opendata/brainwaves/access_data.txt says S3 is 'Coming Soon'.\n"
            "Running a sanity check instead.\n"
        )
        return sanity_check()

    print("Loaded X:", X.shape, "classes:", len(label_names))
    print("Labels:", label_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = GaussianNB(var_smoothing=float(args.var_smoothing))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())