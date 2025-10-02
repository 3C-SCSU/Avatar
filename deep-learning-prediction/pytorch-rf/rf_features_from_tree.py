import argparse, os, sys
import pandas as pd
import numpy as np
from feature_extraction import stat_features

# Normalize folder names to canonical class labels
COMMANDS = ["forward", "backward", "left", "right", "landing", "takeoff"]

def norm_label(folder: str) -> str:
    f = folder.lower()
    for c in COMMANDS:
        if c in f:
            return c
    return f  # fallback to folder name

def make_features_csv(root: str, out_csv: str) -> str:
    rows = []
    count = 0
    for dirpath, _, files in os.walk(root):
        label = norm_label(os.path.basename(dirpath))
        for name in files:
            if not name.lower().endswith(".csv"):
                continue
            fp = os.path.join(dirpath, name)
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"[skip] {fp}: {e}", file=sys.stderr); continue

            num = df.select_dtypes(include=[np.number])
            if num.empty:
                print(f"[skip] {fp}: no numeric columns", file=sys.stderr); continue

            feats = stat_features(num)
            feats.insert(0, "label", label)
            feats["src_file"] = fp
            rows.append(feats)
            count += 1
            if count % 50 == 0:
                print(f"[info] processed {count} files...", file=sys.stderr)

    if not rows:
        raise SystemExit("No usable CSV files found under root.")
    out = pd.concat(rows, ignore_index=True)
    cols = ["label"] + [c for c in out.columns if c not in ("label", "src_file")] + ["src_file"]
    out = out[cols]
    out.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv} with shape {out.shape}")
    print(out["label"].value_counts().to_string())
    return out_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root dir containing class folders with raw CSV trials")
    ap.add_argument("--out", required=True, help="Output combined features CSV")
    args = ap.parse_args()
    make_features_csv(args.root, args.out)

if __name__ == "__main__":
    main()

