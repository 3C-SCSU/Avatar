import argparse, json
from pathlib import Path
import pandas as pd
from joblib import load

def infer(csv: str, artifacts_dir: str, out_csv: str):
    art = Path(artifacts_dir)
    model = load(art / "model.joblib")
    le = load(art / "label_encoder.joblib")
    feat_cols = json.loads((art / "feature_columns.json").read_text())

    df = pd.read_csv(csv)
    X = df[feat_cols].copy()

    preds_enc = model.predict(X)
    preds = le.inverse_transform(preds_enc)

    out = df.copy()
    out["prediction"] = preds
    out.to_csv(out_csv, index=False)
    print(f"Saved predictions to: {Path(out_csv).resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Features CSV (same columns as training)")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--out_csv", default="predictions.csv")
    args = ap.parse_args()
    infer(**vars(args))

if __name__ == "__main__":
    main()
