import argparse, json, tempfile
from pathlib import Path
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from data_utils import load_tabular_csv

# Optional: build features on the fly if only raw root is provided
try:
    from rf_features_from_tree import make_features_csv
except Exception:
    make_features_csv = None

def train_and_eval(train_csv: str, label_col: str, out_dir: str, cv: int, random_state: int):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Load features/labels
    X, y = load_tabular_csv(train_csv, label_col=label_col)
    feature_cols = list(X.columns)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y.values)

    # Model
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state
    )

    # Cross-validated predictions for honest metrics
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    y_pred_cv = cross_val_predict(clf, X, y_enc, cv=skf, n_jobs=-1)

    acc = accuracy_score(y_enc, y_pred_cv)
    f1_macro = f1_score(y_enc, y_pred_cv, average="macro")
    cm = confusion_matrix(y_enc, y_pred_cv)
    report = classification_report(y_enc, y_pred_cv, target_names=le.classes_, output_dict=True)

    # Fit on full data and save artifacts
    clf.fit(X, y_enc)
    dump(clf, out / "model.joblib")
    dump(le, out / "label_encoder.joblib")
    (out / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "per_class_f1": {k: v["f1-score"] for k, v in report.items() if k in le.classes_},
        "classes": le.classes_.tolist(),
        "cv_folds": cv
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Console summary
    print("=== CV RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print("Per-class F1:")
    for cls in metrics["classes"]:
        print(f"  {cls}: {metrics['per_class_f1'][cls]:.4f}")
    print("Confusion matrix:\n", np.array(metrics["confusion_matrix"]))
    print(f"Artifacts saved to: {out.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", help="CSV of features with a label column")
    ap.add_argument("--raw_root", help="(Optional) root of raw trials; will be converted to features")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    if not args.train_csv and not args.raw_root:
        ap.error("Provide either --train_csv or --raw_root")

    # Build temp features if raw root is provided
    if args.raw_root:
        if make_features_csv is None:
            raise SystemExit("rf_features_from_tree.py not available.")
        tmp = Path(tempfile.gettempdir()) / "bci_combined_features.csv"
        make_features_csv(args.raw_root, str(tmp))
        args.train_csv = str(tmp)

    train_and_eval(args.train_csv, args.label_col, args.out_dir, args.cv, args.random_state)

if __name__ == "__main__":
    main()
















