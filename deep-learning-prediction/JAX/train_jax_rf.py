"""
train_jax_rf.py 
deep-learning-prediction/JAX/train_jax_rf.py
JAX Random Forest training (CPU-only, no JIT) + artifact export
Produces: metrics.json, confusion_matrix.png, rf_jax_model.pkl
Author: Esma Ali
"""

import os
# ---- force CPU + disable JIT + headless plotting (Windows-safe) ----
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_DISABLE_JIT"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import json
import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

from jax_rf import RandomForest  # your JAX-implemented RF (TreeNode/DecisionTree/RandomForest)

# ---- paths / hyperparams ----
CSV_PATH = Path("data/brainwaves.csv")
OUT_DIR  = Path("deep-learning-prediction/JAX/artifacts")
N_ESTIMATORS = 120
MAX_DEPTH    = 10
SEED         = 42
TEST_SIZE    = 0.20   # 80/20 split with stratification

def save_confusion_matrix(cm: np.ndarray, out_png: Path) -> None:
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = range(cm.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]:d}",
                ha="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    # ---- load data ----
    df = pd.read_csv(CSV_PATH)
    if "label" not in df.columns:
        raise SystemExit("Expected a 'label' column in the combined CSV.")

    X_np = df.drop(columns=["label"]).astype("float32").to_numpy()
    y_np = df["label"].astype("int32").to_numpy()

    # to JAX arrays (satisfies "use JAX" requirement)
    X = jnp.asarray(X_np)
    y = jnp.asarray(y_np)

    # ---- stratified split (still return NumPy for sklearn metrics if needed) ----
    Xtr_np, Xte_np, ytr_np, yte_np = train_test_split(
        np.asarray(X), np.asarray(y), test_size=TEST_SIZE, stratify=np.asarray(y), random_state=SEED
    )

    # back to JAX for the RF implementation
    Xtr = jnp.asarray(Xtr_np); ytr = jnp.asarray(ytr_np)
    Xte = jnp.asarray(Xte_np); yte = jnp.asarray(yte_np)

    # ---- train JAX RF (CPU, no JIT) ----
    rf = RandomForest(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=None,
        num_classes=int(jnp.max(ytr)) + 1,
        seed=SEED,
    )
    rf.fit(Xtr, ytr)

    # ---- evaluate ----
    ypred_np = rf.predict(Xte)          # rf returns NumPy ints
    yte_np   = np.asarray(yte, dtype=np.int32)

    acc = float(accuracy_score(yte_np, ypred_np))
    f1m = float(f1_score(yte_np, ypred_np, average="macro"))
    cm  = confusion_matrix(yte_np, ypred_np)

    # ---- save artifacts ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # metrics.json
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": acc,
                "f1_macro": f1m,
                "confusion_matrix": cm.tolist(),
                "n_estimators": N_ESTIMATORS,
                "max_depth": MAX_DEPTH,
            },
            indent=2,
        )
    )

    # confusion_matrix.png
    save_confusion_matrix(cm, OUT_DIR / "confusion_matrix.png")

    # trained model (pickle the Python objects)
    joblib.dump(rf, OUT_DIR / "rf_jax_model.pkl")

    # console summary
    print(f"accuracy={acc:.4f}  f1_macro={f1m:.4f}")
    print(f"Artifacts → {OUT_DIR.resolve()}")
    print("\nClassification report:\n")
    print(classification_report(yte_np, ypred_np))

if __name__ == "__main__":
    main()