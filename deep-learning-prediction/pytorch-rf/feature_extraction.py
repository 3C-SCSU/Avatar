import numpy as np
import pandas as pd

def stat_features(num: pd.DataFrame) -> pd.DataFrame:
    """
    Compute robust per-column statistics for a numeric DataFrame (one trial):
    mean, std, var, min, max, median, q25, q75, IQR, energy.
    Returns a single-row DataFrame.
    """
    feats = {
        "mean":   num.mean(),
        "std":    num.std(ddof=1),
        "var":    num.var(ddof=1),
        "min":    num.min(),
        "max":    num.max(),
        "median": num.median(),
        "q25":    num.quantile(0.25),
        "q75":    num.quantile(0.75),
        "energy": (num.pow(2).sum() / max(len(num), 1)).replace([np.inf, -np.inf], np.nan),
    }
    iqr = feats["q75"] - feats["q25"]

    row = {}
    for stat_name, series in feats.items():
        for col, val in series.items():
            row[f"{stat_name}_{col}"] = float(val) if pd.notna(val) else np.nan
    for col, val in iqr.items():
        row[f"iqr_{col}"] = float(val) if pd.notna(val) else np.nan
    return pd.DataFrame([row])
