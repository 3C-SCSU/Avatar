from typing import Tuple, Optional, List
import pandas as pd
import numpy as np

def load_tabular_csv(path: str, label_col: str = "label",
                     drop_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a CSV and split into (X, y):
    - All columns except label_col (and explicit drop_cols) are features.
    - Enforces numeric features.
    """
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found. Columns: {list(df.columns)}")

    if drop_cols is None:
        drop_cols = []
    drop_cols = [c for c in drop_cols if c in df.columns and c != label_col]

    y = df[label_col].copy()
    X = df.drop(columns=[label_col] + drop_cols)

    bad = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if bad:
        raise TypeError(f"All features must be numeric. Non-numeric columns: {bad}")
    return X, y
