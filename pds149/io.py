import os
from typing import Dict, Tuple
import pandas as pd

TABLE_FILES = {
    "core": "CoreTable_training.csv",
    "labs": "LabValue_training.csv",
    "vitals": "VitalSign_training.csv",
    "lesions": "LesionMeasure_training.csv",
    "medhist": "MedHistory_training.csv",
    "priormed": "PriorMed_training.csv",
}

ID_COL = "RPT"
STUDY_COL = "STUDYID"


def read_table(path: str, use_pyarrow: bool = True) -> pd.DataFrame:
    na_vals = [".", ""]
    kwargs = dict(low_memory=False, na_values=na_vals)
    if use_pyarrow:
        kwargs["dtype_backend"] = "pyarrow"
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)


def load_training_tables(data_root: str) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for key, fname in TABLE_FILES.items():
        fpath = os.path.join(data_root, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing table: {fpath}")
        tables[key] = read_table(fpath)
    return tables


def ensure_ids_are_string(df: pd.DataFrame, id_cols: Tuple[str, ...] = (ID_COL, STUDY_COL)) -> pd.DataFrame:
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def audit_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for name, df in tables.items():
        summary[name] = {
            "rows": len(df),
            "cols": df.shape[1],
            "n_id": int(df[ID_COL].notna().sum()) if ID_COL in df.columns else 0,
            "n_study": int(df[STUDY_COL].notna().sum()) if STUDY_COL in df.columns else 0,
        }
    return summary
