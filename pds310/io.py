import os
from typing import Dict, Tuple
import pandas as pd


# Actual 310 columns use SUBJID and lack STUDYID; we synthesize STUDYID as a constant
ID_COL = "SUBJID"
STUDY_COL = "STUDYID"


def _read_sas(path: str) -> pd.DataFrame:
    try:
        import pyreadstat  # type: ignore
    except Exception as e:
        raise ImportError("pyreadstat is required to read SAS7BDAT files. Install via uv: uv add pyreadstat") from e
    df, _ = pyreadstat.read_sas7bdat(path, dates_as_pandas_datetime=True)
    return df


def load_adam_tables(data_root: str) -> Dict[str, pd.DataFrame]:
    """Load required ADaM tables for 310.

    Expected files in data_root:
      - adsl_*.sas7bdat (subject-level)
      - adlb_*.sas7bdat (lab)
      - adae_*.sas7bdat (adverse events) [optional for Phase 2]

    Returns a dict with keys: adsl, adlb, adae (if present).
    """
    # Discover files by prefix
    files = {f.lower(): os.path.join(data_root, f) for f in os.listdir(data_root) if f.lower().endswith('.sas7bdat')}

    def _find(prefix: str) -> str:
        for name, path in files.items():
            if name.startswith(prefix):
                return path
        raise FileNotFoundError(f"Missing SAS file with prefix '{prefix}' in {data_root}")

    adsl_path = _find("adsl_") if any(k.startswith("adsl_") for k in files) else files.get("adsl.sas7bdat") or _find("adsl")
    adlb_path = _find("adlb_") if any(k.startswith("adlb_") for k in files) else files.get("adlb.sas7bdat") or _find("adlb")
    adae_path = None
    for cand in ("adae_", "adae.sas7bdat", "adae"):
        if any(k.startswith(cand) for k in files) or cand in files:
            adae_path = files.get("adae.sas7bdat") or files.get(cand) or next((p for n, p in files.items() if n.startswith("adae_")), None)
            break

    tables: Dict[str, pd.DataFrame] = {}
    tables["adsl"] = _read_sas(adsl_path)
    tables["adlb"] = _read_sas(adlb_path)
    if adae_path:
        try:
            tables["adae"] = _read_sas(adae_path)
        except Exception:
            pass

    # Normalize ID types
    for k, df in list(tables.items()):
        # Ensure SUBJID exists as string
        if ID_COL in df.columns:
            df[ID_COL] = df[ID_COL].astype("string")
        # Synthesize STUDYID if missing
        if STUDY_COL not in df.columns:
            df[STUDY_COL] = "PDS310"
        else:
            df[STUDY_COL] = df[STUDY_COL].astype("string")
        tables[k] = df

    return tables


def audit_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for name, df in tables.items():
        out[name] = {
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "n_id": int(df[ID_COL].notna().sum()) if ID_COL in df.columns else 0,
            "n_study": int(df[STUDY_COL].notna().sum()) if STUDY_COL in df.columns else 0,
        }
    return out


