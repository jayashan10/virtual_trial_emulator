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


def load_adam_tables(data_root: str, verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """Load all available ADaM tables for PDS310.

    Expected files in data_root:
      - adsl_*.sas7bdat (subject-level) - REQUIRED
      - adlb_*.sas7bdat (lab) - REQUIRED
      - adae_*.sas7bdat (adverse events) - optional
      - adls_*.sas7bdat (lesion measurements) - optional
      - adpm_*.sas7bdat (physical measurements) - optional
      - adrsp_*.sas7bdat (response) - optional
      - biomark_*.sas7bdat (biomarkers) - optional

    Returns a dict with keys: adsl, adlb, adae, adls, adpm, adrsp, biomark (where available).
    """
    # Discover files by prefix
    files = {f.lower(): os.path.join(data_root, f) for f in os.listdir(data_root) if f.lower().endswith('.sas7bdat')}

    def _find(prefix: str, required: bool = True) -> str:
        for name, path in files.items():
            if name.startswith(prefix):
                return path
        if required:
            raise FileNotFoundError(f"Missing SAS file with prefix '{prefix}' in {data_root}")
        return None

    # Load required tables
    tables: Dict[str, pd.DataFrame] = {}
    
    # ADSL - required
    adsl_path = _find("adsl_", required=True)
    tables["adsl"] = _read_sas(adsl_path)
    if verbose:
        print(f"Loaded ADSL: {len(tables['adsl'])} rows")
    
    # ADLB - required
    adlb_path = _find("adlb_", required=True)
    tables["adlb"] = _read_sas(adlb_path)
    if verbose:
        print(f"Loaded ADLB: {len(tables['adlb'])} rows")
    
    # Optional tables
    optional_tables = {
        "adae": "adae_",
        "adls": "adls_",
        "adpm": "adpm_",
        "adrsp": "adrsp_",
        "biomark": "biomark_",
    }
    
    for table_name, prefix in optional_tables.items():
        path = _find(prefix, required=False)
        if path:
            try:
                tables[table_name] = _read_sas(path)
                if verbose:
                    print(f"Loaded {table_name.upper()}: {len(tables[table_name])} rows")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load {table_name.upper()}: {e}")
                pass
        else:
            if verbose:
                print(f"Info: {table_name.upper()} not found (optional)")

    # Normalize ID types for all tables
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

    if verbose:
        print(f"\nTotal tables loaded: {len(tables)}")
        print(f"Available tables: {', '.join(tables.keys())}")

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


