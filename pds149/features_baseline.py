from typing import Dict, Tuple
import pandas as pd

ID_COL = "RPT"
STUDY_COL = "STUDYID"

CORE_BASELINE_COLS = [
    ID_COL, STUDY_COL,
    # Demographics and disease baseline
    "AGEGRP", "AGEGRP2", "RACE_C", "ECOG_C", "GLEAS_DX", "TSTAG_DX",
    # Anthropometrics
    "BMI", "HEIGHTBL", "WEIGHTBL",
    # Baseline labs common in CoreTable
    "ALP", "ALT", "AST", "CA", "CREAT", "HB", "LDH", "NEU", "PLT", "PSA", "TBILI", "TESTO", "WBC",
    # Metastasis locations and indicators in CoreTable
    "BONE", "LYMPH_NODES", "LIVER", "LUNGS", "OTHER",
    # Treatment identifiers
    "TRT1_ID", "TRT2_ID", "TRT3_ID",
]


_DEF_MH_FLAGS = [
    ("DIAB", "MH_diabetes"), ("MI", "MH_mi"), ("CHF", "MH_chf"), ("DVT", "MH_dvt"),
]

_DEF_PM_FLAGS = [
    ("ANTI_ANDROGENS", "PM_antiandrogen"), ("GLUCOCORTICOID", "PM_glucocorticoid"),
    ("BISPHOSPHONATE", "PM_bisphosphonate"), ("ESTROGENS", "PM_estrogens"),
]


def _binary_flag_from_core(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0, index=df.index)
    s = df[col].astype("string").str.upper()
    return s.isin(["Y", "YES", "1", "TRUE"]).astype(int)


def build_baseline_core(core: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CORE_BASELINE_COLS if c in core.columns]
    base = core[cols].copy()
    # Normalize some categoricals
    for c in ["AGEGRP", "AGEGRP2", "RACE_C", "ECOG_C", "GLEAS_DX", "TSTAG_DX", "TRT1_ID", "TRT2_ID", "TRT3_ID"]:
        if c in base.columns:
            base[c] = base[c].astype("string").fillna("MISSING")
    # Convert labs to numeric
    for c in ["ALP", "ALT", "AST", "CA", "CREAT", "HB", "LDH", "NEU", "PLT", "PSA", "TBILI", "TESTO", "WBC", "BMI", "HEIGHTBL", "WEIGHTBL"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    # Binary metastasis flags
    for c in ["BONE", "LYMPH_NODES", "LIVER", "LUNGS", "OTHER"]:
        if c in base.columns:
            base[c] = _binary_flag_from_core(base, c)
    return base


def build_comorbidity_flags(mh: pd.DataFrame) -> pd.DataFrame:
    if mh.empty:
        return pd.DataFrame(columns=[ID_COL])
    mh_flags = mh[[ID_COL]].drop_duplicates().copy()
    for col, out in _DEF_MH_FLAGS:
        if col in mh.columns:
            flags = mh.groupby(ID_COL)[col].apply(lambda s: (s.astype("string").str.upper() == "YES").any()).astype(int)
            mh_flags[out] = mh_flags[ID_COL].map(flags).fillna(0).astype(int)
        else:
            mh_flags[out] = 0
    return mh_flags


def build_prior_med_flags(pm: pd.DataFrame) -> pd.DataFrame:
    if pm.empty:
        return pd.DataFrame(columns=[ID_COL])
    pm_flags = pm[[ID_COL]].drop_duplicates().copy()
    for col, out in _DEF_PM_FLAGS:
        if col in pm.columns:
            flags = pm.groupby(ID_COL)[col].apply(lambda s: (s.astype("string").str.upper() == "YES").any()).astype(int)
            pm_flags[out] = pm_flags[ID_COL].map(flags).fillna(0).astype(int)
        else:
            pm_flags[out] = 0
    return pm_flags


def assemble_baseline_feature_view(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    core = build_baseline_core(tables["core"]) if "core" in tables else pd.DataFrame()
    mh = build_comorbidity_flags(tables["medhist"]) if "medhist" in tables else pd.DataFrame()
    pm = build_prior_med_flags(tables["priormed"]) if "priormed" in tables else pd.DataFrame()

    # Merge on patient ID
    df = core
    for extra in (mh, pm):
        if not extra.empty:
            df = df.merge(extra, on=ID_COL, how="left")
    # Fill NA binary flags with 0
    for c in df.columns:
        if c.startswith("MH_") or c.startswith("PM_"):
            df[c] = df[c].fillna(0).astype(int)
    return df
