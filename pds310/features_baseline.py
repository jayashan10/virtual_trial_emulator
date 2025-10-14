from typing import Dict, List
import pandas as pd

from .labs import canonical_lab_list, canonical_lab_name

ID_COL = "SUBJID"
STUDY_COL = "STUDYID"

def build_baseline_from_adsl(adsl: pd.DataFrame) -> pd.DataFrame:
    cols_keep: List[str] = [c for c in [ID_COL, STUDY_COL, "AGE", "SEX", "RACE", "B_ECOG"] if c in adsl.columns]
    base = adsl[cols_keep].copy()
    # Normalize categoricals as strings
    for c in ["RACE", "SEX"]:
        if c in base.columns:
            base[c] = base[c].astype("string").fillna("MISSING")
    # Numeric conversions
    for c in ["AGE"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    return base


def build_baseline_labs_from_adlb(adlb: pd.DataFrame) -> pd.DataFrame:
    if adlb is None or adlb.empty:
        return pd.DataFrame(columns=[ID_COL, STUDY_COL])
    df = adlb.copy()
    # 310: earliest VISITDY as baseline per LBTEST
    df["VISITDY"] = pd.to_numeric(df.get("VISITDY"), errors="coerce")
    df_baseline = df.sort_values([ID_COL, STUDY_COL, "LBTEST", "VISITDY"]).groupby([ID_COL, STUDY_COL, "LBTEST"], as_index=False).first()

    # Use LBSTRESN as numeric value
    val_col = "LBSTRESN" if "LBSTRESN" in df_baseline.columns else None
    if val_col is None:
        return pd.DataFrame(columns=[ID_COL, STUDY_COL])
    df_baseline[val_col] = pd.to_numeric(df_baseline[val_col], errors="coerce")

    # Pivot LBTEST to columns using canonical names
    df_baseline["CANON"] = df_baseline["LBTEST"].map(canonical_lab_name)
    df_baseline = df_baseline.dropna(subset=["CANON"])
    keep_codes = set(canonical_lab_list())
    df_baseline = df_baseline[df_baseline["CANON"].isin(keep_codes)]
    if df_baseline.empty:
        return pd.DataFrame(columns=[ID_COL, STUDY_COL])

    wide = df_baseline.pivot_table(index=[ID_COL, STUDY_COL], columns="CANON", values=val_col, aggfunc="first")
    wide = wide.reset_index()
    rename_map = {code: f"baseline_{code}" for code in wide.columns if code not in [ID_COL, STUDY_COL]}
    wide = wide.rename(columns=rename_map)
    # Merge to one row per subject
    return wide


def assemble_baseline_feature_view(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    adsl = tables.get("adsl")
    adlb = tables.get("adlb")
    base = build_baseline_from_adsl(adsl) if adsl is not None else pd.DataFrame(columns=[ID_COL, STUDY_COL])
    labs = build_baseline_labs_from_adlb(adlb) if adlb is not None else pd.DataFrame(columns=[ID_COL, STUDY_COL])
    out = base
    if labs is not None and not labs.empty:
        out = out.merge(labs, on=[ID_COL, STUDY_COL], how="left")
    return out

