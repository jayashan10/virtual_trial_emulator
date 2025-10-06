from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

ID_COL = "RPT"
STUDY_COL = "STUDYID"


def build_arm_identifier(df: pd.DataFrame) -> pd.Series:
    parts = []
    for c in ["TRT1_ID", "TRT2_ID", "TRT3_ID"]:
        if c in df.columns:
            parts.append(df[c].astype("string").fillna(""))
    if not parts:
        return pd.Series(["ARM" for _ in range(len(df))], index=df.index)
    arm = parts[0]
    for p in parts[1:]:
        arm = arm.where(p.str.len() == 0, arm + "+" + p)
    arm = arm.str.strip("+")
    arm = arm.replace({"": "UNSPEC"})
    return arm


def observed_os_df(core: pd.DataFrame, os_df: pd.DataFrame) -> pd.DataFrame:
    arm_map = core[[ID_COL, STUDY_COL, "TRT1_ID", "TRT2_ID", "TRT3_ID"]].copy()
    arm_map["ARM"] = build_arm_identifier(arm_map)
    arm_map = arm_map[[ID_COL, STUDY_COL, "ARM"]]
    df = os_df.merge(arm_map, on=[ID_COL, STUDY_COL], how="left")
    return df


def km_curves_by_study_arm(df: pd.DataFrame, times: np.ndarray) -> Dict[Tuple[str, str], pd.DataFrame]:
    curves: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (study, arm), g in df.groupby([STUDY_COL, "ARM"]):
        km = KaplanMeierFitter()
        km.fit(g["time"].values, event_observed=g["event"].values)
        s = km.survival_function_at_times(times).reset_index()
        s.columns = ["time", "survival"]
        s[STUDY_COL] = study
        s["ARM"] = arm
        curves[(study, arm)] = s
    return curves


def hazard_ratios_by_study(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for study, g in df.groupby(STUDY_COL):
        if g["ARM"].nunique() < 2:
            continue
        # Use Cox with ARM as categorical
        data = g[["time", "event", "ARM"]].copy()
        data["event"] = data["event"].astype(int)
        # One-hot encode ARM with first as baseline
        data = pd.get_dummies(data, columns=["ARM"], drop_first=True)
        cox = CoxPHFitter()
        try:
            cox.fit(data, duration_col="time", event_col="event")
            for cov in cox.params_.index:
                hr = float(np.exp(cox.params_.loc[cov]))
                rows.append({"STUDYID": study, "contrast": cov.replace("ARM_", ""), "HR": hr})
        except Exception:
            continue
    return pd.DataFrame(rows)
