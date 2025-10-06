from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

ID_COL = "SUBJID"
STUDY_COL = "STUDYID"


def observed_os_df(adsl: pd.DataFrame, os_df: pd.DataFrame) -> pd.DataFrame:
    arm_col = None
    for c in ["TRT", "ARM", "ARMCD"]:
        if c in adsl.columns:
            arm_col = c
            break
    if arm_col is None:
        arm_map = adsl[[ID_COL, STUDY_COL]].copy()
        arm_map["ARM"] = "ARM"
    else:
        arm_map = adsl[[ID_COL, STUDY_COL, arm_col]].copy()
        arm_map = arm_map.rename(columns={arm_col: "ARM"})
    df = os_df.merge(arm_map, on=[ID_COL, STUDY_COL], how="left")
    return df


def km_curves_by_study_arm(df: pd.DataFrame, times: np.ndarray) -> Dict[Tuple[str, str], pd.DataFrame]:
    curves: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (study, arm), g in df.groupby([STUDY_COL, "ARM"]):
        if g.empty:
            continue
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
        data = g[["time", "event", "ARM"]].copy()
        data["event"] = data["event"].astype(int)
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


