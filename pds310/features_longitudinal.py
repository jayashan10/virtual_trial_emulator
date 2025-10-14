from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .labs import canonical_lab_list, canonical_lab_name

ID_COL = "SUBJID"
STUDY_COL = "STUDYID"

# Default subset to manage feature volume
DEFAULT_PARAMCD = list(canonical_lab_list())


def _window_mask(series_days: pd.Series, window: Tuple[int, int]) -> pd.Series:
    lo, hi = window
    return (series_days >= lo) & (series_days <= hi)


def _agg_numeric(
    ts: pd.DataFrame,
    value_col: str,
    time_col: str,
    prefix: str,
    include_slope: bool = True
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    vals = ts[value_col].dropna()

    if vals.empty:
        out[f"{prefix}_last"] = np.nan
        if include_slope:
            out[f"{prefix}_slope"] = np.nan
        return out

    out[f"{prefix}_last"] = float(vals.iloc[-1])

    if include_slope:
        times = ts.loc[vals.index, time_col].astype(float)
        if len(vals) >= 2 and times.nunique() > 1:
            slope, _ = np.polyfit(times.values, vals.values, 1)
            out[f"{prefix}_slope"] = float(slope)
        else:
            out[f"{prefix}_slope"] = np.nan

    return out


def build_longitudinal_features(
    adlb: pd.DataFrame,
    windows: Dict[str, Tuple[int, int]] = None,
    paramcd_keep: List[str] = None,
) -> pd.DataFrame:
    if windows is None:
        windows = {"early": (1, 42)}
    if paramcd_keep is None:
        paramcd_keep = DEFAULT_PARAMCD

    if adlb is None or adlb.empty:
        return pd.DataFrame(columns=[ID_COL, STUDY_COL])

    df = adlb.copy()
    # 310 columns: LBTEST, LBSTRESN, VISITDY
    keep = [c for c in [ID_COL, STUDY_COL, "LBTEST", "LBSTRESN", "VISITDY"] if c in df.columns]
    df = df[keep]
    df["VALUE"] = pd.to_numeric(df.get("LBSTRESN"), errors="coerce")
    df["DAY"] = pd.to_numeric(df.get("VISITDY"), errors="coerce")
    df = df.dropna(subset=["DAY"])  # must have a day value
    df["CANON"] = df["LBTEST"].map(canonical_lab_name)
    valid_codes = set(canonical_lab_list(paramcd_keep))
    df = df[df["CANON"].isin(valid_codes)]

    feats_list = []
    for (sid, usubjid), g in df.groupby([STUDY_COL, ID_COL]):
        feats: Dict[str, float] = {STUDY_COL: sid, ID_COL: usubjid}
        for test in sorted(g["CANON"].dropna().unique()):
            sub = g[g["CANON"] == test].sort_values("DAY")
            for wn, window in windows.items():
                if wn != "early":
                    continue
                mask = _window_mask(sub["DAY"], window)
                agg = _agg_numeric(
                    sub.loc[mask],
                    "VALUE",
                    "DAY",
                    prefix=f"lab_{test}_{wn}",
                    include_slope=True,
                )
                feats.update(agg)
        feats_list.append(feats)

    out = pd.DataFrame(feats_list) if feats_list else pd.DataFrame(columns=[ID_COL, STUDY_COL])
    return out

