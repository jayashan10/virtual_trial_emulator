from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

ID_COL = "SUBJID"
STUDY_COL = "STUDYID"

# Default subsets similar to 149 to manage feature volume
DEFAULT_PARAMCD = ["PSA", "ALP", "HGB", "LDH", "CREATININE"]


def _window_mask(series_days: pd.Series, window: Tuple[int, int]) -> pd.Series:
    lo, hi = window
    return (series_days >= lo) & (series_days <= hi)


def _agg_numeric(ts: pd.DataFrame, value_col: str, time_col: str, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    vals = ts[value_col].dropna()
    if vals.empty:
        return {
            f"{prefix}_last": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_slope": np.nan,
            f"{prefix}_count": 0,
        }
    out[f"{prefix}_last"] = float(vals.iloc[-1])
    out[f"{prefix}_mean"] = float(vals.mean())
    out[f"{prefix}_count"] = int(vals.shape[0])
    # slope via simple linear regression time vs value (days)
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
        windows = {"baseline": (-30, 0), "early": (1, 60)}
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
    df = df[df["LBTEST"].astype("string").str.upper().isin(paramcd_keep)]

    feats_list = []
    for (sid, usubjid), g in df.groupby([STUDY_COL, ID_COL]):
        feats: Dict[str, float] = {STUDY_COL: sid, ID_COL: usubjid}
        for test in sorted(g["LBTEST"].astype("string").str.upper().unique()):
            sub = g[g["LBTEST"].astype("string").str.upper() == test].sort_values("DAY")
            for wn, window in windows.items():
                mask = _window_mask(sub["DAY"], window)
                agg = _agg_numeric(sub.loc[mask], "VALUE", "DAY", prefix=f"lab_{test}_{wn}")
                feats.update(agg)
        feats_list.append(feats)

    out = pd.DataFrame(feats_list) if feats_list else pd.DataFrame(columns=[ID_COL, STUDY_COL])
    return out


