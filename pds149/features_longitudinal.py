from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

ID_COL = "RPT"
STUDY_COL = "STUDYID"

# Default subsets to keep feature explosion manageable
DEFAULT_LBTESTS = ["PSA", "ALP", "HB", "LDH", "CREAT"]
DEFAULT_VSTESTS = ["ECOG", "WEIGHT", "SYSBP", "DIABP"]


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
    labs: pd.DataFrame,
    vitals: pd.DataFrame,
    windows: Dict[str, Tuple[int, int]] = None,
    lab_tests: List[str] = None,
    vital_tests: List[str] = None,
) -> pd.DataFrame:
    if windows is None:
        windows = {"baseline": (-30, 0), "early": (1, 60)}
    if lab_tests is None:
        lab_tests = DEFAULT_LBTESTS
    if vital_tests is None:
        vital_tests = DEFAULT_VSTESTS

    # Prepare labs
    lf = []
    if labs is not None and not labs.empty:
        ldf = labs.copy()
        # Prefer standardized numeric value LBSTRESN if available; else use LBSTRESC coerced
        value_num_col = "LBSTRESN" if "LBSTRESN" in ldf.columns else "LBSTRESC"
        if value_num_col == "LBSTRESC":
            ldf[value_num_col] = pd.to_numeric(ldf[value_num_col], errors="coerce")
        ldf = ldf[[ID_COL, STUDY_COL, "LBTESTCD", value_num_col, "LBDT_PC"]].rename(columns={value_num_col: "VALUE", "LBDT_PC": "DAY"})
        ldf = ldf[ldf["LBTESTCD"].isin(lab_tests)]
        for (sid, rpt), g in ldf.groupby([STUDY_COL, ID_COL]):
            feats: Dict[str, float] = {STUDY_COL: sid, ID_COL: rpt}
            for test in ldf["LBTESTCD"].unique():
                sub = g[g["LBTESTCD"] == test].sort_values("DAY")
                for wn, window in windows.items():
                    mask = _window_mask(sub["DAY"], window)
                    agg = _agg_numeric(sub.loc[mask], "VALUE", "DAY", prefix=f"lab_{test}_{wn}")
                    feats.update(agg)
            lf.append(feats)
    labs_feat = pd.DataFrame(lf) if lf else pd.DataFrame(columns=[ID_COL, STUDY_COL])

    # Prepare vitals
    vf = []
    if vitals is not None and not vitals.empty:
        vdf = vitals.copy()
        value_col = "VSSTRESN" if "VSSTRESN" in vdf.columns else "VSSTRESC"
        if value_col == "VSSTRESC":
            vdf[value_col] = pd.to_numeric(vdf[value_col], errors="coerce")
        vdf = vdf[[ID_COL, STUDY_COL, "VSTESTCD", value_col, "VSDT_PC"]].rename(columns={value_col: "VALUE", "VSDT_PC": "DAY"})
        vdf = vdf[vdf["VSTESTCD"].isin(vital_tests)]
        for (sid, rpt), g in vdf.groupby([STUDY_COL, ID_COL]):
            feats: Dict[str, float] = {STUDY_COL: sid, ID_COL: rpt}
            for test in vdf["VSTESTCD"].unique():
                sub = g[g["VSTESTCD"] == test].sort_values("DAY")
                for wn, window in windows.items():
                    mask = _window_mask(sub["DAY"], window)
                    agg = _agg_numeric(sub.loc[mask], "VALUE", "DAY", prefix=f"vs_{test}_{wn}")
                    feats.update(agg)
            vf.append(feats)
    vitals_feat = pd.DataFrame(vf) if vf else pd.DataFrame(columns=[ID_COL, STUDY_COL])

    # Merge lab and vital features
    out = None
    for df in [labs_feat, vitals_feat]:
        if df is None or df.empty:
            continue
        out = df if out is None else out.merge(df, on=[ID_COL, STUDY_COL], how="outer")
    if out is None:
        out = pd.DataFrame(columns=[ID_COL, STUDY_COL])
    return out
