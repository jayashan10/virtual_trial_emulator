from typing import Dict
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from sklearn.compose import ColumnTransformer

from .os_simulate import _design_matrix, build_arm_identifier

ID_COL = "RPT"
STUDY_COL = "STUDYID"


def _inverse_sample_from_baseline(cox: CoxPHFitter, log_partial_hazard: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sampling for Cox PH using baseline cumulative hazard.

    For each subject i, draw u ~ Uniform(0,1) and solve H0(t) = -log(u) / exp(lp_i),
    where lp_i is the log-partial hazard and H0 is the baseline cumulative hazard.
    """
    # Extract baseline cumulative hazard curve
    bh = cox.baseline_cumulative_hazard_
    # Ensure we have a single column and get arrays
    times = bh.index.values.astype(float)
    cumhaz = bh.iloc[:, 0].values.astype(float)

    # Guard against degenerate baseline curves
    if len(times) == 0 or len(cumhaz) == 0:
        return np.zeros_like(log_partial_hazard)

    # Compute targets z = -log(u)/exp(lp)
    u = rng.uniform(1e-12, 1 - 1e-12, size=len(log_partial_hazard))
    z = -np.log(u) / np.exp(log_partial_hazard)

    # Vectorized inversion: find first index where cumhaz >= z
    idx = np.searchsorted(cumhaz, z, side="left")

    # Initialize sampled times
    t = np.empty_like(z, dtype=float)

    # Exact hits inside observed range
    inside = idx < len(times)
    t[inside] = times[idx[inside]]

    # For those beyond the last observed cumhaz, linearly extrapolate using last slope if possible
    outside = ~inside
    if np.any(outside):
        # Compute tail slope dH/dt from the last interval (avoid zero-length)
        if len(times) >= 2:
            dt = times[-1] - times[-2]
            dH = cumhaz[-1] - cumhaz[-2]
            slope = dH / dt if dt > 0 else 0.0
        else:
            slope = 0.0
        extra = z[outside] - cumhaz[-1]
        # If slope is near zero, cap at last time; else extend linearly
        add_t = np.where(slope <= 1e-12, 0.0, extra / max(slope, 1e-12))
        t[outside] = times[-1] + np.maximum(add_t, 0.0)

    # Ensure non-negative
    t = np.maximum(t, 0.0)
    return t


def simulate_os_times_cox(cox_bundle: Dict,
                           df_features: pd.DataFrame,
                           os_labels: pd.DataFrame,
                           core_for_arm: pd.DataFrame,
                           n_sim: int = 50,
                           seed: int = 42) -> pd.DataFrame:
    """Simulate OS times using a fitted CoxPHFitter via inverse sampling.

    Applies administrative censoring from observed OS labels. Returns a long dataframe
    with columns: RPT, STUDYID, ARM, sim, time, event.
    """
    cox: CoxPHFitter = cox_bundle["model"]
    pre: ColumnTransformer = cox_bundle["preprocessor"] if "preprocessor" in cox_bundle else cox_bundle["pre"]
    feat_names = cox_bundle["feature_names"] if "feature_names" in cox_bundle else cox_bundle["feat_names"]

    rng = np.random.default_rng(seed)

    # Build design matrix consistent with training preprocessor
    base_feats = df_features.drop(columns=[c for c in ["time", "event"] if c in df_features.columns])
    X = _design_matrix(pre, feat_names, base_feats)

    # Observed censoring times and arm identifiers
    arm_map = core_for_arm[[ID_COL, STUDY_COL, "TRT1_ID", "TRT2_ID", "TRT3_ID"]].copy()
    arm_map["ARM"] = build_arm_identifier(arm_map)
    arm_map = arm_map[[ID_COL, STUDY_COL, "ARM"]]

    lab = os_labels[[ID_COL, STUDY_COL, "time", "event"]].rename(columns={"time": "time_obs", "event": "event_obs"})
    meta = lab.merge(arm_map, on=[ID_COL, STUDY_COL], how="left")

    # Pre-compute linear predictors for all patients
    lp = cox.predict_log_partial_hazard(X).values.reshape(-1)

    sims = []
    for s in range(n_sim):
        t_raw = _inverse_sample_from_baseline(cox, lp, rng)
        # Apply censoring from observed labels
        time = np.minimum(t_raw, meta["time_obs"].values.astype(float))
        event = (t_raw <= meta["time_obs"].values.astype(float)).astype(int)
        sims.append(pd.DataFrame({
            ID_COL: df_features[ID_COL].values,
            STUDY_COL: df_features[STUDY_COL].values,
            "ARM": meta["ARM"].values,
            "sim": s,
            "time": time,
            "event": event,
        }))

    out = pd.concat(sims, ignore_index=True)
    return out


