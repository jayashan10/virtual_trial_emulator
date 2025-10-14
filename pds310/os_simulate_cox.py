from typing import Dict
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from sklearn.compose import ColumnTransformer


ID_COL = "SUBJID"
STUDY_COL = "STUDYID"


def _design_matrix(pre: ColumnTransformer, feat_names, df_feats: pd.DataFrame) -> pd.DataFrame:
    X_np = pre.transform(df_feats)
    X_df_full = pd.DataFrame(X_np)
    try:
        full_names = [str(c) for c in pre.get_feature_names_out()]
    except Exception:
        full_names = [f"f{i}" for i in range(X_df_full.shape[1])]
    X_df_full.columns = full_names
    X_df = X_df_full[[str(c) for c in feat_names]]
    return X_df


def _inverse_sample_from_baseline(cox: CoxPHFitter, log_partial_hazard: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bh = cox.baseline_cumulative_hazard_
    times = bh.index.values.astype(float)
    cumhaz = bh.iloc[:, 0].values.astype(float)
    if len(times) == 0 or len(cumhaz) == 0:
        return np.zeros_like(log_partial_hazard)
    u = rng.uniform(1e-12, 1 - 1e-12, size=len(log_partial_hazard))
    z = -np.log(u) / np.exp(log_partial_hazard)
    idx = np.searchsorted(cumhaz, z, side="left")
    t = np.empty_like(z, dtype=float)
    inside = idx < len(times)
    t[inside] = times[idx[inside]]
    outside = ~inside
    if np.any(outside):
        if len(times) >= 2:
            dt = times[-1] - times[-2]
            dH = cumhaz[-1] - cumhaz[-2]
            slope = dH / dt if dt > 0 else 0.0
        else:
            slope = 0.0
        extra = z[outside] - cumhaz[-1]
        add_t = np.where(slope <= 1e-12, 0.0, extra / max(slope, 1e-12))
        t[outside] = times[-1] + np.maximum(add_t, 0.0)
    t = np.maximum(t, 0.0)
    return t


def simulate_os_times_cox(
    cox_bundle: Dict,
    df_features: pd.DataFrame,
    os_labels: pd.DataFrame,
    adsl_for_arm: pd.DataFrame,
    n_sim: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate OS times using fitted Cox, tailored for ADaM ADSL/ADLB inputs.

    Returns long dataframe with columns: USUBJID, STUDYID, ARM, sim, time, event.
    """
    cox: CoxPHFitter = cox_bundle["model"]
    pre: ColumnTransformer = cox_bundle.get("preprocessor") or cox_bundle.get("pre")
    feat_names = cox_bundle.get("feature_names") or cox_bundle.get("feat_names")

    rng = np.random.default_rng(seed)

    base_feats = df_features.drop(columns=[c for c in ["time", "event"] if c in df_features.columns])
    X = _design_matrix(pre, feat_names, base_feats)

    # ARM mapping from ADSL
    arm_col = None
    for c in ["ATRT", "TRT", "ARM", "ARMCD"]:
        if c in adsl_for_arm.columns:
            arm_col = c
            break
    if arm_col is None:
        arm_map = adsl_for_arm[[ID_COL, STUDY_COL]].copy()
        arm_map["ARM"] = "ARM"
    else:
        arm_map = adsl_for_arm[[ID_COL, STUDY_COL, arm_col]].copy().rename(columns={arm_col: "ARM"})

    lab = os_labels[[ID_COL, STUDY_COL, "time", "event"]].rename(columns={"time": "time_obs", "event": "event_obs"})
    meta = lab.merge(arm_map, on=[ID_COL, STUDY_COL], how="left")

    lp = cox.predict_log_partial_hazard(X).values.reshape(-1)

    sims = []
    for s in range(n_sim):
        t_raw = _inverse_sample_from_baseline(cox, lp, rng)
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


