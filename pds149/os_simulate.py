from typing import Dict
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from sklearn.compose import ColumnTransformer

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


def simulate_os_times(aft_bundle: Dict,
                      df_features: pd.DataFrame,
                      os_labels: pd.DataFrame,
                      core_for_arm: pd.DataFrame,
                      n_sim: int = 50,
                      seed: int = 42) -> pd.DataFrame:
    model: WeibullAFTFitter = aft_bundle["model"]
    pre: ColumnTransformer = aft_bundle["pre"]
    feat_names = aft_bundle["feat_names"]

    rng = np.random.default_rng(seed)

    # Build design matrix with the same preprocessor
    base_feats = df_features.drop(columns=[c for c in ["time", "event"] if c in df_features.columns])
    X = _design_matrix(pre, feat_names, base_feats)

    # Join censoring times from observed labels and ARM from core
    arm_map = core_for_arm[[ID_COL, STUDY_COL, "TRT1_ID", "TRT2_ID", "TRT3_ID"]].copy()
    arm_map["ARM"] = build_arm_identifier(arm_map)
    arm_map = arm_map[[ID_COL, STUDY_COL, "ARM"]]

    lab = os_labels[[ID_COL, STUDY_COL, "time", "event"]].rename(columns={"time": "time_obs", "event": "event_obs"})
    meta = lab.merge(arm_map, on=[ID_COL, STUDY_COL], how="left")

    sims = []
    for s in range(n_sim):
        u = rng.uniform(1e-6, 1 - 1e-6, size=len(X))
        try:
            t = model.predict_quantile(X, u)
            t = t.values.reshape(-1)
        except Exception:
            t = model.predict_median(X).values.reshape(-1)
        # Apply censoring from observed
        time = np.minimum(t, meta["time_obs"].values.astype(float))
        event = (t <= meta["time_obs"].values.astype(float)).astype(int)
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
