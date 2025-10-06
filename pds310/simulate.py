from typing import Dict
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


ID_COL = "SUBJID"
STUDY_COL = "STUDYID"


def _predict_linear_cox(model: CoxPHFitter, X: pd.DataFrame) -> np.ndarray:
    ph = model.predict_partial_hazard(X).values.reshape(-1)
    return np.log(ph + 1e-12)


def _encode_with_ohe(df_feat: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    numeric_cols = df_feat.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df_feat.columns if c not in numeric_cols]
    X_num = df_feat[numeric_cols].copy()
    non_empty_numeric = [c for c in numeric_cols if X_num[c].notna().any()]
    X_num = X_num[non_empty_numeric]
    X_num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_num), columns=non_empty_numeric)
    X_cat = pd.DataFrame(ohe.transform(df_feat[categorical_cols]), columns=ohe.get_feature_names_out(categorical_cols))
    X = pd.concat([X_num, X_cat], axis=1)
    return X


def simulate_patients(
    df_features: pd.DataFrame,
    aft_allcause: Dict,
    cox_ae: CoxPHFitter,
    ohe: OneHotEncoder,
    feature_cols: list,
    adsl: pd.DataFrame,
    n_sim: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ARM mapping
    arm_col = None
    for c in ["TRT", "ARM", "ARMCD"]:
        if c in adsl.columns:
            arm_col = c
            break
    if arm_col is None:
        arm_map = adsl[[ID_COL, STUDY_COL]].copy()
        arm_map["ARM"] = "ARM"
    else:
        arm_map = adsl[[ID_COL, STUDY_COL, arm_col]].copy().rename(columns={arm_col: "ARM"})

    feat_base = df_features.drop(columns=[c for c in [ID_COL, STUDY_COL] if c in df_features.columns])
    X = _encode_with_ohe(feat_base, ohe)
    # Align to training columns
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    # Sample EOT times
    mtype = aft_allcause.get("model_type")
    if mtype in ("weibull", "lognormal") and aft_allcause.get("model") is not None:
        model = aft_allcause["model"]
        scale = float(aft_allcause.get("time_scale", 1.0))
        u = rng.uniform(0.1, 0.9, size=len(X))
        try:
            t_scaled = model.predict_quantile(X, u)
        except Exception:
            t_scaled = model.predict_median(X)
        t_eot = t_scaled.values.reshape(-1) * scale
    else:
        times = aft_allcause.get("empirical_times")
        if times is not None and len(times) > 0:
            t_eot = rng.choice(times, size=len(X), replace=True)
        else:
            t_eot = np.full(len(X), 60.0)

    # AE event probability via Cox
    lp_ae = _predict_linear_cox(cox_ae, X)
    p_ae = 1.0 / (1.0 + np.exp(-lp_ae))
    u = rng.uniform(0, 1, size=len(X))
    event_ae = (u < p_ae).astype(int)

    # Build base output per simulation
    sims = []
    base = df_features[[ID_COL, STUDY_COL]].copy()
    base = base.merge(arm_map, on=[ID_COL, STUDY_COL], how="left")
    for sim in range(n_sim):
        sims.append(
            pd.DataFrame(
                {
                    ID_COL: base[ID_COL].values,
                    STUDY_COL: base[STUDY_COL].values,
                    "ARM": base["ARM"].values,
                    "sim": sim,
                    "t_eot": t_eot,
                    "event_ae": event_ae,
                }
            )
        )
    out = pd.concat(sims, ignore_index=True)
    return out


