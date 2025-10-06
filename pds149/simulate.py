import os
from typing import Dict
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

ID_COL = "RPT"
STUDY_COL = "STUDYID"


def _predict_linear_cox(model: CoxPHFitter, X: pd.DataFrame) -> np.ndarray:
    ph = model.predict_partial_hazard(X).values.reshape(-1)
    return np.log(ph + 1e-12)


def _encode_with_ohe(df_feat: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    numeric_cols = df_feat.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df_feat.columns if c not in numeric_cols]
    X_num = df_feat[numeric_cols].copy()
    X_num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_num), columns=numeric_cols)
    X_cat = pd.DataFrame(ohe.transform(df_feat[categorical_cols]), columns=ohe.get_feature_names_out(categorical_cols))
    X = pd.concat([X_num, X_cat], axis=1)
    return X


def _sample_eot_times(eot_model: Dict, X: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    mtype = eot_model.get("model_type")
    if mtype in ("weibull", "lognormal") and eot_model.get("model") is not None:
        model = eot_model["model"]
        scale = float(eot_model.get("time_scale", 1.0))
        # Use 50th percentile as baseline and add variability via percentiles
        u = rng.uniform(0.1, 0.9, size=len(X))
        try:
            t_scaled = model.predict_quantile(X, u)
        except Exception:
            t_scaled = model.predict_median(X)
        return t_scaled.values.reshape(-1) * scale
    # Empirical fallback
    times = eot_model.get("empirical_times")
    if times is not None and len(times) > 0:
        return rng.choice(times, size=len(X), replace=True)
    # Absolute fallback
    return np.full(len(X), 60.0)


def simulate_patients(df_features: pd.DataFrame,
                      aft_allcause: Dict,
                      cox_ae: CoxPHFitter,
                      ohe: OneHotEncoder,
                      feature_cols: list,
                      n_sim: int = 1,
                      seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feat_base = df_features.drop(columns=[c for c in [ID_COL, STUDY_COL] if c in df_features.columns])
    X = _encode_with_ohe(feat_base, ohe)
    # Align to training columns
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    sims = []
    for sim in range(n_sim):
        t_eot = _sample_eot_times(aft_allcause, X, rng)
        lp_ae = _predict_linear_cox(cox_ae, X)
        p_ae = 1.0 / (1.0 + np.exp(-lp_ae))
        u = rng.uniform(0, 1, size=len(X))
        event_ae = (u < p_ae).astype(int)
        sims.append(pd.DataFrame({
            ID_COL: df_features[ID_COL].values if ID_COL in df_features.columns else np.arange(len(X)),
            STUDY_COL: df_features[STUDY_COL].values if STUDY_COL in df_features.columns else "NA",
            "sim": sim,
            "t_eot": t_eot,
            "event_ae": event_ae,
        }))
    out = pd.concat(sims, ignore_index=True)
    return out
