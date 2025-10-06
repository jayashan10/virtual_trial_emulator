from typing import Dict
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter
from lifelines.exceptions import ConvergenceError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

ID_COL = "RPT"
STUDY_COL = "STUDYID"


def prepare_eot_competing(core: pd.DataFrame,
                          time_col: str = "ENTRT_PC",
                          reason_col: str = "ENDTRS_C") -> pd.DataFrame:
    df = core[[ID_COL, STUDY_COL, time_col, reason_col]].copy()
    df.rename(columns={time_col: "time", reason_col: "cause"}, inplace=True)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    df["cause"] = df["cause"].astype("string").str.strip()
    df["event_ae"] = df["cause"].isin(["AE", "possible_AE"]).astype(int)
    df["event_prog"] = (df["cause"] == "progression").astype(int)
    df["event_complete"] = (df["cause"] == "complete").astype(int)
    return df


def _prepare_X(df_joined: pd.DataFrame) -> (pd.DataFrame, OneHotEncoder):
    # Exclude non-predictor columns
    drop_cols = [ID_COL, STUDY_COL, "cause", "time", "event_ae", "event_prog", "event_complete"]
    predictors = df_joined.drop(columns=[c for c in drop_cols if c in df_joined.columns])

    numeric_cols = predictors.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in predictors.columns if c not in numeric_cols]

    X_num = predictors[numeric_cols].copy()
    X_cat = predictors[categorical_cols].copy()

    X_num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_num), columns=numeric_cols)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat = pd.DataFrame(ohe.fit_transform(X_cat), columns=ohe.get_feature_names_out(categorical_cols))

    X = pd.concat([X_num, X_cat], axis=1)
    variances = X.var(axis=0, numeric_only=True)
    keep = variances[variances > 1e-12].index.tolist()
    X = X[keep]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, ohe


def fit_cause_specific_cox_ae(df_joined: pd.DataFrame) -> Dict:
    X, ohe = _prepare_X(df_joined)
    data = X.copy()
    data["time"] = df_joined["time"].astype(float).values
    data["event"] = df_joined["event_ae"].astype(int).values
    model = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
    model.fit(data, duration_col="time", event_col="event")
    return {"model": model, "feature_cols": list(X.columns), "ohe": ohe}


def fit_eot_allcause(df_joined: pd.DataFrame) -> Dict:
    X, ohe = _prepare_X(df_joined)
    times = df_joined["time"].astype(float).values

    try:
        data = X.copy()
        scale = 100.0
        data["time"] = times / scale
        data["event"] = 1
        aft_w = WeibullAFTFitter(penalizer=0.1)
        aft_w._scipy_fit_method = "SLSQP"
        aft_w.fit(data, duration_col="time", event_col="event")
        return {"model": aft_w, "feature_cols": list(X.columns), "ohe": ohe, "time_scale": scale, "model_type": "weibull"}
    except Exception:
        pass

    try:
        data = X.copy()
        scale = 100.0
        data["time"] = times / scale
        data["event"] = 1
        aft_ln = LogNormalAFTFitter(penalizer=0.1)
        aft_ln._scipy_fit_method = "SLSQP"
        aft_ln.fit(data, duration_col="time", event_col="event")
        return {"model": aft_ln, "feature_cols": list(X.columns), "ohe": ohe, "time_scale": scale, "model_type": "lognormal"}
    except Exception:
        pass

    return {"model": None, "feature_cols": list(X.columns), "ohe": ohe, "empirical_times": times, "time_scale": 1.0, "model_type": "empirical"}
