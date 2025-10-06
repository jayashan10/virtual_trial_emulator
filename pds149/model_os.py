from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer

ID_COL = "RPT"
STUDY_COL = "STUDYID"


def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for drop_col in ["time", "event"]:
        if drop_col in numeric_cols:
            numeric_cols.remove(drop_col)
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ["time", "event"]]
    return numeric_cols, categorical_cols


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = _split_columns(df)
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("ohe", _make_ohe()),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )
    return pre


def _fit_transform(pre: ColumnTransformer, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    X_np = pre.fit_transform(df.drop(columns=["time", "event"]))
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_np.shape[1])]
    X_df = pd.DataFrame(X_np, columns=[str(c) for c in feat_names])
    variances = X_df.var(axis=0, numeric_only=True)
    keep = variances[variances > 1e-12].index.tolist()
    X_df = X_df[keep]
    return X_df, list(X_df.columns)


def _transform(pre: ColumnTransformer, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    X_np = pre.transform(df.drop(columns=["time", "event"]))
    X_df_full = pd.DataFrame(X_np)
    try:
        full_names = [str(c) for c in pre.get_feature_names_out()]
    except Exception:
        full_names = [f"f{i}" for i in range(X_df_full.shape[1])]
    X_df_full.columns = full_names
    X_df = X_df_full[feature_names]
    return X_df


def _adaptive_cv_splits(groups: pd.Series, max_splits: int = 5) -> int:
    n_groups = pd.Series(groups).nunique()
    if n_groups < 2:
        return 0
    return min(max_splits, n_groups)


def fit_cox(df: pd.DataFrame, groups: pd.Series) -> Dict:
    pre = _make_preprocessor(df)
    X_df, feat_names = _fit_transform(pre, df)
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_df["time"] = df["time"].astype(float).values
    X_df["event"] = df["event"].astype(int).values

    cox = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
    cox.fit(X_df, duration_col="time", event_col="event")

    n_splits = _adaptive_cv_splits(groups)
    cidx_scores = []
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        from lifelines.utils import concordance_index
        for tr, te in gkf.split(df, groups=groups):
            pre_cv = _make_preprocessor(df.iloc[tr])
            X_tr_df, feat_tr = _fit_transform(pre_cv, df.iloc[tr])
            X_tr_df = X_tr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X_tr_df["time"] = df.iloc[tr]["time"].astype(float).values
            X_tr_df["event"] = df.iloc[tr]["event"].astype(int).values
            model = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
            model.fit(X_tr_df, duration_col="time", event_col="event")

            X_te_df = _transform(pre_cv, df.iloc[te], feat_tr)
            X_te_df = X_te_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            surv = model.predict_partial_hazard(X_te_df)
            cidx = concordance_index(
                event_times=df.iloc[te]["time"].values,
                predicted_scores=-surv.values.ravel(),
                event_observed=df.iloc[te]["event"].values,
            )
            cidx_scores.append(cidx)
    cv_mean = float(np.mean(cidx_scores)) if cidx_scores else float("nan")

    return {"model": cox, "preprocessor": pre, "feature_names": feat_names, "cv_cindex_mean": cv_mean}


def fit_weibull_aft(df: pd.DataFrame, groups: pd.Series) -> Dict:
    pre = _make_preprocessor(df)
    X_df, feat_names = _fit_transform(pre, df)
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_df["time"] = df["time"].astype(float).values
    X_df["event"] = df["event"].astype(int).values

    aft = WeibullAFTFitter(penalizer=0.01)
    aft.fit(X_df, duration_col="time", event_col="event")

    n_splits = _adaptive_cv_splits(groups)
    cidx_scores = []
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        from lifelines.utils import concordance_index
        for tr, te in gkf.split(df, groups=groups):
            pre_cv = _make_preprocessor(df.iloc[tr])
            X_tr_df, feat_tr = _fit_transform(pre_cv, df.iloc[tr])
            X_tr_df = X_tr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X_tr_df["time"] = df.iloc[tr]["time"].astype(float).values
            X_tr_df["event"] = df.iloc[tr]["event"].astype(int).values
            model = WeibullAFTFitter(penalizer=0.01)
            model.fit(X_tr_df, duration_col="time", event_col="event")

            X_te_df = _transform(pre_cv, df.iloc[te], feat_tr)
            X_te_df = X_te_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            preds = model.predict_median(X_te_df)
            cidx = concordance_index(
                event_times=df.iloc[te]["time"].values,
                predicted_scores=-preds.values.ravel(),
                event_observed=df.iloc[te]["event"].values,
            )
            cidx_scores.append(cidx)
    cv_mean = float(np.mean(cidx_scores)) if cidx_scores else float("nan")

    return {"model": aft, "preprocessor": pre, "feature_names": feat_names, "cv_cindex_mean": cv_mean}


def fit_weibull_aft_robust(df: pd.DataFrame, groups: pd.Series) -> Dict:
    """Robust AFT fitting with scaling and LogNormal fallback. Returns keys 'model','pre','feat_names'."""
    pre = _make_preprocessor(df)
    X_df, feat_names = _fit_transform(pre, df)
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_df["time"] = (df["time"].astype(float) / 100.0).values  # scale times to aid convergence
    X_df["event"] = df["event"].astype(int).values

    # Try Weibull with SLSQP
    try:
        aft = WeibullAFTFitter(penalizer=0.05)
        aft._scipy_fit_method = "SLSQP"
        aft.fit(X_df, duration_col="time", event_col="event")
        return {"model": aft, "pre": pre, "feat_names": feat_names, "cv_cindex_mean": float("nan")}
    except Exception:
        pass

    # Fallback to LogNormal with SLSQP
    aft_ln = LogNormalAFTFitter(penalizer=0.05)
    aft_ln._scipy_fit_method = "SLSQP"
    aft_ln.fit(X_df, duration_col="time", event_col="event")
    return {"model": aft_ln, "pre": pre, "feat_names": feat_names, "cv_cindex_mean": float("nan")}
