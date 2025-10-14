from typing import List
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold, StratifiedKFold

from .preprocessing import _make_preprocessor, _fit_transform, _transform


def _stratified_indices(event: pd.Series, n_splits: int, random_state: int) -> List:
    y = event.astype(int).values
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(skf.split(y, y))
    except Exception:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(kf.split(y))


def cox_kfold_cindex(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> float:
    if len(df) < n_splits:
        return float("nan")
    splits = _stratified_indices(df["event"], n_splits, random_state)
    scores: List[float] = []
    for tr, te in splits:
        pre = _make_preprocessor(df.iloc[tr])
        X_tr_df, feat_tr = _fit_transform(pre, df.iloc[tr])
        X_tr_df = X_tr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_tr_df["time"] = df.iloc[tr]["time"].astype(float).values
        X_tr_df["event"] = df.iloc[tr]["event"].astype(int).values

        model = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
        model.fit(X_tr_df, duration_col="time", event_col="event")

        X_te_df = _transform(pre, df.iloc[te], feat_tr)
        X_te_df = X_te_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        surv = model.predict_partial_hazard(X_te_df)
        cidx = concordance_index(
            event_times=df.iloc[te]["time"].values,
            predicted_scores=-surv.values.ravel(),
            event_observed=df.iloc[te]["event"].values,
        )
        scores.append(float(cidx))
    return float(np.mean(scores)) if scores else float("nan")


def _make_preprocessor_adv(df: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for drop_col in ["time", "event"]:
        if drop_col in num_cols:
            num_cols.remove(drop_col)
    cat_cols = [c for c in df.columns if c not in num_cols + ["time", "event"]]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("ohe", ohe),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )
    return pre


def rsf_kfold_cindex(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> float:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored

    if len(df) < n_splits:
        return float("nan")
    splits = _stratified_indices(df["event"], n_splits, random_state)
    scores: List[float] = []
    for tr, te in splits:
        pre = _make_preprocessor_adv(df.iloc[tr])
        X_tr = pre.fit_transform(df.iloc[tr].drop(columns=["time", "event"]))
        y_tr = np.array([(bool(e), float(t)) for e, t in zip(df.iloc[tr]["event"].values, df.iloc[tr]["time"].values)],
                        dtype=[("event", bool), ("time", float)])
        model = RandomSurvivalForest(
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
        model.fit(X_tr, y_tr)
        X_te = pre.transform(df.iloc[te].drop(columns=["time", "event"]))
        risk = -model.predict(X_te)
        ci = concordance_index_censored(df.iloc[te]["event"].astype(bool).values, df.iloc[te]["time"].values, risk)[0]
        scores.append(float(ci))
    return float(np.mean(scores)) if scores else float("nan")


def gb_kfold_cindex(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> float:
    try:
        from sksurv.gradient_boosting import GradientBoostingSurvivalAnalysis
    except Exception:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis  # type: ignore
    from sksurv.metrics import concordance_index_censored

    if len(df) < n_splits:
        return float("nan")
    splits = _stratified_indices(df["event"], n_splits, random_state)
    scores: List[float] = []
    for tr, te in splits:
        pre = _make_preprocessor_adv(df.iloc[tr])
        X_tr = pre.fit_transform(df.iloc[tr].drop(columns=["time", "event"]))
        y_tr = np.array([(bool(e), float(t)) for e, t in zip(df.iloc[tr]["event"].values, df.iloc[tr]["time"].values)],
                        dtype=[("event", bool), ("time", float)])
        model = GradientBoostingSurvivalAnalysis(learning_rate=0.05, max_depth=3, n_estimators=300, random_state=random_state)
        model.fit(X_tr, y_tr)
        X_te = pre.transform(df.iloc[te].drop(columns=["time", "event"]))
        risk = -model.predict(X_te)
        ci = concordance_index_censored(df.iloc[te]["event"].astype(bool).values, df.iloc[te]["time"].values, risk)[0]
        scores.append(float(ci))
    return float(np.mean(scores)) if scores else float("nan")


