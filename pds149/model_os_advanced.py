from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold

from sksurv.ensemble import RandomSurvivalForest
try:
    from sksurv.gradient_boosting import GradientBoostingSurvivalAnalysis
except Exception:  # fallback older API
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis  # type: ignore
from sksurv.metrics import concordance_index_censored

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
        ("scale", StandardScaler(with_mean=True, with_std=True)),
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


def _y_structured(df: pd.DataFrame):
    # scikit-survival expects structured array with event as boolean
    return np.array([(bool(e), float(t)) for e, t in zip(df["event"].values, df["time"].values)],
                    dtype=[("event", bool), ("time", float)])


def fit_rsf(df: pd.DataFrame, groups: pd.Series) -> Dict:
    pre = _make_preprocessor(df)
    X = pre.fit_transform(df.drop(columns=["time", "event"]))
    y = _y_structured(df)

    rsf = RandomSurvivalForest(
        n_estimators=300,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rsf.fit(X, y)

    n_groups = pd.Series(groups).nunique()
    cidx_scores: List[float] = []
    if n_groups >= 2:
        gkf = GroupKFold(n_splits=min(5, n_groups))
        for tr, te in gkf.split(df, groups=groups):
            pre_cv = _make_preprocessor(df.iloc[tr])
            X_tr = pre_cv.fit_transform(df.iloc[tr].drop(columns=["time", "event"]))
            y_tr = _y_structured(df.iloc[tr])
            model = RandomSurvivalForest(
                n_estimators=300,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            )
            model.fit(X_tr, y_tr)
            X_te = pre_cv.transform(df.iloc[te].drop(columns=["time", "event"]))
            risk = -model.predict(X_te)  # lower is better survival; use negative as risk
            ci = concordance_index_censored(df.iloc[te]["event"].astype(bool).values, df.iloc[te]["time"].values, risk)[0]
            cidx_scores.append(float(ci))
    cv_mean = float(np.mean(cidx_scores)) if cidx_scores else float("nan")
    return {"model": rsf, "pre": pre, "cv_cindex_mean": cv_mean}


def fit_gb(df: pd.DataFrame, groups: pd.Series) -> Dict:
    pre = _make_preprocessor(df)
    X = pre.fit_transform(df.drop(columns=["time", "event"]))
    y = _y_structured(df)

    gb = GradientBoostingSurvivalAnalysis(learning_rate=0.05, max_depth=3, n_estimators=300, random_state=42)
    gb.fit(X, y)

    n_groups = pd.Series(groups).nunique()
    cidx_scores: List[float] = []
    if n_groups >= 2:
        gkf = GroupKFold(n_splits=min(5, n_groups))
        for tr, te in gkf.split(df, groups=groups):
            pre_cv = _make_preprocessor(df.iloc[tr])
            X_tr = pre_cv.fit_transform(df.iloc[tr].drop(columns=["time", "event"]))
            y_tr = _y_structured(df.iloc[tr])
            model = GradientBoostingSurvivalAnalysis(learning_rate=0.05, max_depth=3, n_estimators=300, random_state=42)
            model.fit(X_tr, y_tr)
            X_te = pre_cv.transform(df.iloc[te].drop(columns=["time", "event"]))
            risk = -model.predict(X_te)
            ci = concordance_index_censored(df.iloc[te]["event"].astype(bool).values, df.iloc[te]["time"].values, risk)[0]
            cidx_scores.append(float(ci))
    cv_mean = float(np.mean(cidx_scores)) if cidx_scores else float("nan")
    return {"model": gb, "pre": pre, "cv_cindex_mean": cv_mean}
