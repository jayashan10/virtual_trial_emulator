"""Preprocessing utilities tailored to the PDS310 data model."""

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("ohe", _make_ohe()),
        ]
    )
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
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_np.shape[1])]
    X_df = pd.DataFrame(X_np, columns=[str(c) for c in feature_names])
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


__all__ = [
    "_make_preprocessor",
    "_fit_transform",
    "_transform",
]
