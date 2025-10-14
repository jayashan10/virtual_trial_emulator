"""
Overall Survival (OS) modelling utilities for PDS310 digital profiles.

Fits a Cox proportional hazards model using engineered patient profiles and
stores the preprocessing pipeline so downstream simulation can evaluate
counterfactual survival under different treatment assignments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone
from sklearn.model_selection import KFold


def prepare_os_data(
    profile_db: pd.DataFrame,
    duration_col: str = "DTHDYX",
    event_col: str = "DTHX",
    min_duration: float = 1.0,
    exclude_outcomes: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare digital profiles for OS modelling.

    Args:
        profile_db: Patient digital profile database.
        duration_col: Column containing survival durations (days).
        event_col: Column containing event indicator (1=event, 0=censored).
        min_duration: Minimum positive duration to retain.
        exclude_outcomes: Whether to drop downstream outcome columns to avoid leakage.

    Returns:
        X: Feature matrix with baseline covariates.
        durations: Survival durations.
        events: Event indicators.
    """
    if duration_col not in profile_db.columns or event_col not in profile_db.columns:
        raise ValueError(f"Required columns '{duration_col}' and '{event_col}' not found in profiles.")

    df = profile_db.copy()
    df = df[df[duration_col].notna() & df[event_col].notna()].copy()
    df = df[df[duration_col] >= min_duration].copy()

    if df.empty:
        raise ValueError("No valid records available for OS modelling.")

    durations = pd.to_numeric(df[duration_col], errors="coerce")
    events = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)

    exclude_cols = {
        "SUBJID",
        "STUDYID",
        duration_col,
        event_col,
    }
    if exclude_outcomes:
        exclude_cols.update({"PFSDYCR", "PFSCR", "best_response", "response_at_week8", "response_at_week16", "time_to_response"})
    exclude_cols.update({
        "lab_risk_score",
        "performance_risk",
        "tumor_burden_risk",
        "molecular_risk",
        "composite_risk_score",
        "predicted_good_prognosis_flag",
    })

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()

    categorical_candidate_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    if categorical_candidate_cols:
        X = _collapse_rare_categories(X, categorical_candidate_cols)

    # Drop completely-missing columns which break imputers.
    all_missing = [col for col in X.columns if X[col].isna().all()]
    if all_missing:
        X = X.drop(columns=all_missing)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype("string")

    return X, durations, events


def _split_feature_types(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    X_clean = X.copy()

    bool_cols = X_clean.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X_clean[bool_cols] = X_clean[bool_cols].astype(float)

    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if X_clean[c].notna().any()]

    categorical_cols = [c for c in X_clean.columns if c not in numeric_cols]
    categorical_cols = [c for c in categorical_cols if X_clean[c].notna().any()]

    for col in categorical_cols:
        X_clean[col] = X_clean[col].astype("object")
        X_clean[col] = X_clean[col].where(X_clean[col].notna(), np.nan)

    ordered_cols = numeric_cols + categorical_cols
    return X_clean[ordered_cols], numeric_cols, categorical_cols, ordered_cols


def _collapse_rare_categories(
    df: pd.DataFrame,
    columns: List[str],
    min_count: int = 10,
    other_label: str = "Other",
) -> pd.DataFrame:
    """Replace infrequent categorical levels with a shared label to improve model stability."""

    collapsed = df.copy()
    threshold = max(1, min_count)

    for col in columns:
        if col not in collapsed.columns:
            continue

        series = collapsed[col]
        if series.isna().all():
            continue

        counts = series.value_counts(dropna=True)
        rare_mask = counts < threshold
        if not rare_mask.any():
            continue

        rare_levels = counts[rare_mask].index
        rare_total = int(counts[rare_mask].sum())

        if rare_total < threshold:
            collapsed[col] = series.where(~series.isin(rare_levels), np.nan)
            continue

        collapsed[col] = series.apply(
            lambda value: other_label
            if (pd.notna(value) and value in rare_levels)
            else value
        )

    return collapsed


def _build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> Optional[ColumnTransformer]:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        return None
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def _get_pipeline_feature_names(pipeline: Pipeline, numeric_cols: List[str], categorical_cols: List[str]) -> np.ndarray:
    preprocess = pipeline.named_steps.get("preprocess")
    if preprocess is not None:
        feature_names = preprocess.get_feature_names_out()
    else:
        feature_names = np.array(numeric_cols + categorical_cols, dtype=object)

    variance = pipeline.named_steps.get("variance")
    if variance is not None:
        support = variance.get_support()
        feature_names = np.asarray(feature_names)[support]
    return feature_names


def _prepare_prediction_frame(
    profiles: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    ordered_cols: List[str],
) -> pd.DataFrame:
    X = profiles.copy()
    for col in numeric_cols:
        if col not in X.columns:
            X[col] = np.nan
    for col in categorical_cols:
        if col not in X.columns:
            X[col] = pd.Series([np.nan] * len(X), dtype="object")

    X = X.reindex(columns=ordered_cols)

    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(float)
    for col in categorical_cols:
        X[col] = X[col].astype("object")
        X[col] = X[col].where(X[col].notna(), np.nan)
    return X


def train_os_model(
    X: pd.DataFrame,
    durations: pd.Series,
    events: pd.Series,
    penalizer: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a Cox proportional hazards model on digital profile features.
    """
    X_clean, numeric_cols, categorical_cols, ordered_cols = _split_feature_types(X)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    steps: List[Tuple[str, Any]] = []
    if preprocessor is not None:
        steps.append(("preprocess", preprocessor))
    steps.append(("variance", VarianceThreshold(threshold=0.0)))
    processing = Pipeline(steps)

    X_processed = processing.fit_transform(X_clean)
    feature_names = _get_pipeline_feature_names(processing, numeric_cols, categorical_cols)

    X_df = pd.DataFrame(X_processed, columns=feature_names)
    X_df["duration"] = durations.values
    X_df["event"] = events.values.astype(int)

    penalties = [penalizer, penalizer * 5, penalizer * 10]
    last_error: Optional[Exception] = None
    cox = CoxPHFitter(penalizer=penalizer)
    for pen in penalties:
        try:
            cox = CoxPHFitter(penalizer=pen)
            cox.fit(X_df, duration_col="duration", event_col="event", show_progress=False)
            last_error = None
            break
        except ConvergenceError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error

    # Cross-validated concordance index using explicit folds
    k = min(5, max(2, len(X_clean) // 50))
    cv_scores: List[float] = []
    if k >= 2:
        splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
        for train_idx, test_idx in splitter.split(X_clean):
            processing_fold: Pipeline = clone(processing)
            X_train_processed = processing_fold.fit_transform(X_clean.iloc[train_idx])
            feature_names_fold = _get_pipeline_feature_names(processing_fold, numeric_cols, categorical_cols)

            train_df = pd.DataFrame(X_train_processed, columns=feature_names_fold)
            train_df["duration"] = durations.iloc[train_idx].values
            train_df["event"] = events.iloc[train_idx].astype(int).values

            cox_fold: Optional[CoxPHFitter] = None
            for pen in [cox.penalizer, cox.penalizer * 5, cox.penalizer * 10]:
                try:
                    cox_fold = CoxPHFitter(penalizer=pen)
                    cox_fold.fit(train_df, duration_col="duration", event_col="event", show_progress=False)
                    break
                except ConvergenceError:
                    cox_fold = None
                    continue

            if cox_fold is None:
                continue

            X_test_processed = processing_fold.transform(X_clean.iloc[test_idx])
            test_df = pd.DataFrame(X_test_processed, columns=feature_names_fold)
            durations_test = durations.iloc[test_idx].values
            events_test = events.iloc[test_idx].astype(int).values

            if np.sum(events_test) == 0:
                continue

            risk_scores = cox_fold.predict_partial_hazard(test_df).values.ravel()
            ci = concordance_index(durations_test, risk_scores, events_test)
            if np.isfinite(ci):
                cv_scores.append(ci)

    results: Dict[str, Any] = {
        "model": cox,
        "processing": processing,
        "feature_names": feature_names,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "raw_feature_names": ordered_cols,
        "train_concordance": float(cox.concordance_index_),
        "cv_cindex_mean": float(np.mean(cv_scores)) if cv_scores else float("nan"),
        "cv_cindex_std": float(np.std(cv_scores)) if cv_scores else float("nan"),
        "duration_col": "duration",
        "event_col": "event",
        "penalizer": float(penalizer),
    }
    return results


def prepare_features_for_model(
    model_bundle: Dict[str, Any],
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare raw profiles for survival prediction using the stored pipeline.
    """
    if isinstance(profiles, dict):
        profiles = pd.DataFrame([profiles])

    numeric_cols: List[str] = model_bundle.get("numeric_features", [])
    categorical_cols: List[str] = model_bundle.get("categorical_features", [])
    ordered_cols: List[str] = model_bundle.get("raw_feature_names", numeric_cols + categorical_cols)

    X = _prepare_prediction_frame(profiles.copy(), numeric_cols, categorical_cols, ordered_cols)
    processing: Pipeline = model_bundle["processing"]
    transformed = processing.transform(X)

    feature_names = model_bundle.get("feature_names")
    if feature_names is None:
        raise ValueError("Survival model bundle missing 'feature_names'.")

    return pd.DataFrame(transformed, columns=feature_names, index=profiles.index)


def save_os_model(model_bundle: Dict[str, Any], filepath: str) -> None:
    """
    Persist trained OS model bundle to disk.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, filepath)
