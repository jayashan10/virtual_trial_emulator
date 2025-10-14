"""
Response Classification Model for PDS310.

Predicts best overall response (CR/PR/SD/PD) using patient features.
Following CAMP methodology target: 90%+ accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.impute import SimpleImputer


def prepare_response_data(
    profile_db: pd.DataFrame,
    exclude_outcomes: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for response classification.

    Args:
        profile_db: Patient profiles with features and outcomes.
        exclude_outcomes: Whether to exclude survival outcome features to avoid leakage.

    Returns:
        X: Raw feature matrix (no encoding/imputation applied).
        y: Raw multi-class response labels (CR/PR/SD/PD/NE).
        y_simplified: Optional collapsed labels (Responder/Stable/Progressor/Unknown).
    """
    df = profile_db[profile_db["best_response"].notna()].copy()
    if len(df) == 0:
        df = profile_db[profile_db["response_at_week8"].notna()].copy()
        response_col = "response_at_week8"
    else:
        response_col = "best_response"

    if df.empty:
        raise ValueError("No patients with response labels available.")

    y = df[response_col].astype("string")

    response_mapping = {
        "CR": "Responder",
        "PR": "Responder",
        "SD": "Stable",
        "PD": "Progressor",
        "NE": "Unknown",
    }
    y_simplified = y.map(response_mapping).fillna("Unknown")

    exclude_cols = {
        "SUBJID",
        "STUDYID",
        "best_response",
        "response_at_week8",
        "response_at_week16",
        "time_to_response",
    }
    if exclude_outcomes:
        exclude_cols.update({"DTHDYX", "DTHX", "PFSDYCR", "PFSCR"})
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

    # Drop columns that are entirely missing – they add no information and break imputers.
    all_missing = [col for col in X.columns if X[col].isna().all()]
    if all_missing:
        X = X.drop(columns=all_missing)

    # Normalise dtypes for downstream pipelines.
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype("string")

    return X, y, y_simplified


def _split_feature_types(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prepare feature matrix for modelling and identify numeric / categorical columns.

    Returns:
        X_clean: Copy of X with boolean columns cast to float and categorical columns to pandas string dtype.
        numeric_cols: List of numeric feature names.
        categorical_cols: List of categorical feature names.
    """
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
    return X_clean[ordered_cols], numeric_cols, categorical_cols


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


def _get_pipeline_feature_names(
    pipeline: Pipeline,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> np.ndarray:
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
    profile: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    ordered_cols: List[str],
) -> pd.DataFrame:
    X = profile.copy()
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


def prepare_features_for_model(
    model_bundle: Dict[str, Any],
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare raw profile dataframe for the stored response model pipeline.
    """
    if isinstance(profiles, dict):
        profiles = pd.DataFrame([profiles])
    numeric_cols: List[str] = model_bundle.get("numeric_features", [])
    categorical_cols: List[str] = model_bundle.get("categorical_features", [])
    ordered_cols: List[str] = model_bundle.get("raw_feature_names", numeric_cols + categorical_cols)
    return _prepare_prediction_frame(profiles.copy(), numeric_cols, categorical_cols, ordered_cols)


def train_response_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "rf",
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train response classification model.

    Args:
        X: Raw feature matrix.
        y: Response labels.
        model_type: "logreg", "rf", "gb", or "xgboost" (falls back to GB if XGBoost unavailable).
        cv_folds: Stratified CV folds.
        random_state: RNG seed.
    """
    X_clean, numeric_cols, categorical_cols = _split_feature_types(X)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier

            estimator = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                eval_metric="mlogloss",
            )
        except Exception:
            print("XGBoost not available, falling back to GradientBoostingClassifier.")
            estimator = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=random_state,
            )
    elif model_type == "gb":
        estimator = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
        )
    elif model_type == "rf":
        estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "logreg":
        estimator = LogisticRegression(
            multi_class="multinomial",
            penalty="l2",
            class_weight="balanced",
            solver="saga",
            C=0.5,
            max_iter=5000,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    steps = []
    if preprocessor is not None:
        steps.append(("preprocess", preprocessor))
    steps.extend(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("model", estimator),
        ]
    )
    if model_type == "logreg":
        steps.insert(-1, ("scale", MaxAbsScaler()))
    pipeline = Pipeline(steps)

    cv = StratifiedKFold(
        n_splits=min(cv_folds, len(np.unique(y))),
        shuffle=True,
        random_state=random_state,
    )

    unique_classes = np.unique(y)
    cv_accuracy: List[float] = []
    cv_balanced_accuracy: List[float] = []
    cv_macro_f1: List[float] = []
    cv_reports: List[Dict[str, Any]] = []

    for train_idx, test_idx in cv.split(X_clean, y):
        pipeline.fit(X_clean.iloc[train_idx], y.iloc[train_idx])
        y_true_fold = y.iloc[test_idx]
        y_pred_fold = pipeline.predict(X_clean.iloc[test_idx])

        cv_accuracy.append(accuracy_score(y_true_fold, y_pred_fold))
        cv_balanced_accuracy.append(balanced_accuracy_score(y_true_fold, y_pred_fold))
        cv_macro_f1.append(f1_score(y_true_fold, y_pred_fold, average="macro", zero_division=0))

        cv_reports.append(
            classification_report(
                y_true_fold,
                y_pred_fold,
                labels=unique_classes,
                output_dict=True,
                zero_division=0,
            )
        )

    print(
        "Cross-validation metrics:\n"
        f"  Accuracy: {np.mean(cv_accuracy):.3f} ± {np.std(cv_accuracy):.3f}\n"
        f"  Balanced Accuracy: {np.mean(cv_balanced_accuracy):.3f} ± {np.std(cv_balanced_accuracy):.3f}\n"
        f"  Macro F1: {np.mean(cv_macro_f1):.3f} ± {np.std(cv_macro_f1):.3f}"
    )

    pipeline.fit(X_clean, y)
    estimator_fitted = pipeline.named_steps["model"]
    classes = estimator_fitted.classes_

    y_pred = pipeline.predict(X_clean)
    y_proba = pipeline.predict_proba(X_clean) if hasattr(pipeline, "predict_proba") else None

    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=0)
    train_balanced_accuracy = balanced_accuracy_score(y, y_pred)
    train_macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(y, y_pred, labels=classes)
    class_report = classification_report(y, y_pred, labels=classes, output_dict=True, zero_division=0)

    feature_names_out = _get_pipeline_feature_names(pipeline, numeric_cols, categorical_cols)
    if hasattr(estimator_fitted, "feature_importances_"):
        feature_importance = (
            pd.DataFrame(
                {
                    "feature": feature_names_out,
                    "importance": estimator_fitted.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    else:
        feature_importance = None

    cv_scores = np.array(cv_accuracy)

    results: Dict[str, Any] = {
        "model": pipeline,
        "model_type": model_type,
        "cv_scores": cv_scores,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_balanced_accuracy_mean": float(np.mean(cv_balanced_accuracy)),
        "cv_balanced_accuracy_std": float(np.std(cv_balanced_accuracy)),
        "cv_macro_f1_mean": float(np.mean(cv_macro_f1)),
        "cv_macro_f1_std": float(np.std(cv_macro_f1)),
        "train_accuracy": float(accuracy),
        "train_precision": float(precision),
        "train_recall": float(recall),
        "train_f1": float(f1),
        "train_balanced_accuracy": float(train_balanced_accuracy),
        "train_macro_f1": float(train_macro_f1),
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "cv_classification_reports": cv_reports,
        "feature_importance": feature_importance,
        "feature_names": feature_names_out.tolist(),
        "raw_feature_names": (numeric_cols + categorical_cols),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "classes": classes.tolist(),
        "n_samples": int(len(X_clean)),
    }

    # Optional ROC-AUC for binary/multi-class via OvR
    if y_proba is not None:
        try:
            if len(classes) == 2:
                roc_auc = roc_auc_score(y, y_proba[:, 1])
            else:
                from sklearn.preprocessing import label_binarize

                y_bin = label_binarize(y, classes=classes)
                if y_bin.shape[1] > 1:
                    roc_auc = roc_auc_score(y_bin, y_proba, average="weighted", multi_class="ovr")
                else:
                    roc_auc = None
            if roc_auc is not None:
                results["roc_auc"] = float(roc_auc)
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"Could not calculate ROC-AUC: {exc}")

    return results


def predict_response_probability(
    model_bundle: Dict[str, Any],
    profile: pd.DataFrame,
) -> Dict[str, float]:
    """
    Predict response probability for a patient profile.

    Args:
        model_bundle: Trained model results dictionary.
        profile: Patient profile (dict or single-row DataFrame).
    """
    if isinstance(profile, dict):
        profile_df = pd.DataFrame([profile])
    else:
        profile_df = profile.copy()

    pipeline: Pipeline = model_bundle["model"]
    numeric_cols: List[str] = model_bundle.get("numeric_features", [])
    categorical_cols: List[str] = model_bundle.get("categorical_features", [])
    ordered_cols: List[str] = model_bundle.get("raw_feature_names", numeric_cols + categorical_cols)

    X = _prepare_prediction_frame(profile_df, numeric_cols, categorical_cols, ordered_cols)
    proba = pipeline.predict_proba(X)[0]
    classes = pipeline.named_steps["model"].classes_

    prob_dict = {class_name: float(prob) for class_name, prob in zip(classes, proba)}
    prob_dict["predicted_class"] = classes[np.argmax(proba)]
    return prob_dict


def save_response_model(results: Dict[str, Any], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, output_path)
    print(f"Response model saved to: {output_path}")


def load_response_model(model_path: str) -> Dict[str, Any]:
    results = joblib.load(model_path)
    print(f"Loaded response model from: {model_path}")
    print(f"  Model type: {results.get('model_type')}")
    print(f"  CV accuracy: {results.get('cv_mean', float('nan')):.3f}")
    print(f"  Classes: {results.get('classes')}")
    return results


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
    output_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    if feature_importance is None or feature_importance.empty:
        print("No feature importance data available")
        return

    top_features = feature_importance.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features["importance"].values, color="steelblue")
    plt.yticks(range(len(top_features)), top_features["feature"].values)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Features for Response Prediction")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Feature importance plot saved to: {output_path}")

    plt.close()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Response Classification Confusion Matrix")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix plot saved to: {output_path}")

    plt.close()
