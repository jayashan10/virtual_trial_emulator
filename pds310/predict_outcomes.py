"""
Multi-Outcome Prediction System for PDS310 Digital Twins.

Provides a unified interface for response, time-to-response, biomarker, and
overall-survival prediction. Each outcome can be evaluated under counterfactual
treatment assignments to support virtual-trial simulations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from pds310.digital_profile import get_profile_feature_groups
from pds310.model_response import prepare_features_for_model as prepare_response_features
from pds310.model_ttr import prepare_features_for_model as prepare_ttr_features
from pds310.model_survival import prepare_features_for_model as prepare_os_features


_PROFILE_GROUPS = get_profile_feature_groups()
LEAKAGE_FEATURES = set(_PROFILE_GROUPS.get("risk_scores", [])) | set(_PROFILE_GROUPS.get("outcomes", []))

TREATMENT_NORMALISATION = {
    "panitumumab + bsc": "panit. plus best supportive care",
    "panitumumab plus best supportive care": "panit. plus best supportive care",
    "panitumab + bsc": "panit. plus best supportive care",
    "panit. plus best supportive care": "panit. plus best supportive care",
    "best supportive care": "Best supportive care",
    "bsc": "Best supportive care",
}


def _to_serialisable(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


class OutcomePredictor:
    """
    Unified outcome prediction for digital twins.

    Predicts (per treatment arm):
      - Response classification (CR/PR/SD/PD)
      - Time-to-response (days)
      - Overall survival (sampled event time + censoring indicator)
      - Biomarker trajectories (optional)
    """

    def __init__(
        self,
        response_model_path: Optional[str] = None,
        ttr_model_path: Optional[str] = None,
        biomarker_model_path: Optional[str] = None,
        survival_model_path: Optional[str] = None,
        ae_model_path: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.models: Dict[str, Any] = {}
        self.random_seed = random_seed

        # Load response model
        if response_model_path and Path(response_model_path).exists():
            self.models["response"] = joblib.load(response_model_path)
            print("✅ Loaded response model")
        else:
            self.models["response"] = None
            print("⚠️  Response model not loaded")

        # Load TTR model
        if ttr_model_path and Path(ttr_model_path).exists():
            self.models["ttr"] = joblib.load(ttr_model_path)
            print("✅ Loaded TTR model")
        else:
            self.models["ttr"] = None
            print("⚠️  TTR model not loaded")

        # Load biomarker models
        if biomarker_model_path and Path(biomarker_model_path).exists():
            self.models["biomarkers"] = joblib.load(biomarker_model_path)
            print("✅ Loaded biomarker models")
        else:
            self.models["biomarkers"] = None
            print("⚠️  Biomarker models not loaded")

        # Load survival model
        if survival_model_path and Path(survival_model_path).exists():
            self.models["os"] = joblib.load(survival_model_path)
            print("✅ Loaded OS model")
        else:
            self.models["os"] = None
            print("⚠️  OS model not loaded")

        # Placeholder for AE models
        if ae_model_path and Path(ae_model_path).exists():
            self.models["ae"] = joblib.load(ae_model_path)
        else:
            self.models["ae"] = None

    # ------------------------------------------------------------------ #
    # Core helpers
    # ------------------------------------------------------------------ #

    def _ensure_dataframe(self, profile: Any) -> pd.DataFrame:
        if isinstance(profile, pd.DataFrame):
            return profile.copy()
        if isinstance(profile, pd.Series):
            return profile.to_frame().T
        if isinstance(profile, dict):
            return pd.DataFrame([profile])
        raise TypeError(f"Unsupported profile type: {type(profile)!r}")

    def _canonical_treatment_label(self, value: Any) -> Any:
        if value is None:
            return value
        text = str(value).strip()
        if not text:
            return value
        normalised = TREATMENT_NORMALISATION.get(text.lower())
        return normalised if normalised is not None else value

    def _normalise_treatment_columns(self, profile_df: pd.DataFrame) -> pd.DataFrame:
        for col in ("TRT", "ATRT"):
            if col in profile_df.columns:
                profile_df[col] = profile_df[col].apply(self._canonical_treatment_label)
        return profile_df

    def _strip_leakage_features(self, profile_df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [col for col in LEAKAGE_FEATURES if col in profile_df.columns]
        if drop_cols:
            return profile_df.drop(columns=drop_cols)
        return profile_df

    def _apply_treatment(self, profile_df: pd.DataFrame, arm_name: str) -> pd.DataFrame:
        adjusted = profile_df.copy()
        canonical = self._canonical_treatment_label(arm_name)
        for col in ("TRT", "ATRT"):
            if col in adjusted.columns:
                adjusted[col] = canonical
        return adjusted

    def _safe_call(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Run callable, wrapping exceptions into an error dict."""
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # Individual outcome predictors
    # ------------------------------------------------------------------ #

    def _predict_response(self, profile_df: pd.DataFrame) -> Dict[str, Any]:
        bundle = self.models.get("response")
        if bundle is None:
            return {"error": "Response model not loaded"}

        pipeline = bundle["model"]
        classes = pipeline.named_steps["model"].classes_

        X = prepare_response_features(bundle, profile_df)
        proba = pipeline.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_class = classes[pred_idx]

        return {
            "predicted_response": pred_class,
            "probabilities": {class_name: float(prob) for class_name, prob in zip(classes, proba)},
            "confidence": float(proba[pred_idx]),
        }

    def _predict_ttr(self, profile_df: pd.DataFrame) -> Dict[str, Any]:
        bundle = self.models.get("ttr")
        if bundle is None:
            return {"error": "TTR model not loaded"}

        pipeline = bundle["model"]
        X = prepare_ttr_features(bundle, profile_df)
        pred = float(pipeline.predict(X)[0])

        rmse = float(bundle.get("train_rmse", 10))
        return {
            "predicted_ttr": pred,
            "lower_95ci": float(pred - 1.96 * rmse),
            "upper_95ci": float(pred + 1.96 * rmse),
            "uncertainty": rmse,
        }

    def _sample_from_baseline(self, cox_model, log_partial_hazard: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        baseline = cox_model.baseline_cumulative_hazard_
        if baseline.empty:
            return np.full_like(log_partial_hazard, np.nan, dtype=float)

        times = baseline.index.values.astype(float)
        cumhaz = baseline.iloc[:, 0].values.astype(float)

        u = rng.uniform(1e-12, 1 - 1e-12, size=len(log_partial_hazard))
        z = -np.log(u) / np.exp(log_partial_hazard)

        idx = np.searchsorted(cumhaz, z, side="left")
        sampled = np.empty_like(z, dtype=float)
        inside = idx < len(times)
        sampled[inside] = times[idx[inside]]

        outside = ~inside
        if np.any(outside):
            if len(times) >= 2:
                dt = times[-1] - times[-2]
                dH = cumhaz[-1] - cumhaz[-2]
                slope = dH / dt if dt > 0 else 0.0
            else:
                slope = 0.0
            extra = z[outside] - cumhaz[-1]
            add_t = np.where(slope <= 1e-12, 0.0, extra / max(slope, 1e-12))
            sampled[outside] = times[-1] + np.maximum(add_t, 0.0)

        return np.maximum(sampled, 0.0)

    def _predict_os(self, profile_df: pd.DataFrame, rng: Optional[np.random.Generator] = None, max_followup: Optional[float] = None) -> Dict[str, Any]:
        bundle = self.models.get("os")
        if bundle is None:
            return {"error": "OS model not loaded"}

        features = prepare_os_features(bundle, profile_df)
        cox_model = bundle["model"]

        log_partial = cox_model.predict_log_partial_hazard(features).values.reshape(-1)
        rng = rng or np.random.default_rng(self.random_seed)
        sampled = self._sample_from_baseline(cox_model, log_partial, rng)

        max_follow = float(max_followup) if max_followup is not None else float("inf")
        raw_time = float(sampled[0]) if sampled.size else float("nan")
        time = float(min(raw_time, max_follow))
        event = int(raw_time <= max_follow) if not np.isnan(raw_time) else 0

        return {
            "time": time,
            "event": event,
            "raw_time": raw_time,
            "partial_hazard": float(np.exp(log_partial[0])),
            "log_partial_hazard": float(log_partial[0]),
        }

    def _predict_biomarkers(self, profile_df: pd.DataFrame, biomarkers: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        bundle = self.models.get("biomarkers")
        if bundle is None:
            return {"error": "Biomarker models not loaded"}

        all_models = bundle
        requested = list(biomarkers) if biomarkers is not None else list(all_models.keys())
        predictions: Dict[str, Dict[int, Dict[str, float]]] = {}

        for biomarker in requested:
            if biomarker not in all_models:
                continue
            biomarker_preds: Dict[int, Dict[str, float]] = {}
            for timepoint, model_data in all_models[biomarker].items():
                model = model_data["model"]
                feature_names = model_data["feature_names"]
                X = self._prepare_features(profile_df, feature_names)
                pred = float(model.predict(X)[0])
                rmse = float(model_data.get("train_rmse", abs(pred) * 0.1))
                biomarker_preds[int(timepoint)] = {
                    "predicted": pred,
                    "lower_95ci": float(pred - 1.96 * rmse),
                    "upper_95ci": float(pred + 1.96 * rmse),
                }
            predictions[biomarker] = biomarker_preds

        return predictions

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict_all(
        self,
        profile: Any,
        treatment_arm: Optional[str] = None,
        include_biomarkers: bool = True,
        rng: Optional[np.random.Generator] = None,
        max_followup: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Predict all outcomes for a patient under an optional treatment arm.
        """
        profile_df = self._ensure_dataframe(profile)
        profile_df = self._strip_leakage_features(profile_df)
        profile_df = self._normalise_treatment_columns(profile_df)
        if treatment_arm is not None:
            profile_df = self._apply_treatment(profile_df, treatment_arm)

        results: Dict[str, Any] = {
            "patient_id": profile_df.iloc[0].get("SUBJID", "UNKNOWN"),
            "treatment_arm": treatment_arm or profile_df.iloc[0].get("ATRT"),
        }

        response = self._safe_call(self._predict_response, profile_df)
        results["response"] = _to_serialisable(response)

        if response.get("predicted_response") in {"CR", "PR"}:
            ttr = self._safe_call(self._predict_ttr, profile_df)
        elif "error" in response:
            ttr = {"error": "Response unavailable"}
        else:
            ttr = {"note": "Not applicable (non-responder)"}
        results["time_to_response"] = _to_serialisable(ttr)

        os_pred = self._safe_call(self._predict_os, profile_df, rng=rng, max_followup=max_followup)
        results["overall_survival"] = _to_serialisable(os_pred)

        if include_biomarkers:
            biomarkers_pred = self._safe_call(self._predict_biomarkers, profile_df, None)
            results["biomarkers"] = _to_serialisable(biomarkers_pred)

        return results

    def predict_counterfactuals(
        self,
        profile: Any,
        arm_names: Iterable[str],
        include_biomarkers: bool = False,
        max_followup: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Predict outcomes for each treatment arm (counterfactual analysis).
        """
        base_seed = self.random_seed if seed is None else seed
        profile_df = self._ensure_dataframe(profile)

        results: Dict[str, Dict[str, Any]] = {}
        for idx, arm in enumerate(arm_names):
            rng = np.random.default_rng(base_seed + idx)
            pred = self.predict_all(
                profile_df,
                treatment_arm=arm,
                include_biomarkers=include_biomarkers,
                rng=rng,
                max_followup=max_followup,
            )
            results[arm] = pred
        return results

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #

    def _prepare_features(self, profile: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """
        Prepare features for simple scikit-learn style models (biomarkers).
        """
        X = profile.copy()
        for feat in feature_names:
            if feat not in X.columns:
                X[feat] = np.nan

        X = X[feature_names]
        categorical_cols = X.select_dtypes(include=["object", "string"]).columns
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isna().all():
                X[col] = 0.0
            else:
                X[col] = X[col].fillna(X[col].median())
        return X

    def predict_cohort(
        self,
        profiles: pd.DataFrame,
        include_biomarkers: bool = False,
        verbose: bool = True,
        treatment_arm: Optional[str] = None,
        max_followup: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Predict outcomes for an entire cohort of patients.
        """
        results: List[Dict[str, Any]] = []
        for idx, (_, row) in enumerate(profiles.iterrows()):
            if verbose and (idx + 1) % 100 == 0:
                print(f"  Predicted {idx + 1}/{len(profiles)} patients...")

            row_df = row.to_frame().T
            arm = treatment_arm or row.get("ATRT")
            pred = self.predict_all(
                row_df,
                treatment_arm=arm,
                include_biomarkers=include_biomarkers,
                max_followup=max_followup,
            )

            flat: Dict[str, Any] = {
                "SUBJID": pred.get("patient_id"),
                "treatment_arm": pred.get("treatment_arm"),
            }

            resp = pred.get("response", {})
            flat["predicted_response"] = resp.get("predicted_response")
            flat["response_confidence"] = resp.get("confidence")
            for cls, prob in resp.get("probabilities", {}).items():
                flat[f"prob_{cls}"] = prob

            ttr = pred.get("time_to_response", {})
            if "predicted_ttr" in ttr:
                flat["predicted_ttr"] = ttr["predicted_ttr"]
                flat["ttr_lower_95ci"] = ttr.get("lower_95ci")
                flat["ttr_upper_95ci"] = ttr.get("upper_95ci")

            os_pred = pred.get("overall_survival", {})
            flat["os_time"] = os_pred.get("time")
            flat["os_event"] = os_pred.get("event")
            flat["os_partial_hazard"] = os_pred.get("partial_hazard")

            results.append(_to_serialisable(flat))

        return pd.DataFrame(results)


# ---------------------------------------------------------------------- #
# Convenience helpers
# ---------------------------------------------------------------------- #


def create_outcome_predictor(
    models_dir: str = "outputs/pds310/models",
    random_seed: int = 42,
) -> OutcomePredictor:
    models_path = Path(models_dir)
    predictor = OutcomePredictor(
        response_model_path=str(models_path / "response_model.joblib"),
        ttr_model_path=str(models_path / "ttr_model.joblib"),
        biomarker_model_path=str(models_path / "biomarker_models.joblib"),
        survival_model_path=str(models_path / "os_model.joblib"),
        random_seed=random_seed,
    )
    return predictor


def save_predictions(predictions: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def summarize_predictions(predictions: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_patients": int(len(predictions)),
    }

    if "predicted_response" in predictions.columns:
        dist = predictions["predicted_response"].value_counts().to_dict()
        summary["predicted_response_distribution"] = {k: int(v) for k, v in dist.items()}
        responders = predictions["predicted_response"].isin(["CR", "PR"]).sum()
        if len(predictions):
            summary["predicted_response_rate"] = float(responders / len(predictions) * 100)

    if "predicted_ttr" in predictions.columns:
        ttr_vals = predictions["predicted_ttr"].dropna()
        if len(ttr_vals) > 0:
            summary["predicted_ttr"] = {
                "mean": float(ttr_vals.mean()),
                "median": float(ttr_vals.median()),
                "std": float(ttr_vals.std()),
            }

    if "os_time" in predictions.columns:
        os_vals = predictions["os_time"].dropna()
        if len(os_vals) > 0:
            summary["simulated_os"] = {
                "median": float(os_vals.median()),
                "mean": float(os_vals.mean()),
            }

    return summary

