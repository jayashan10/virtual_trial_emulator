"""
Digital Profile Builder for PDS310 Colorectal Cancer Digital Twin System.

This module creates comprehensive patient digital profiles by integrating all
available ADaM tables (ADSL, ADLB, ADLS, ADAE, ADPM, ADRSP, BIOMARK).

Following the CAMP methodology for digital twin generation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .io import ID_COL, STUDY_COL


# ============================================================================
# DIGITAL PROFILE CONSTRUCTION
# ============================================================================


def extract_demographics(adsl: pd.DataFrame, subjid: str) -> Dict[str, Any]:
    """
    Extract demographic features from ADSL.
    
    Features (12 total):
    - Identifiers: SUBJID, STUDYID
    - Demographics: AGE, SEX, RACE
    - Performance: B_ECOG (baseline ECOG status)
    - Anthropometrics: B_WEIGHT (baseline weight)
    - Disease: DIAGMONS (months since diagnosis), HISSUBTY (histology subtype),
               DIAGTYPE (diagnosis type)
    - Treatment: TRT (treatment arm), ATRT (actual treatment)
    - Prior history: SXANY (prior surgery indicator)
    """
    row = adsl[adsl[ID_COL] == subjid]
    if row.empty:
        return {}
    
    row = row.iloc[0]
    profile = {
        ID_COL: subjid,
        STUDY_COL: row.get(STUDY_COL, "PDS310"),
    }
    
    # Demographics
    for col in ["AGE", "SEX", "RACE"]:
        if col in row.index:
            profile[col] = row[col] if pd.notna(row[col]) else None
    
    # Performance status
    if "B_ECOG" in row.index:
        profile["B_ECOG"] = row["B_ECOG"] if pd.notna(row["B_ECOG"]) else None
    
    # Weight
    if "B_WEIGHT" in row.index:
        profile["B_WEIGHT"] = float(row["B_WEIGHT"]) if pd.notna(row["B_WEIGHT"]) else None
    
    # Disease characteristics
    for col in ["DIAGMONS", "HISSUBTY", "DIAGTYPE", "SXANY"]:
        if col in row.index:
            profile[col] = row[col] if pd.notna(row[col]) else None
    
    # Treatment
    for col in ["TRT", "ATRT"]:
        if col in row.index:
            profile[col] = row[col] if pd.notna(row[col]) else None
    
    return profile


def extract_baseline_labs(adlb: pd.DataFrame, subjid: str) -> Dict[str, Any]:
    """
    Extract baseline laboratory values from ADLB.
    
    Features (8 total):
    - ALB (albumin), ALP (alkaline phosphatase), CEA (carcinoembryonic antigen)
    - CREAT (creatinine), HGB (hemoglobin), LDH (lactate dehydrogenase)
    - PLT (platelets), WBC (white blood cell count)
    
    Baseline = earliest available value (minimum VISITDY)
    """
    if adlb is None or adlb.empty:
        return {}
    
    subj_labs = adlb[adlb[ID_COL] == subjid].copy()
    if subj_labs.empty:
        return {}
    
    # Convert VISITDY to numeric
    subj_labs["VISITDY"] = pd.to_numeric(subj_labs.get("VISITDY"), errors="coerce")
    
    # Get baseline (earliest) values for each lab test
    baseline_labs = {}
    
    # Map LBTEST names to standard names
    lab_mapping = {
        "ALBUMIN": "ALB",
        "ALP": "ALP",
        "CEA": "CEA",
        "CREATININE": "CREAT",
        "HGB": "HGB",
        "LDH": "LDH",
        "PLATELETS": "PLT",
        "WBC": "WBC",
    }
    
    for original_name, standard_name in lab_mapping.items():
        lab_data = subj_labs[subj_labs["LBTEST"].astype(str).str.upper() == original_name]
        if not lab_data.empty:
            # Get earliest value
            earliest = lab_data.sort_values("VISITDY").iloc[0]
            value = pd.to_numeric(earliest.get("LBSTRESN"), errors="coerce")
            baseline_labs[f"baseline_{standard_name}"] = float(value) if pd.notna(value) else None
    
    return baseline_labs


def extract_longitudinal_labs(
    adlb: pd.DataFrame,
    subjid: str,
    windows: Dict[str, tuple] = None
) -> Dict[str, Any]:
    """
    Extract longitudinal laboratory features from ADLB.
    
    Features (40 total for default setup):
    - For each of 5 labs × 2 windows × 4 metrics = 40 features
    - Labs: PSA, ALP, HGB, LDH, CREATININE
    - Windows: baseline (-30 to 0 days), early (1 to 60 days)
    - Metrics: last, mean, slope, count
    
    Example features:
    - lab_LDH_baseline_last
    - lab_LDH_baseline_mean
    - lab_LDH_baseline_slope
    - lab_LDH_baseline_count
    """
    if windows is None:
        windows = {"baseline": (-30, 0), "early": (1, 60)}
    
    if adlb is None or adlb.empty:
        return {}
    
    subj_labs = adlb[adlb[ID_COL] == subjid].copy()
    if subj_labs.empty:
        return {}
    
    subj_labs["VISITDY"] = pd.to_numeric(subj_labs.get("VISITDY"), errors="coerce")
    subj_labs["VALUE"] = pd.to_numeric(subj_labs.get("LBSTRESN"), errors="coerce")
    subj_labs = subj_labs.dropna(subset=["VISITDY", "VALUE"])
    
    # Track specific labs
    labs_to_track = ["PSA", "ALP", "HGB", "LDH", "CREATININE"]
    
    features = {}
    
    for lab_name in labs_to_track:
        lab_data = subj_labs[subj_labs["LBTEST"].astype(str).str.upper() == lab_name]
        
        for window_name, (day_min, day_max) in windows.items():
            window_data = lab_data[
                (lab_data["VISITDY"] >= day_min) &
                (lab_data["VISITDY"] <= day_max)
            ]
            
            prefix = f"lab_{lab_name}_{window_name}"
            
            if window_data.empty:
                features[f"{prefix}_last"] = None
                features[f"{prefix}_mean"] = None
                features[f"{prefix}_slope"] = None
                features[f"{prefix}_count"] = 0
            else:
                values = window_data["VALUE"].values
                days = window_data["VISITDY"].values
                
                features[f"{prefix}_last"] = float(values[-1])
                features[f"{prefix}_mean"] = float(np.mean(values))
                features[f"{prefix}_count"] = int(len(values))
                
                # Calculate slope if we have at least 2 points
                if len(values) >= 2 and len(np.unique(days)) > 1:
                    slope, _ = np.polyfit(days, values, 1)
                    features[f"{prefix}_slope"] = float(slope)
                else:
                    features[f"{prefix}_slope"] = None
    
    return features


def extract_tumor_characteristics(adls: pd.DataFrame, subjid: str) -> Dict[str, Any]:
    """
    Extract tumor/lesion characteristics from ADLS.
    
    Features (8 total):
    - target_lesion_count: Number of target lesions
    - nontarget_lesion_count: Number of non-target lesions
    - sum_target_diameters: Sum of longest diameters (baseline)
    - max_lesion_size: Largest lesion diameter
    - mean_lesion_size: Average lesion diameter
    - lesion_sites_count: Number of different anatomical sites
    - new_lesions_flag: Indicator if new lesions appeared
    - tumor_burden_category: Low/Medium/High based on sum of diameters
    """
    if adls is None or adls.empty:
        return {}
    
    subj_lesions = adls[adls[ID_COL] == subjid].copy()
    if subj_lesions.empty:
        return {}
    
    features = {}
    
    # Count lesion types
    if "LSCAT" in subj_lesions.columns:
        target_count = (subj_lesions["LSCAT"] == "TARGET").sum()
        nontarget_count = (subj_lesions["LSCAT"] == "NON-TARGET").sum()
        features["target_lesion_count"] = int(target_count)
        features["nontarget_lesion_count"] = int(nontarget_count)
    
    # Get baseline measurements (earliest visit)
    if "VISITDY" in subj_lesions.columns:
        subj_lesions["VISITDY"] = pd.to_numeric(subj_lesions["VISITDY"], errors="coerce")
        baseline_lesions = subj_lesions[subj_lesions["VISITDY"] == subj_lesions["VISITDY"].min()]
    else:
        baseline_lesions = subj_lesions
    
    # Extract diameter information
    if "LSSTRESN" in baseline_lesions.columns:
        diameters = pd.to_numeric(baseline_lesions["LSSTRESN"], errors="coerce").dropna()
        
        if not diameters.empty:
            # Target lesions only for sum
            target_diameters = pd.to_numeric(
                baseline_lesions[baseline_lesions.get("LSCAT") == "TARGET"]["LSSTRESN"],
                errors="coerce"
            ).dropna()
            
            features["sum_target_diameters"] = float(target_diameters.sum()) if not target_diameters.empty else 0.0
            features["max_lesion_size"] = float(diameters.max())
            features["mean_lesion_size"] = float(diameters.mean())
            
            # Tumor burden categorization
            sum_diameters = features["sum_target_diameters"]
            if sum_diameters < 50:
                features["tumor_burden_category"] = "low"
            elif sum_diameters < 100:
                features["tumor_burden_category"] = "medium"
            else:
                features["tumor_burden_category"] = "high"
        else:
            features["sum_target_diameters"] = None
            features["max_lesion_size"] = None
            features["mean_lesion_size"] = None
            features["tumor_burden_category"] = None
    
    # Count lesion sites
    if "LSLOC" in subj_lesions.columns:
        site_count = subj_lesions["LSLOC"].nunique()
        features["lesion_sites_count"] = int(site_count)
    
    # Check for new lesions (if LSNEW or similar column exists)
    if "LSNEW" in subj_lesions.columns:
        has_new = (subj_lesions["LSNEW"] == "Y").any()
        features["new_lesions_flag"] = int(has_new)
    else:
        features["new_lesions_flag"] = 0
    
    return features


def extract_molecular_biomarkers(biomark: pd.DataFrame, subjid: str) -> Dict[str, Any]:
    """
    Extract molecular biomarker data from BIOMARK.
    
    Features (8 total):
    - KRAS_exon2, KRAS_exon3, KRAS_exon4: KRAS mutation status by exon
    - NRAS_exon2, NRAS_exon3, NRAS_exon4: NRAS mutation status by exon
    - BRAF_exon15: BRAF mutation status
    - RAS_status: Combined RAS wild-type vs mutant classification
    
    Values: "WT" (wild-type), "MUT" (mutant), "UNK" (unknown)
    """
    if biomark is None or biomark.empty:
        return {}
    
    subj_biomark = biomark[biomark[ID_COL] == subjid]
    if subj_biomark.empty:
        return {}
    
    features = {}
    
    # PDS310 biomarker table structure: BMMTNM1-7 (names), BMMTR1-7 (results)
    # Map columns to feature names
    biomarker_mapping = {
        "BMMTR1": "KRAS_exon2",     # KRAS exon 2
        "BMMTR2": "KRAS_exon3",     # KRAS exon 3
        "BMMTR3": "KRAS_exon4",     # KRAS exon 4
        "BMMTR4": "NRAS_exon2",     # NRAS exon 2
        "BMMTR5": "NRAS_exon3",     # NRAS exon 3
        "BMMTR6": "NRAS_exon4",     # NRAS exon 4
        "BMMTR7": "BRAF_exon15",    # BRAF exon 15
    }
    
    # Extract each biomarker
    row = subj_biomark.iloc[0]
    for result_col, feature_name in biomarker_mapping.items():
        if result_col in row.index:
            result = row[result_col]
            if pd.notna(result) and result != "":
                # Normalize result values
                result_str = str(result).strip().lower()
                if "wild" in result_str or result_str == "wt":
                    features[feature_name] = "WT"
                elif "mutant" in result_str or "mut" in result_str:
                    features[feature_name] = "MUT"
                elif "failure" in result_str or "fail" in result_str:
                    features[feature_name] = "FAIL"
                else:
                    features[feature_name] = "UNK"
            else:
                features[feature_name] = "UNK"
        else:
            features[feature_name] = "UNK"
    
    # Derive combined RAS status
    ras_markers = ["KRAS_exon2", "KRAS_exon3", "KRAS_exon4", 
                   "NRAS_exon2", "NRAS_exon3", "NRAS_exon4"]
    
    ras_values = [features.get(m, "UNK") for m in ras_markers]
    
    # Count wild-type and mutant results
    wt_count = sum(1 for v in ras_values if v == "WT")
    mut_count = sum(1 for v in ras_values if v == "MUT")
    
    # Classify RAS status
    if mut_count > 0:
        # Any mutation = mutant
        features["RAS_status"] = "MUTANT"
    elif wt_count == len(ras_markers):
        # All wild-type = wild-type
        features["RAS_status"] = "WILD-TYPE"
    elif wt_count > 0 and mut_count == 0:
        # Some wild-type, some unknown/fail = likely wild-type
        features["RAS_status"] = "WILD-TYPE"
    else:
        # All unknown/fail
        features["RAS_status"] = "UNKNOWN"
    
    return features


def extract_physical_measurements(adpm: pd.DataFrame, subjid: str) -> Dict[str, Any]:
    """
    Extract physical measurement trends from ADPM.
    
    Features (4 total):
    - weight_baseline: Baseline weight (kg)
    - weight_change_abs: Absolute change from baseline (kg)
    - weight_change_pct: Percent change from baseline (%)
    - weight_trajectory: Categorical (stable/declining/increasing)
    
    Trajectory defined as:
    - declining: >5% loss
    - increasing: >5% gain
    - stable: within ±5%
    """
    if adpm is None or adpm.empty:
        return {}
    
    subj_pm = adpm[adpm[ID_COL] == subjid].copy()
    if subj_pm.empty:
        return {}
    
    features = {}
    
    # Focus on weight measurements
    if "PMTEST" in subj_pm.columns:
        weight_data = subj_pm[subj_pm["PMTEST"].astype(str).str.upper().str.contains("WEIGHT", na=False)]
        
        if not weight_data.empty and "PMSTRESN" in weight_data.columns:
            weight_data["VISITDY"] = pd.to_numeric(weight_data.get("VISITDY"), errors="coerce")
            weight_data["VALUE"] = pd.to_numeric(weight_data.get("PMSTRESN"), errors="coerce")
            weight_data = weight_data.dropna(subset=["VISITDY", "VALUE"]).sort_values("VISITDY")
            
            if not weight_data.empty:
                baseline_weight = weight_data.iloc[0]["VALUE"]
                latest_weight = weight_data.iloc[-1]["VALUE"]
                
                features["weight_baseline"] = float(baseline_weight)
                features["weight_change_abs"] = float(latest_weight - baseline_weight)
                features["weight_change_pct"] = float(
                    (latest_weight - baseline_weight) / baseline_weight * 100
                )
                
                # Categorize trajectory
                pct_change = features["weight_change_pct"]
                if pct_change < -5:
                    features["weight_trajectory"] = "declining"
                elif pct_change > 5:
                    features["weight_trajectory"] = "increasing"
                else:
                    features["weight_trajectory"] = "stable"
            else:
                features["weight_baseline"] = None
                features["weight_change_abs"] = None
                features["weight_change_pct"] = None
                features["weight_trajectory"] = None
    
    return features


def extract_prior_history(adae: pd.DataFrame, adsl: pd.DataFrame, subjid: str) -> Dict[str, Any]:
    """
    Extract prior treatment and AE history.
    
    Features (5 total):
    - prior_ae_count: Number of prior AEs
    - prior_severe_ae_count: Number of grade 3+ AEs
    - prior_skin_toxicity_flag: Any skin toxicity history
    - num_prior_therapies: Number of prior treatment lines
    - time_since_diagnosis: Months since initial diagnosis
    """
    features = {}
    
    # Prior AE history
    if adae is not None and not adae.empty:
        subj_ae = adae[adae[ID_COL] == subjid]
        if not subj_ae.empty:
            features["prior_ae_count"] = int(len(subj_ae))
            
            # Count severe AEs (grade 3+)
            if "AESEV" in subj_ae.columns:
                severe = subj_ae[subj_ae["AESEV"].astype(str).str.contains("3|4|5", na=False)]
                features["prior_severe_ae_count"] = int(len(severe))
            elif "AETOXGR" in subj_ae.columns:
                severe = subj_ae[pd.to_numeric(subj_ae["AETOXGR"], errors="coerce") >= 3]
                features["prior_severe_ae_count"] = int(len(severe))
            else:
                features["prior_severe_ae_count"] = 0
            
            # Check for skin toxicity
            if "AEDECOD" in subj_ae.columns:
                skin_terms = ["RASH", "DERMATITIS", "PRURITUS", "SKIN", "ACNE"]
                has_skin = subj_ae["AEDECOD"].astype(str).str.upper().str.contains(
                    "|".join(skin_terms), na=False
                ).any()
                features["prior_skin_toxicity_flag"] = int(has_skin)
            else:
                features["prior_skin_toxicity_flag"] = 0
        else:
            features["prior_ae_count"] = 0
            features["prior_severe_ae_count"] = 0
            features["prior_skin_toxicity_flag"] = 0
    else:
        features["prior_ae_count"] = 0
        features["prior_severe_ae_count"] = 0
        features["prior_skin_toxicity_flag"] = 0
    
    # Prior therapies from ADSL
    if adsl is not None and not adsl.empty:
        row = adsl[adsl[ID_COL] == subjid]
        if not row.empty:
            row = row.iloc[0]
            
            # Number of prior therapies (if available)
            if "NUMPRIOR" in row.index:
                features["num_prior_therapies"] = int(row["NUMPRIOR"]) if pd.notna(row["NUMPRIOR"]) else 0
            else:
                features["num_prior_therapies"] = 0
            
            # Time since diagnosis
            if "DIAGMONS" in row.index:
                features["time_since_diagnosis"] = float(row["DIAGMONS"]) if pd.notna(row["DIAGMONS"]) else None
            else:
                features["time_since_diagnosis"] = None
    
    return features


def compute_derived_risk_scores(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute derived risk scores from profile features.
    
    Features (6 total):
    - lab_risk_score: Composite of abnormal lab values (0-1 scale)
    - performance_risk: ECOG-based risk (0=good, 1-2=intermediate, 3-4=poor)
    - tumor_burden_risk: Based on lesion characteristics (0-1 scale)
    - molecular_risk: RAS mutation status (WT=0, MUT=1)
    - composite_risk_score: Weighted average of all risks
    - predicted_good_prognosis_flag: Binary indicator (1=good prognosis)
    """
    scores = {}
    
    # Lab risk score (count abnormal labs)
    lab_abnormal_count = 0
    lab_total = 0
    
    # Reference ranges (simplified)
    lab_ranges = {
        "baseline_ALB": (3.5, 5.5),    # g/dL
        "baseline_ALP": (30, 130),      # U/L
        "baseline_HGB": (12, 18),       # g/dL
        "baseline_LDH": (100, 250),     # U/L
        "baseline_CREAT": (0.6, 1.2),   # mg/dL
        "baseline_PLT": (150, 400),     # K/uL
        "baseline_WBC": (4, 11),        # K/uL
    }
    
    for lab_name, (low, high) in lab_ranges.items():
        value = profile.get(lab_name)
        if value is not None:
            lab_total += 1
            if value < low or value > high:
                lab_abnormal_count += 1
    
    scores["lab_risk_score"] = lab_abnormal_count / lab_total if lab_total > 0 else 0.5
    
    # Performance risk
    ecog = profile.get("B_ECOG")
    if ecog is not None and pd.notna(ecog):
        # Handle both numeric and text ECOG values
        if isinstance(ecog, str):
            # Map text to numeric
            ecog_map = {
                "Asymptomatic": 0,
                "Symptoms but ambulatory": 1,
                "In bed <50% of day": 2,
                "In bed >50% of day": 3,
                "Bedridden": 4,
            }
            ecog = ecog_map.get(ecog, None)
        else:
            try:
                ecog = int(ecog)
            except (ValueError, TypeError):
                ecog = None
    else:
        ecog = None
    
    if ecog is not None:
        if ecog == 0:
            scores["performance_risk"] = 0.0  # Good
        elif ecog in [1, 2]:
            scores["performance_risk"] = 0.5  # Intermediate
        else:
            scores["performance_risk"] = 1.0  # Poor
    else:
        scores["performance_risk"] = 0.5  # Unknown = intermediate
    
    # Tumor burden risk
    tumor_burden = profile.get("tumor_burden_category")
    if tumor_burden == "low":
        scores["tumor_burden_risk"] = 0.0
    elif tumor_burden == "medium":
        scores["tumor_burden_risk"] = 0.5
    elif tumor_burden == "high":
        scores["tumor_burden_risk"] = 1.0
    else:
        scores["tumor_burden_risk"] = 0.5  # Unknown
    
    # Molecular risk
    ras_status = profile.get("RAS_status")
    if ras_status == "WILD-TYPE":
        scores["molecular_risk"] = 0.0  # Good prognosis for EGFR inhibitor
    elif ras_status == "MUTANT":
        scores["molecular_risk"] = 1.0  # Poor prognosis for EGFR inhibitor
    else:
        scores["molecular_risk"] = 0.5  # Unknown
    
    # Composite risk (weighted average)
    weights = {
        "lab_risk_score": 0.25,
        "performance_risk": 0.30,
        "tumor_burden_risk": 0.25,
        "molecular_risk": 0.20,
    }
    
    composite = sum(scores[k] * weights[k] for k in weights.keys())
    scores["composite_risk_score"] = composite
    
    # Good prognosis flag (composite < 0.4)
    scores["predicted_good_prognosis_flag"] = int(composite < 0.4)
    
    return scores


def extract_outcomes(
    adsl: pd.DataFrame,
    adrsp: pd.DataFrame,
    subjid: str
) -> Dict[str, Any]:
    """
    Extract outcome data (survival, response).
    
    Features (8 total):
    - DTHDYX: Overall survival time (days)
    - DTHX: Death event indicator (0/1)
    - PFSDYCR: Progression-free survival time (days)
    - PFSCR: PFS event indicator (0/1)
    - best_response: Best overall response (CR/PR/SD/PD)
    - response_at_week8: Response at week 8 assessment
    - response_at_week16: Response at week 16 assessment
    - time_to_response: Days to first CR or PR
    """
    outcomes = {}
    
    # Overall survival from ADSL
    if adsl is not None and not adsl.empty:
        row = adsl[adsl[ID_COL] == subjid]
        if not row.empty:
            row = row.iloc[0]
            
            # OS
            if "DTHDYX" in row.index:
                outcomes["DTHDYX"] = float(row["DTHDYX"]) if pd.notna(row["DTHDYX"]) else None
            if "DTHX" in row.index:
                outcomes["DTHX"] = int(row["DTHX"]) if pd.notna(row["DTHX"]) else 0
            
            # PFS
            if "PFSDYCR" in row.index:
                outcomes["PFSDYCR"] = float(row["PFSDYCR"]) if pd.notna(row["PFSDYCR"]) else None
            if "PFSCR" in row.index:
                outcomes["PFSCR"] = int(row["PFSCR"]) if pd.notna(row["PFSCR"]) else 0
    
    # Response data from ADRSP
    if adrsp is not None and not adrsp.empty:
        subj_rsp = adrsp[adrsp[ID_COL] == subjid].copy()
        if not subj_rsp.empty:
            # Best overall response
            if "AVALC" in subj_rsp.columns:
                # Get best response (prioritize CR > PR > SD > PD)
                response_order = {"CR": 1, "PR": 2, "SD": 3, "PD": 4}
                subj_rsp["response_rank"] = subj_rsp["AVALC"].map(response_order)
                best = subj_rsp.sort_values("response_rank").iloc[0]
                outcomes["best_response"] = best["AVALC"] if pd.notna(best["AVALC"]) else "UNK"
            
            # Response at specific timepoints
            if "VISITDY" in subj_rsp.columns and "AVALC" in subj_rsp.columns:
                subj_rsp["VISITDY"] = pd.to_numeric(subj_rsp["VISITDY"], errors="coerce")
                
                # Week 8 (around day 56)
                week8 = subj_rsp[subj_rsp["VISITDY"].between(49, 63)]
                if not week8.empty:
                    outcomes["response_at_week8"] = week8.iloc[0]["AVALC"]
                else:
                    outcomes["response_at_week8"] = None
                
                # Week 16 (around day 112)
                week16 = subj_rsp[subj_rsp["VISITDY"].between(105, 119)]
                if not week16.empty:
                    outcomes["response_at_week16"] = week16.iloc[0]["AVALC"]
                else:
                    outcomes["response_at_week16"] = None
                
                # Time to response (first CR or PR)
                responders = subj_rsp[subj_rsp["AVALC"].isin(["CR", "PR"])]
                if not responders.empty:
                    first_response = responders.sort_values("VISITDY").iloc[0]
                    outcomes["time_to_response"] = float(first_response["VISITDY"])
                else:
                    outcomes["time_to_response"] = None
            else:
                outcomes["response_at_week8"] = None
                outcomes["response_at_week16"] = None
                outcomes["time_to_response"] = None
    
    return outcomes


def create_complete_digital_profile(
    subjid: str,
    tables: Dict[str, pd.DataFrame],
    include_outcomes: bool = True
) -> Dict[str, Any]:
    """
    Create comprehensive digital profile from all ADaM tables.
    
    Integrates ~100 features across 9 categories:
    1. Demographics (12 features)
    2. Baseline Labs (8 features)
    3. Longitudinal Labs (40 features)
    4. Tumor Characteristics (8 features)
    5. Molecular Biomarkers (8 features)
    6. Physical Measurements (4 features)
    7. Prior History (5 features)
    8. Derived Risk Scores (6 features)
    9. Outcomes (8 features) - optional
    
    Args:
        subjid: Subject identifier
        tables: Dictionary of ADaM DataFrames (keys: adsl, adlb, adls, adae, adpm, adrsp, biomark)
        include_outcomes: Whether to include outcome data (default: True)
    
    Returns:
        Dictionary containing complete patient digital profile
    """
    profile = {}
    
    # 1. Demographics
    if "adsl" in tables:
        profile.update(extract_demographics(tables["adsl"], subjid))
    
    # 2. Baseline Labs
    if "adlb" in tables:
        profile.update(extract_baseline_labs(tables["adlb"], subjid))
    
    # 3. Longitudinal Labs
    if "adlb" in tables:
        profile.update(extract_longitudinal_labs(tables["adlb"], subjid))
    
    # 4. Tumor Characteristics
    if "adls" in tables:
        profile.update(extract_tumor_characteristics(tables["adls"], subjid))
    
    # 5. Molecular Biomarkers
    if "biomark" in tables:
        profile.update(extract_molecular_biomarkers(tables["biomark"], subjid))
    
    # 6. Physical Measurements
    if "adpm" in tables:
        profile.update(extract_physical_measurements(tables["adpm"], subjid))
    
    # 7. Prior History
    profile.update(extract_prior_history(
        tables.get("adae"),
        tables.get("adsl"),
        subjid
    ))
    
    # 8. Derived Risk Scores
    profile.update(compute_derived_risk_scores(profile))
    
    # 9. Outcomes (optional)
    if include_outcomes:
        profile.update(extract_outcomes(
            tables.get("adsl"),
            tables.get("adrsp"),
            subjid
        ))
    
    return profile


def get_profile_feature_groups() -> Dict[str, List[str]]:
    """
    Return dictionary mapping feature group names to feature lists.
    
    Useful for feature selection and visualization.
    """
    return {
        "identifiers": [ID_COL, STUDY_COL],
        "demographics": ["AGE", "SEX", "RACE", "B_ECOG", "B_WEIGHT"],
        "disease": ["DIAGMONS", "HISSUBTY", "DIAGTYPE", "SXANY"],
        "treatment": ["TRT", "ATRT"],
        "baseline_labs": [
            "baseline_ALB", "baseline_ALP", "baseline_CEA", "baseline_CREAT",
            "baseline_HGB", "baseline_LDH", "baseline_PLT", "baseline_WBC"
        ],
        "tumor": [
            "target_lesion_count", "nontarget_lesion_count", "sum_target_diameters",
            "max_lesion_size", "mean_lesion_size", "lesion_sites_count",
            "new_lesions_flag", "tumor_burden_category"
        ],
        "molecular": [
            "KRAS_exon2", "KRAS_exon3", "KRAS_exon4",
            "NRAS_exon2", "NRAS_exon3", "NRAS_exon4",
            "BRAF_exon15", "RAS_status"
        ],
        "physical": [
            "weight_baseline", "weight_change_abs", "weight_change_pct", "weight_trajectory"
        ],
        "history": [
            "prior_ae_count", "prior_severe_ae_count", "prior_skin_toxicity_flag",
            "num_prior_therapies", "time_since_diagnosis"
        ],
        "risk_scores": [
            "lab_risk_score", "performance_risk", "tumor_burden_risk",
            "molecular_risk", "composite_risk_score", "predicted_good_prognosis_flag"
        ],
        "outcomes": [
            "DTHDYX", "DTHX", "PFSDYCR", "PFSCR",
            "best_response", "response_at_week8", "response_at_week16", "time_to_response"
        ]
    }
