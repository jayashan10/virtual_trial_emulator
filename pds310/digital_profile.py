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
from .labs import canonical_lab_list, canonical_lab_name


# ============================================================================
# DIGITAL PROFILE CONSTRUCTION
# ============================================================================


def extract_demographics(adsl: pd.DataFrame, subjid: str) -> Dict[str, Any]:
    """
    Extract demographic features from ADSL.
    
    Features (11 total):
    - Identifiers: SUBJID, STUDYID
    - Demographics: AGE, SEX, RACE
    - Performance: B_ECOG (baseline ECOG status)
    - Anthropometrics: B_WEIGHT (baseline weight)
    - Disease: DIAGMONS (months since diagnosis), HISSUBTY (histology subtype),
               DIAGTYPE (diagnosis type)
    - Treatment: ATRT (actual treatment)
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
    
    # Treatment (actual received only)
    if "ATRT" in row.index:
        profile["ATRT"] = row["ATRT"] if pd.notna(row["ATRT"]) else None
    
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
    
    subj_labs["CANON"] = subj_labs["LBTEST"].map(canonical_lab_name)
    
    baseline_labs: Dict[str, Any] = {}
    for code in canonical_lab_list():
        lab_data = subj_labs[subj_labs["CANON"] == code]
        if lab_data.empty:
            baseline_labs[f"baseline_{code}"] = None
            continue
        earliest = lab_data.sort_values("VISITDY").iloc[0]
        value = pd.to_numeric(earliest.get("LBSTRESN"), errors="coerce")
        baseline_labs[f"baseline_{code}"] = float(value) if pd.notna(value) else None
    
    return baseline_labs


def extract_longitudinal_labs(
    adlb: pd.DataFrame,
    subjid: str,
    windows: Dict[str, tuple] = None
) -> Dict[str, Any]:
    """
    Extract longitudinal laboratory features from ADLB.
    
    Features (16 total for default setup):
    - For each of 8 labs × 2 metrics = 16 features
    - Labs: ALB, ALP, CEA, CREAT, HGB, LDH, PLT, WBC
    - Window: early (1 to 42 days)
    - Metrics: last, slope
    
    Example features:
    - lab_LDH_early_last
    - lab_LDH_early_slope
    
    NOTE: Early window restricted to day 42 (2-week buffer before Week 8 at day 56)
    to prevent temporal leakage. Week 8 labs reflect treatment response.
    """
    if windows is None:
        # CRITICAL FIX: Restrict early window to day 42 (prevent Week 8 leakage)
        # Week 8 assessment at day 56, need 2-week safety buffer
        windows = {"early": (1, 42)}
    
    if adlb is None or adlb.empty:
        return {}
    
    subj_labs = adlb[adlb[ID_COL] == subjid].copy()
    if subj_labs.empty:
        return {}
    
    subj_labs["VISITDY"] = pd.to_numeric(subj_labs.get("VISITDY"), errors="coerce")
    subj_labs["VALUE"] = pd.to_numeric(subj_labs.get("LBSTRESN"), errors="coerce")
    subj_labs = subj_labs.dropna(subset=["VISITDY", "VALUE"])
    
    subj_labs["CANON"] = subj_labs["LBTEST"].map(canonical_lab_name)
    subj_labs = subj_labs.dropna(subset=["CANON"])
    
    features = {}
    
    for short_code in canonical_lab_list():
        lab_data = subj_labs[subj_labs["CANON"] == short_code]

        for window_name, (day_min, day_max) in windows.items():
            if window_name != "early":
                continue

            window_data = lab_data[
                (lab_data["VISITDY"] >= day_min) &
                (lab_data["VISITDY"] <= day_max)
            ]

            prefix = f"lab_{short_code}_{window_name}"

            if window_data.empty:
                features[f"{prefix}_last"] = None
                features[f"{prefix}_slope"] = None
                continue

            values = window_data["VALUE"].values
            days = window_data["VISITDY"].values

            features[f"{prefix}_last"] = float(values[-1])

            if len(values) >= 2 and len(np.unique(days)) > 1:
                slope, _ = np.polyfit(days, values, 1)
                features[f"{prefix}_slope"] = float(slope)
            else:
                features[f"{prefix}_slope"] = None
    
    return features


def extract_tumor_characteristics(adls: pd.DataFrame, subjid: str) -> Dict[str, Any]:
    """
    Extract tumor/lesion characteristics from ADLS (SCREENING ONLY).
    
    Features (7 total):
    - target_lesion_count: Number of target lesions
    - nontarget_lesion_count: Number of non-target lesions
    - sum_target_diameters: Sum of longest diameters (baseline)
    - max_lesion_size: Largest lesion diameter
    - mean_lesion_size: Average lesion diameter
    - lesion_sites_count: Number of different anatomical sites
    - tumor_burden_category: Low/Medium/High based on sum of diameters
    
    NOTE: Uses screening visit ONLY to prevent temporal leakage.
    Post-screening tumor measurements directly define RECIST response.
    """
    if adls is None or adls.empty:
        return {}
    
    subj_lesions = adls[adls[ID_COL] == subjid].copy()
    if subj_lesions.empty:
        return {}
    
    features = {
        "target_lesion_count": 0,
        "nontarget_lesion_count": 0,
        "sum_target_diameters": None,
        "max_lesion_size": None,
        "mean_lesion_size": None,
        "lesion_sites_count": 0,
        "tumor_burden_category": None,
    }
    
    # CRITICAL FIX: Get SCREENING measurements only (prevent temporal leakage)
    # Tumor measurements at Week 8+ directly define RECIST response
    if "VISIT" in subj_lesions.columns:
        # Use explicit "Screening" visit
        baseline_lesions = subj_lesions[subj_lesions["VISIT"] == "Screening"].copy()
        # Override to ensure we only use screening data for ALL subsequent calculations
        subj_lesions = baseline_lesions
    elif "VISITDY" in subj_lesions.columns:
        # Fallback: use minimum VISITDY if VISIT column not available
        subj_lesions["VISITDY"] = pd.to_numeric(subj_lesions["VISITDY"], errors="coerce")
        screening_day = subj_lesions["VISITDY"].min()
        baseline_lesions = subj_lesions[subj_lesions["VISITDY"] == screening_day].copy()
        subj_lesions = baseline_lesions
    else:
        baseline_lesions = subj_lesions
    
    # Count lesion types (using screening data only)
    if "LSCAT" in subj_lesions.columns:
        target_count = (subj_lesions["LSCAT"] == "Target lesion").sum()
        nontarget_count = (subj_lesions["LSCAT"] == "Non-target lesion").sum()
        features["target_lesion_count"] = int(target_count)
        features["nontarget_lesion_count"] = int(nontarget_count)
    
    # Extract diameter information (LSLD = longest diameter of individual lesion)
    if "LSLD" in baseline_lesions.columns:
        diameters = pd.to_numeric(baseline_lesions["LSLD"], errors="coerce").dropna()
        
        if not diameters.empty:
            # Target lesions only for sum
            target_diameters = pd.to_numeric(
                baseline_lesions[baseline_lesions.get("LSCAT") == "Target lesion"]["LSLD"],
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
    
    # Count lesion sites (using screening data only)
    if "LSLOC" in subj_lesions.columns:
        site_count = subj_lesions["LSLOC"].nunique()
        features["lesion_sites_count"] = int(site_count)
    
    # REMOVED: new_lesions_flag - causes SEVERE temporal leakage
    # New lesions appearing at Week 8 = PD by RECIST definition
    # Model would achieve 100% accuracy by cheating
    
    return features


def extract_molecular_biomarkers(biomark: pd.DataFrame, subjid: str) -> Dict[str, Any]:
    """Extract enumerated biomarker results for KRAS/NRAS/BRAF assays."""

    if biomark is None or biomark.empty:
        return {}

    subj_biomark = biomark[biomark[ID_COL] == subjid]
    if subj_biomark.empty:
        return {}

    canonical_map = {
        "kras exon 2 (c12/13)": "KRAS_exon2",
        "kras exon 3 (c61)": "KRAS_exon3",
        "kras exon 4 (c117/146)": "KRAS_exon4",
        "nras exon 2 (c12/13)": "NRAS_exon2",
        "nras exon 3 (c61)": "NRAS_exon3",
        "nras exon 4 (c117/146)": "NRAS_exon4",
        "braf exon 15 (c600)": "BRAF_exon15",
    }

    def _normalise_result(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"wt", "wild", "wildtype"} or "wild" in lowered:
            return "WT"
        if "mut" in lowered:
            return "MUT"
        if "fail" in lowered or "invalid" in lowered:
            return None
        return "UNKNOWN"

    features = {feature: None for feature in canonical_map.values()}

    row = subj_biomark.iloc[0]
    for idx in range(1, 8):
        name = row.get(f"BMMTNM{idx}")
        result = row.get(f"BMMTR{idx}")
        if pd.isna(name):
            continue
        feature_key = canonical_map.get(str(name).strip().lower())
        if not feature_key:
            continue
        normalised = _normalise_result(result)
        if normalised is not None or features[feature_key] is None:
            features[feature_key] = normalised

    ras_markers = [
        "KRAS_exon2",
        "KRAS_exon3",
        "KRAS_exon4",
        "NRAS_exon2",
        "NRAS_exon3",
        "NRAS_exon4",
    ]

    ras_values = [features.get(marker) for marker in ras_markers]
    has_mut = any(val == "MUT" for val in ras_values)
    has_wt = any(val == "WT" for val in ras_values)

    if has_mut:
        features["RAS_status"] = "MUTANT"
    elif has_wt:
        features["RAS_status"] = "WILD-TYPE"
    else:
        features["RAS_status"] = "UNKNOWN"

    return features


def extract_physical_measurements(
    adpm: pd.DataFrame,
    subjid: str,
    baseline_weight: Optional[float] = None
) -> Dict[str, Any]:
    """
    Extract physical measurement trends from ADPM.
    
    Features (1 total):
    - weight_change_pct_42d: Percent change from baseline (B_WEIGHT) to
      the latest measurement within days 0-42.
    """
    features = {"weight_change_pct_42d": None}

    if adpm is None or adpm.empty:
        return features

    baseline_series = pd.to_numeric(pd.Series([baseline_weight]), errors="coerce")
    baseline_value = baseline_series.iloc[0]
    if pd.isna(baseline_value) or baseline_value == 0:
        return features

    subj_pm = adpm[adpm[ID_COL] == subjid].copy()
    if subj_pm.empty:
        return features
    
    # Extract weight measurements directly from WEIGHT column
    if "WEIGHT" in subj_pm.columns:
        subj_pm["VISITDY"] = pd.to_numeric(subj_pm.get("VISITDY"), errors="coerce")
        subj_pm["WEIGHT_VAL"] = pd.to_numeric(subj_pm["WEIGHT"], errors="coerce")
        weight_data = subj_pm.dropna(subset=["VISITDY", "WEIGHT_VAL"]).sort_values("VISITDY")

        if weight_data.empty:
            return features

        window_data = weight_data[(weight_data["VISITDY"] >= 0) & (weight_data["VISITDY"] <= 42)]
        if window_data.empty:
            return features

        latest_weight = window_data.iloc[-1]["WEIGHT_VAL"]
        if pd.isna(latest_weight):
            return features

        change_pct = (latest_weight - baseline_value) / baseline_value * 100
        features["weight_change_pct_42d"] = float(change_pct)
    
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
            # PDS310 response table uses RSRESP column (not AVALC)
            # Map response values
            response_mapping = {
                "Complete response": "CR",
                "Partial response": "PR",
                "Stable disease": "SD",
                "Progressive disease": "PD",
                "Unable to evaluate": "NE",
                "Unknown": "UNK",
            }
            
            if "RSRESP" in subj_rsp.columns:
                subj_rsp["response_coded"] = subj_rsp["RSRESP"].astype(str).map(response_mapping)
                subj_rsp["response_coded"] = subj_rsp["response_coded"].fillna("UNK")
                
                # Get best response (prioritize CR > PR > SD > PD)
                response_order = {"CR": 1, "PR": 2, "SD": 3, "PD": 4, "NE": 5, "UNK": 6}
                subj_rsp["response_rank"] = subj_rsp["response_coded"].map(response_order)
                
                # Filter out Unknown/NE for best response
                valid_responses = subj_rsp[subj_rsp["response_coded"].isin(["CR", "PR", "SD", "PD"])]
                if not valid_responses.empty:
                    best = valid_responses.sort_values("response_rank").iloc[0]
                    outcomes["best_response"] = best["response_coded"]
                else:
                    outcomes["best_response"] = None
            
            # Response at specific timepoints
            if "VISITDY" in subj_rsp.columns and "response_coded" in subj_rsp.columns:
                subj_rsp["VISITDY"] = pd.to_numeric(subj_rsp["VISITDY"], errors="coerce")
                subj_rsp = subj_rsp.dropna(subset=["VISITDY"])
                
                if not subj_rsp.empty:
                    # Week 8 (around day 56)
                    week8 = subj_rsp[subj_rsp["VISITDY"].between(49, 63)]
                    if not week8.empty:
                        week8_valid = week8[week8["response_coded"].isin(["CR", "PR", "SD", "PD"])]
                        if not week8_valid.empty:
                            outcomes["response_at_week8"] = week8_valid.iloc[0]["response_coded"]
                        else:
                            outcomes["response_at_week8"] = None
                    else:
                        outcomes["response_at_week8"] = None
                    
                    # Week 16 (around day 112)
                    week16 = subj_rsp[subj_rsp["VISITDY"].between(105, 119)]
                    if not week16.empty:
                        week16_valid = week16[week16["response_coded"].isin(["CR", "PR", "SD", "PD"])]
                        if not week16_valid.empty:
                            outcomes["response_at_week16"] = week16_valid.iloc[0]["response_coded"]
                        else:
                            outcomes["response_at_week16"] = None
                    else:
                        outcomes["response_at_week16"] = None
                    
                    # Time to response (first CR or PR)
                    responders = subj_rsp[subj_rsp["response_coded"].isin(["CR", "PR"])]
                    if not responders.empty:
                        first_response = responders.sort_values("VISITDY").iloc[0]
                        outcomes["time_to_response"] = float(first_response["VISITDY"])
                    else:
                        outcomes["time_to_response"] = None
                else:
                    outcomes["response_at_week8"] = None
                    outcomes["response_at_week16"] = None
                    outcomes["time_to_response"] = None
            else:
                outcomes["response_at_week8"] = None
                outcomes["response_at_week16"] = None
                outcomes["time_to_response"] = None
        else:
            outcomes["best_response"] = None
            outcomes["response_at_week8"] = None
            outcomes["response_at_week16"] = None
            outcomes["time_to_response"] = None
    else:
        outcomes["best_response"] = None
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
    
    Integrates 71 features across 12 categories:
    1. Identifiers (2 features)
    2. Demographics (5 features)
    3. Disease Characteristics (4 features)
    4. Treatment Assignment (1 feature)
    5. Baseline Labs (8 features)
    6. Longitudinal Labs (16 features) - early window only (days 1-42)
    7. Tumor Characteristics (7 features)
    8. Molecular Biomarkers (8 features)
    9. Physical Measurements (1 feature)
    10. Prior History (5 features)
    11. Derived Risk Scores (6 features)
    12. Outcomes (8 features) - optional
    
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
        profile.update(
            extract_physical_measurements(
                tables["adpm"],
                subjid,
                baseline_weight=profile.get("B_WEIGHT")
            )
        )
    
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
        "treatment": ["ATRT"],
        "baseline_labs": [
            "baseline_ALB", "baseline_ALP", "baseline_CEA", "baseline_CREAT",
            "baseline_HGB", "baseline_LDH", "baseline_PLT", "baseline_WBC"
        ],
        "longitudinal_labs": [
            *[
                f"lab_{lab}_early_last"
                for lab in canonical_lab_list()
            ],
            *[
                f"lab_{lab}_early_slope"
                for lab in canonical_lab_list()
            ],
        ],
        "tumor": [
            "target_lesion_count", "nontarget_lesion_count", "sum_target_diameters",
            "max_lesion_size", "mean_lesion_size", "lesion_sites_count",
            "tumor_burden_category"
        ],
        "molecular": [
            "KRAS_exon2", "KRAS_exon3", "KRAS_exon4",
            "NRAS_exon2", "NRAS_exon3", "NRAS_exon4",
            "BRAF_exon15", "RAS_status"
        ],
        "physical": [
            "weight_change_pct_42d"
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
