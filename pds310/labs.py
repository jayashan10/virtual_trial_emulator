"""
Shared laboratory name utilities for PDS310.

Normalises the variety of LBTEST / PARAMCD values in the ADLB table
to a small set of canonical short codes used across feature builders.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional


# Canonical short codes we care about for baseline/longitudinal labs.
CANONICAL_LABS = [
    "ALB",   # Albumin
    "ALP",   # Alkaline phosphatase
    "CEA",   # Carcinoembryonic antigen
    "CREAT", # Creatinine
    "HGB",   # Haemoglobin
    "LDH",   # Lactate dehydrogenase
    "PLT",   # Platelets
    "WBC",   # White blood cells
]


# Mapping of LBTEST / PARAM / synonyms to canonical codes.
_LAB_NAME_MAP: Dict[str, str] = {
    "ALB": "ALB",
    "ALBUMIN": "ALB",
    "SERUM ALBUMIN": "ALB",
    "ALP": "ALP",
    "ALKALINE PHOSPHATASE": "ALP",
    "ALK PHOSPHATASE": "ALP",
    "CEA": "CEA",
    "CARCINOEMBRYONIC ANTIGEN": "CEA",
    "CARCINOEMBRYONIC AG": "CEA",
    "CREAT": "CREAT",
    "CREATININE": "CREAT",
    "SERUM CREATININE": "CREAT",
    "HGB": "HGB",
    "HGB(HGB)": "HGB",
    "HEMOGLOBIN": "HGB",
    "HEMOGLOBIN (HGB)": "HGB",
    "HB": "HGB",
    "LDH": "LDH",
    "LACTATE DEHYDROGENASE": "LDH",
    "LACTIC DEHYDROGENASE": "LDH",
    "PLT": "PLT",
    "PLATELETS": "PLT",
    "PLATELET COUNT": "PLT",
    "WBC": "WBC",
    "WHITE BLOOD CELLS": "WBC",
    "WHITE BLOOD CELL COUNT": "WBC",
    # Historical prostate-specific antigen entries (retain for compatibility)
    "PSA": "PSA",
    "PROSTATE SPECIFIC ANTIGEN": "PSA",
}


def canonical_lab_name(raw_name: Optional[str]) -> Optional[str]:
    """Return the canonical short code for a lab test, or None if unknown."""
    if raw_name is None:
        return None
    key = raw_name.strip().upper()
    return _LAB_NAME_MAP.get(key)


def canonical_lab_names(raw_names: Iterable[str]) -> Dict[str, str]:
    """
    Convenience helper to map an iterable of LBTEST values to canonical codes.

    Returns a dictionary of {original_name: canonical_code} for recognised labs.
    """
    out: Dict[str, str] = {}
    for name in raw_names:
        canon = canonical_lab_name(name)
        if canon is not None:
            out[name] = canon
    return out


def is_canonical_lab(code: str) -> bool:
    """Check if a short code is one of our canonical lab identifiers."""
    return code in CANONICAL_LABS


def canonical_lab_list(preferred: Optional[Iterable[str]] = None) -> Iterable[str]:
    """
    Return an ordered iterable of canonical lab codes.

    If *preferred* is provided it is filtered/ordered according to CANONICAL_LABS.
    """
    if preferred is None:
        return CANONICAL_LABS
    preferred_upper = [p.upper() for p in preferred]
    return [c for c in CANONICAL_LABS if c in preferred_upper]

