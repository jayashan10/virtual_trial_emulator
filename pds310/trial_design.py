"""
Virtual Trial Design Specification for PDS310.

Defines trial structure, eligibility, endpoints, and analysis plans.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import json


@dataclass
class EligibilityCriteria:
    """Define patient eligibility criteria."""
    
    min_age: Optional[float] = None
    max_age: Optional[float] = None
    
    required_ecog: Optional[List[int]] = None  # e.g., [0, 1]
    
    ras_status: Optional[List[str]] = None  # e.g., ["WILD-TYPE"]
    
    min_weight: Optional[float] = None
    max_weight: Optional[float] = None
    
    required_sex: Optional[str] = None  # "M" or "F"
    
    exclude_prior_egfr: bool = False
    
    custom_filter: Optional[Callable] = None  # Custom function
    
    def apply(self, profile_db: pd.DataFrame) -> pd.DataFrame:
        """
        Apply eligibility criteria to filter patient profiles.
        
        Args:
            profile_db: Patient profile database
        
        Returns:
            Filtered DataFrame of eligible patients
        """
        eligible = profile_db.copy()
        
        # Age criteria (handle both 'age' and 'AGE')
        age_col = 'AGE' if 'AGE' in eligible.columns else 'age'
        if self.min_age is not None and age_col in eligible.columns:
            eligible = eligible[eligible[age_col] >= self.min_age]
        if self.max_age is not None and age_col in eligible.columns:
            eligible = eligible[eligible[age_col] <= self.max_age]
        
        # ECOG performance status (handle both 'baseline_ecog' and 'B_ECOG')
        ecog_col = 'B_ECOG' if 'B_ECOG' in eligible.columns else 'baseline_ecog'
        if self.required_ecog is not None and ecog_col in eligible.columns:
            # Map text ECOG to numeric if needed
            if eligible[ecog_col].dtype == 'object':
                ecog_map = {
                    'Fully active': 0,
                    'Symptoms but ambulatory': 1,
                    'In bed less than 50% of the time': 2,
                    'In bed more than 50% of the time': 3,
                    'Bedridden': 4,
                }
                eligible_ecog = eligible[ecog_col].map(ecog_map)
                eligible = eligible[eligible_ecog.isin(self.required_ecog)]
            else:
                eligible = eligible[eligible[ecog_col].isin(self.required_ecog)]
        
        # RAS status
        if self.ras_status is not None and 'RAS_status' in eligible.columns:
            eligible = eligible[eligible['RAS_status'].isin(self.ras_status)]
        
        # Weight (handle both 'weight' and 'B_WEIGHT')
        weight_col = 'B_WEIGHT' if 'B_WEIGHT' in eligible.columns else 'weight'
        if self.min_weight is not None and weight_col in eligible.columns:
            eligible = eligible[eligible[weight_col] >= self.min_weight]
        if self.max_weight is not None and weight_col in eligible.columns:
            eligible = eligible[eligible[weight_col] <= self.max_weight]
        
        # Sex (handle both 'sex' and 'SEX')
        sex_col = 'SEX' if 'SEX' in eligible.columns else 'sex'
        if self.required_sex is not None and sex_col in eligible.columns:
            eligible = eligible[eligible[sex_col] == self.required_sex]
        
        # Prior EGFR therapy
        if self.exclude_prior_egfr:
            if 'prior_egfr_therapy' in eligible.columns:
                eligible = eligible[eligible['prior_egfr_therapy'] == 0]
        
        # Custom filter function
        if self.custom_filter is not None:
            eligible = self.custom_filter(eligible)
        
        return eligible
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'min_age': self.min_age,
            'max_age': self.max_age,
            'required_ecog': self.required_ecog,
            'ras_status': self.ras_status,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'required_sex': self.required_sex,
            'exclude_prior_egfr': self.exclude_prior_egfr,
        }


@dataclass
class TreatmentArm:
    """Define a treatment arm."""
    
    name: str
    description: str
    
    # Treatment effect modifiers (relative to control)
    response_rate_modifier: float = 1.0  # Multiplier for response rate
    ttr_modifier: float = 1.0  # Multiplier for time-to-response
    os_hr: float = 1.0  # Hazard ratio for overall survival
    pfs_hr: float = 1.0  # Hazard ratio for progression-free survival
    
    # Treatment-specific features
    is_control: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'response_rate_modifier': self.response_rate_modifier,
            'ttr_modifier': self.ttr_modifier,
            'os_hr': self.os_hr,
            'pfs_hr': self.pfs_hr,
            'is_control': self.is_control,
        }


@dataclass
class TrialEndpoint:
    """Define a trial endpoint."""
    
    name: str
    endpoint_type: str  # "primary" or "secondary"
    
    # Outcome metric
    metric: str  # e.g., "os", "pfs", "response_rate", "ttr"
    
    # Statistical test
    test_type: str = "survival"  # "survival", "proportion", "continuous"
    
    # Analysis parameters
    alpha: float = 0.05
    power: float = 0.80
    
    # Success criteria
    target_hr: Optional[float] = None  # For survival endpoints
    target_difference: Optional[float] = None  # For proportions/continuous
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'endpoint_type': self.endpoint_type,
            'metric': self.metric,
            'test_type': self.test_type,
            'alpha': self.alpha,
            'power': self.power,
            'target_hr': self.target_hr,
            'target_difference': self.target_difference,
        }


@dataclass
class TrialDesign:
    """Complete virtual trial design specification."""
    
    # Trial metadata
    trial_name: str
    description: str
    phase: str = "3"  # "2", "3", "4"
    
    # Design parameters
    n_patients: int = 300
    allocation_ratio: List[float] = field(default_factory=lambda: [1.0, 1.0])  # e.g., [2, 1] for 2:1
    
    # Arms
    treatment_arms: List[TreatmentArm] = field(default_factory=list)
    
    # Eligibility
    eligibility: EligibilityCriteria = field(default_factory=EligibilityCriteria)
    
    # Endpoints
    endpoints: List[TrialEndpoint] = field(default_factory=list)
    
    # Stratification factors
    stratification_factors: List[str] = field(default_factory=list)  # e.g., ["RAS_status", "baseline_ecog"]
    
    # Randomization
    random_seed: Optional[int] = None
    
    # Follow-up
    max_followup_days: int = 730  # 2 years
    
    def __post_init__(self):
        """Validate design after initialization."""
        if len(self.treatment_arms) == 0:
            raise ValueError("At least one treatment arm must be specified")
        
        if len(self.allocation_ratio) != len(self.treatment_arms):
            raise ValueError(f"Allocation ratio length ({len(self.allocation_ratio)}) "
                           f"must match number of arms ({len(self.treatment_arms)})")
        
        if len(self.endpoints) == 0:
            raise ValueError("At least one endpoint must be specified")
    
    def get_arm_sizes(self) -> Dict[str, int]:
        """
        Calculate number of patients per arm based on allocation ratio.
        
        Returns:
            Dictionary mapping arm name to patient count
        """
        # Normalize allocation ratio
        total_ratio = sum(self.allocation_ratio)
        proportions = [r / total_ratio for r in self.allocation_ratio]
        
        # Calculate sizes (ensuring sum equals n_patients)
        sizes = [int(self.n_patients * p) for p in proportions]
        
        # Adjust for rounding
        while sum(sizes) < self.n_patients:
            sizes[0] += 1
        while sum(sizes) > self.n_patients:
            sizes[0] -= 1
        
        return {arm.name: size for arm, size in zip(self.treatment_arms, sizes)}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'trial_name': self.trial_name,
            'description': self.description,
            'phase': self.phase,
            'n_patients': self.n_patients,
            'allocation_ratio': self.allocation_ratio,
            'treatment_arms': [arm.to_dict() for arm in self.treatment_arms],
            'eligibility': self.eligibility.to_dict(),
            'endpoints': [ep.to_dict() for ep in self.endpoints],
            'stratification_factors': self.stratification_factors,
            'random_seed': self.random_seed,
            'max_followup_days': self.max_followup_days,
            'arm_sizes': self.get_arm_sizes(),
        }
    
    def save(self, filepath: str):
        """Save trial design to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"Trial design saved to: {filepath}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrialDesign':
        """Load trial design from dictionary."""
        # Reconstruct treatment arms
        arms = [TreatmentArm(
            name=arm['name'],
            description=arm['description'],
            response_rate_modifier=arm.get('response_rate_modifier', 1.0),
            ttr_modifier=arm.get('ttr_modifier', 1.0),
            os_hr=arm.get('os_hr', 1.0),
            pfs_hr=arm.get('pfs_hr', 1.0),
            is_control=arm.get('is_control', False),
        ) for arm in data.get('treatment_arms', [])]
        
        # Reconstruct eligibility
        elig_data = data.get('eligibility', {})
        eligibility = EligibilityCriteria(
            min_age=elig_data.get('min_age'),
            max_age=elig_data.get('max_age'),
            required_ecog=elig_data.get('required_ecog'),
            ras_status=elig_data.get('ras_status'),
            min_weight=elig_data.get('min_weight'),
            max_weight=elig_data.get('max_weight'),
            required_sex=elig_data.get('required_sex'),
            exclude_prior_egfr=elig_data.get('exclude_prior_egfr', False),
        )
        
        # Reconstruct endpoints
        endpoints = [TrialEndpoint(
            name=ep['name'],
            endpoint_type=ep['endpoint_type'],
            metric=ep['metric'],
            test_type=ep.get('test_type', 'survival'),
            alpha=ep.get('alpha', 0.05),
            power=ep.get('power', 0.80),
            target_hr=ep.get('target_hr'),
            target_difference=ep.get('target_difference'),
        ) for ep in data.get('endpoints', [])]
        
        return cls(
            trial_name=data['trial_name'],
            description=data['description'],
            phase=data.get('phase', '3'),
            n_patients=data['n_patients'],
            allocation_ratio=data.get('allocation_ratio', [1.0, 1.0]),
            treatment_arms=arms,
            eligibility=eligibility,
            endpoints=endpoints,
            stratification_factors=data.get('stratification_factors', []),
            random_seed=data.get('random_seed'),
            max_followup_days=data.get('max_followup_days', 730),
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'TrialDesign':
        """Load trial design from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Predefined trial designs
def create_pds310_original_design() -> TrialDesign:
    """
    Recreate the original PDS310 trial design.
    
    Panitumumab + BSC vs BSC alone in RAS wild-type mCRC.
    """
    return TrialDesign(
        trial_name="PDS310-Original",
        description="Panitumumab + BSC vs BSC alone in RAS wild-type metastatic colorectal cancer",
        phase="3",
        n_patients=377,
        allocation_ratio=[1.0, 1.0],  # 1:1 randomization
        treatment_arms=[
            TreatmentArm(
                name="Panitumumab + BSC",
                description="Panitumumab 6 mg/kg IV Q2W + best supportive care",
                response_rate_modifier=1.5,  # Assume 50% improvement
                ttr_modifier=1.0,
                os_hr=0.88,  # Observed from real trial
                pfs_hr=0.54,
                is_control=False,
            ),
            TreatmentArm(
                name="BSC",
                description="Best supportive care alone",
                response_rate_modifier=1.0,
                ttr_modifier=1.0,
                os_hr=1.0,
                pfs_hr=1.0,
                is_control=True,
            ),
        ],
        eligibility=EligibilityCriteria(
            min_age=18,
            ras_status=["WILD-TYPE"],
            required_ecog=[0, 1, 2],
        ),
        endpoints=[
            TrialEndpoint(
                name="Overall Survival",
                endpoint_type="primary",
                metric="os",
                test_type="survival",
                alpha=0.05,
                power=0.80,
                target_hr=0.88,
            ),
            TrialEndpoint(
                name="Progression-Free Survival",
                endpoint_type="secondary",
                metric="pfs",
                test_type="survival",
                target_hr=0.54,
            ),
            TrialEndpoint(
                name="Overall Response Rate",
                endpoint_type="secondary",
                metric="response_rate",
                test_type="proportion",
            ),
        ],
        stratification_factors=["baseline_ecog", "num_metastatic_sites"],
        random_seed=42,
        max_followup_days=730,
    )


def create_enriched_design() -> TrialDesign:
    """
    Create an enriched trial design for high-risk patients.
    """
    return TrialDesign(
        trial_name="PDS310-Enriched",
        description="Enriched trial in high-risk RAS wild-type mCRC patients",
        phase="2",
        n_patients=150,
        allocation_ratio=[2.0, 1.0],  # 2:1 randomization
        treatment_arms=[
            TreatmentArm(
                name="Experimental",
                description="Enhanced panitumumab regimen",
                response_rate_modifier=2.0,
                os_hr=0.70,
                pfs_hr=0.50,
            ),
            TreatmentArm(
                name="Control",
                description="Standard panitumumab",
                response_rate_modifier=1.0,
                os_hr=1.0,
                pfs_hr=1.0,
                is_control=True,
            ),
        ],
        eligibility=EligibilityCriteria(
            min_age=18,
            max_age=75,
            ras_status=["WILD-TYPE"],
            required_ecog=[0, 1],
        ),
        endpoints=[
            TrialEndpoint(
                name="Progression-Free Survival",
                endpoint_type="primary",
                metric="pfs",
                test_type="survival",
                target_hr=0.60,
            ),
        ],
        stratification_factors=["baseline_ecog"],
        random_seed=42,
    )
