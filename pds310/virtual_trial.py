"""
Virtual Clinical Trial Simulation Engine for PDS310.

Runs complete virtual trials using digital twins and predictive models.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from .trial_design import TrialDesign, TreatmentArm
from .twin_generator import generate_twin_cohort
from .predict_outcomes import OutcomePredictor


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    return value


class VirtualTrial:
    """
    Virtual clinical trial simulator.
    
    Combines digital twin generation, treatment assignment,
    outcome prediction, and statistical analysis.
    """
    
    def __init__(
        self,
        design: TrialDesign,
        source_profiles: pd.DataFrame,
        outcome_predictor: OutcomePredictor,
        effect_source: str = "learned",
        verbose: bool = True
    ):
        """
        Initialize virtual trial.
        
        Args:
            design: Trial design specification
            source_profiles: Real patient profiles for twin generation
            outcome_predictor: Trained outcome prediction models
            verbose: Print progress messages
        """
        self.design = design
        self.source_profiles = source_profiles
        self.predictor = outcome_predictor
        self.effect_source = effect_source.lower()
        if self.effect_source not in {"learned", "assumed"}:
            raise ValueError(f"effect_source must be 'learned' or 'assumed', got '{effect_source}'")
        self.verbose = verbose
        
        # Trial results (populated after run)
        self.enrolled_patients: Optional[pd.DataFrame] = None
        self.trial_results: Optional[Dict[str, Any]] = None
    
    def enroll_patients(self) -> pd.DataFrame:
        """
        Generate and enroll virtual patients.
        
        Returns:
            DataFrame of enrolled patients with treatment assignments
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"ENROLLING PATIENTS: {self.design.trial_name}")
            print(f"{'='*80}")
        
        # Apply eligibility criteria to source profiles
        eligible_source = self.design.eligibility.apply(self.source_profiles)
        
        if self.verbose:
            print(f"Eligible source patients: {len(eligible_source)} / {len(self.source_profiles)}")
        
        if len(eligible_source) == 0:
            raise ValueError("No eligible patients found in source profiles")
        
        # Generate digital twins for trial
        if self.verbose:
            print(f"Generating {self.design.n_patients} digital twins...")
        
        twins = generate_twin_cohort(
            profile_db=eligible_source,
            n_twins=self.design.n_patients,
            strategy="random",  # Use random for faster generation
            validation_threshold=0.0,  # Accept all twins
            seed=self.design.random_seed,
            verbose=self.verbose
        )
        
        # Add patient IDs
        twins['patient_id'] = [f"VPT-{i:04d}" for i in range(len(twins))]
        
        if self.verbose:
            print(f"✅ Generated {len(twins)} virtual patients")
        
        return twins
    
    def randomize_treatment(
        self,
        patients: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Randomize patients to treatment arms with stratification.
        
        Args:
            patients: Enrolled patients
        
        Returns:
            DataFrame with treatment assignments
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RANDOMIZING TREATMENT ASSIGNMENTS")
            print(f"{'='*80}")
        
        patients = patients.copy()
        
        # Get arm sizes
        arm_sizes = self.design.get_arm_sizes()
        
        if self.verbose:
            print("Target allocation:")
            for arm_name, size in arm_sizes.items():
                print(f"  {arm_name}: {size} patients")
        
        # Stratified randomization if specified
        if self.design.stratification_factors:
            if self.verbose:
                print(f"Stratifying by: {', '.join(self.design.stratification_factors)}")
            
            # Create strata - handle column name variations
            strata_cols = []
            for factor in self.design.stratification_factors:
                # Try both original and uppercase versions
                if factor in patients.columns:
                    strata_cols.append(factor)
                elif factor.upper() in patients.columns:
                    strata_cols.append(factor.upper())
                elif factor.replace('baseline_', 'B_') in patients.columns:
                    strata_cols.append(factor.replace('baseline_', 'B_'))
            
            if strata_cols:
                patients['_stratum'] = patients[strata_cols].astype(str).agg('-'.join, axis=1)
            else:
                patients['_stratum'] = 'all'
        else:
            patients['_stratum'] = 'all'
        
        # Randomize within each stratum
        np.random.seed(self.design.random_seed)
        
        treatment_assignments = []
        
        for stratum in patients['_stratum'].unique():
            stratum_patients = patients[patients['_stratum'] == stratum].copy()
            n_stratum = len(stratum_patients)
            
            # Calculate arm assignments for this stratum
            total_ratio = sum(self.design.allocation_ratio)
            proportions = [r / total_ratio for r in self.design.allocation_ratio]
            
            # Generate assignments
            assignments = []
            for i, arm in enumerate(self.design.treatment_arms):
                n_arm = int(n_stratum * proportions[i])
                assignments.extend([arm.name] * n_arm)
            
            # Fill any remaining spots
            while len(assignments) < n_stratum:
                assignments.append(self.design.treatment_arms[0].name)
            
            # Shuffle
            np.random.shuffle(assignments)
            
            treatment_assignments.extend(assignments[:n_stratum])
        
        patients['treatment_arm'] = treatment_assignments
        patients = patients.drop(columns=['_stratum'])
        
        # Summary
        if self.verbose:
            print("\nActual allocation:")
            for arm_name in arm_sizes.keys():
                n = (patients['treatment_arm'] == arm_name).sum()
                print(f"  {arm_name}: {n} patients ({n/len(patients)*100:.1f}%)")
        
        return patients
    
    def simulate_outcomes(
        self,
        patients: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Simulate outcomes for all patients using predictive models.
        
        Args:
            patients: Patients with treatment assignments
        
        Returns:
            DataFrame with predicted outcomes
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"SIMULATING PATIENT OUTCOMES")
            print(f"{'='*80}")
        
        if self.effect_source == "assumed":
            return self._simulate_outcomes_assumed(patients.copy())
        return self._simulate_outcomes_learned(patients.copy())

    def _simulate_outcomes_assumed(self, patients: pd.DataFrame) -> pd.DataFrame:
        """Legacy simulation that applies design-specified modifiers."""
        arm_map = {arm.name: arm for arm in self.design.treatment_arms}
        predictions = []

        for idx, patient in patients.iterrows():
            profile = patient.to_frame().T
            arm = arm_map[patient["treatment_arm"]]
            try:
                prediction = self.predictor.predict_all(profile, treatment_arm=arm.name, include_biomarkers=False)
                predictions.append(self._build_record(patient, prediction, counterfactuals=None))
            except Exception as exc:  # pragma: no cover - diagnostic
                if self.verbose:
                    print(f"Warning: Could not predict for patient {patient.get('patient_id', idx)}: {exc}")
                predictions.append({"patient_id": patient.get("patient_id", idx)})

        combined = pd.DataFrame(predictions)
        if self.verbose:
            print(f"✅ Simulated outcomes for {len(combined)} patients (assumed effects)")
        return combined

    def _simulate_outcomes_learned(self, patients: pd.DataFrame) -> pd.DataFrame:
        """Simulation using learned treatment effects (counterfactual predictions)."""
        arm_names = [arm.name for arm in self.design.treatment_arms]
        rng_base = self.design.random_seed or 42
        records: List[Dict[str, Any]] = []

        for idx, patient in patients.iterrows():
            profile = patient.to_frame().T
            seed = rng_base + idx * 7919
            counterfactuals = self.predictor.predict_counterfactuals(
                profile,
                arm_names=arm_names,
                include_biomarkers=False,
                max_followup=self.design.max_followup_days,
                seed=seed,
            )

            assigned_arm = patient["treatment_arm"]
            realized = counterfactuals.get(assigned_arm, {})
            record = self._build_record(patient, realized, counterfactuals=counterfactuals)
            records.append(record)

        results = pd.DataFrame(records)
        if self.verbose:
            print(f"✅ Simulated outcomes for {len(results)} patients (learned effects)")
        return results

    def _build_record(
        self,
        patient: pd.Series,
        prediction: Dict[str, Any],
        counterfactuals: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Flatten prediction dictionary into a single row record."""
        record = patient.to_dict()
        record["patient_id"] = prediction.get("patient_id", record.get("patient_id"))
        record["treatment_arm"] = prediction.get("treatment_arm", record.get("treatment_arm"))

        resp = prediction.get("response", {}) or {}
        record["predicted_response"] = resp.get("predicted_response")
        record["response_confidence"] = resp.get("confidence")
        for cls, prob in resp.get("probabilities", {}).items():
            record[f"prob_{cls}"] = prob

        ttr = prediction.get("time_to_response", {}) or {}
        record["predicted_ttr"] = ttr.get("predicted_ttr")
        record["ttr_lower_95ci"] = ttr.get("lower_95ci")
        record["ttr_upper_95ci"] = ttr.get("upper_95ci")

        os_pred = prediction.get("overall_survival", {}) or {}
        record["os_time"] = os_pred.get("time")
        record["os_event"] = os_pred.get("event")
        record["os_partial_hazard"] = os_pred.get("partial_hazard")

        if counterfactuals is not None:
            record["counterfactuals"] = json.dumps(_json_safe(counterfactuals))
        else:
            record["counterfactuals"] = None

        return record
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete virtual trial.
        
        Returns:
            Dictionary with trial results
        """
        start_time = datetime.now()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"VIRTUAL TRIAL SIMULATION")
            print(f"Trial: {self.design.trial_name}")
            print(f"Design: {self.design.description}")
            print(f"{'='*80}")
        
        # Step 1: Enroll patients
        enrolled = self.enroll_patients()
        
        # Step 2: Randomize treatment
        randomized = self.randomize_treatment(enrolled)
        
        # Step 3: Simulate outcomes
        with_outcomes = self.simulate_outcomes(randomized)
        
        # Store results
        self.enrolled_patients = with_outcomes
        
        # Compile trial results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.trial_results = {
            'trial_name': self.design.trial_name,
            'n_enrolled': len(with_outcomes),
            'arms': list(self.design.get_arm_sizes().keys()),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'design': self.design.to_dict(),
            'effect_source': self.effect_source,
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"TRIAL SIMULATION COMPLETE")
            print(f"Duration: {duration:.1f} seconds")
            print(f"{'='*80}")
        
        return self.trial_results
    
    def get_arm_data(self, arm_name: str) -> pd.DataFrame:
        """Get data for specific treatment arm."""
        if self.enrolled_patients is None:
            raise ValueError("Trial has not been run yet")
        
        return self.enrolled_patients[
            self.enrolled_patients['treatment_arm'] == arm_name
        ].copy()
    
    def save_results(self, output_dir: str):
        """
        Save trial results to files.
        
        Args:
            output_dir: Directory to save results
        """
        if self.enrolled_patients is None:
            raise ValueError("Trial has not been run yet")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save patient-level data
        patient_file = output_path / f"{self.design.trial_name}_patients.csv"
        self.enrolled_patients.to_csv(patient_file, index=False)
        
        # Save trial results
        results_file = output_path / f"{self.design.trial_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.trial_results, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ Results saved to: {output_dir}")
            print(f"  - Patient data: {patient_file.name}")
            print(f"  - Trial results: {results_file.name}")
    
    def summary(self) -> pd.DataFrame:
        """
        Create summary table of trial results by arm.
        
        Returns:
            DataFrame with summary statistics
        """
        if self.enrolled_patients is None:
            raise ValueError("Trial has not been run yet")
        
        summary_rows = []
        
        for arm in self.design.treatment_arms:
            arm_data = self.get_arm_data(arm.name)
            
            row = {
                'Treatment Arm': arm.name,
                'N': len(arm_data),
            }
            
            # Add outcome summaries
            if 'predicted_response' in arm_data.columns:
                response_counts = arm_data['predicted_response'].value_counts()
                row['CR/PR (%)'] = (response_counts.get('CR', 0) + 
                                   response_counts.get('PR', 0)) / len(arm_data) * 100
            
            if 'predicted_ttr' in arm_data.columns:
                ttr_vals = arm_data['predicted_ttr'].dropna()
                row['TTR (days)'] = float(ttr_vals.median()) if len(ttr_vals) else None

            if 'os_time' in arm_data.columns:
                os_vals = arm_data['os_time'].dropna()
                row['OS median (days)'] = float(os_vals.median()) if len(os_vals) else None
            
            # Add baseline characteristics
            age_col = 'AGE' if 'AGE' in arm_data.columns else 'age'
            if age_col in arm_data.columns:
                row['Age (median)'] = arm_data[age_col].median()
            
            ecog_col = 'B_ECOG' if 'B_ECOG' in arm_data.columns else 'baseline_ecog'
            if ecog_col in arm_data.columns:
                # Handle text ECOG values
                if arm_data[ecog_col].dtype == 'object':
                    ecog_01_values = ['Fully active', 'Symptoms but ambulatory']
                    row['ECOG 0-1 (%)'] = arm_data[ecog_col].isin(ecog_01_values).sum() / len(arm_data) * 100
                else:
                    row['ECOG 0-1 (%)'] = (arm_data[ecog_col] <= 1).sum() / len(arm_data) * 100
            
            summary_rows.append(row)
        
        return pd.DataFrame(summary_rows)
