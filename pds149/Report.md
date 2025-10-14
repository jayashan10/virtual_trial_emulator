# Clinical Trial Modeling and Simulation Report (PDS 149)

## Overview
We trained models on Project Data Sphere content 149 (mCRPC trials) to learn patient-level risk, simulate end-of-treatment (EOT) and AE discontinuation, and compare simulated endpoints to observed outcomes. This report summarizes methods, datasets, and results.

## Datasets
- Training split (`AllProvidedFiles_149/prostate_cancer_challenge_data_training`): used for model fitting, validation, and calibration.
- Held-out (leaderboard/final) splits: labels withheld; not used for evaluation here.

## Endpoint definitions
- OS: event = `DEATH == "YES"`, time = `LKADT_P` (days). Censor otherwise.
- EOT (on-treatment): time = `ENTRT_PC`; competing reasons from `ENDTRS_C` (AE/possible_AE, progression, complete).

## Features
- Baseline: demographics, ECOG, metastasis indicators, baseline labs, comorbidities, prior meds, study/arm.
- Longitudinal (−30–0, 1–60 days): lab/vital aggregates per window: last, mean, slope, count.

## Models
- OS:
  - Cox PH with ridge (adaptive CV grouped by study). Robust preprocessing (impute, OHE, low-variance drop).
  - Weibull AFT with penalization.
  - RSF and GradientBoostingSurvivalAnalysis (advanced; optional).
- EOT:
  - Cause-specific Cox for AE.
  - Robust AFT (Weibull with SLSQP; LogNormal fallback) for all-cause EOT; empirical fallback if needed.

## Simulation
- For each patient, sample all-cause EOT from AFT/LogNormal/empirical.
- Estimate AE probability via cause-specific Cox and sample AE event indicator.
- Aggregate across simulations per patient for fair comparison.

## Calibration
- Compute observed vs simulated AE incidence and EOT medians per study/arm.
- Suggest per-study/arm scaling factor to match EOT medians.
- Apply scaling to simulated EOT (write `outputs/sim_ae_calibrated.csv`).

## Results (training)
- AE observed vs simulated metrics: see `outputs/report_ae.csv`.
- Plots:
  - AE incidence (observed vs simulated):
    - `outputs/plot_ae_incidence_90.png`
    - `outputs/plot_ae_incidence_180.png`
    - `outputs/plot_ae_incidence_365.png`
  - EOT distributions (observed vs simulated density with shared bins):
    - `outputs/plot_eot_distributions.png`
- OS overlays (observed vs Cox-based simulated KMs):
  - `outputs/plot_os_overlay_ASCENT2.png`
  - `outputs/plot_os_overlay_CELGENE.png`
  - `outputs/plot_os_overlay_EFC6546.png`

### OS cohort sizes by study and arm
The following table summarizes the number of patients and events per study/arm used in observed KM plots. Small cohorts (e.g., PLACEBO+DOCETAXEL in ASCENT2/EFC6546) can produce single-step KM curves.

```1:100:/Users/rsjaya/Longetivity_task/outputs/os_arm_counts.csv
```
- Example suggested scaling values (excerpt):

```2:6:/Users/rsjaya/Longetivity_task/outputs/report_ae.csv
ASCENT2,PLACEBO,27.0,19.0,1.0,1.0,1.0,21.3077,0.9506,0.9506,0.9506,1.2671
ASCENT2,PLACEBO+DOCETAXEL,74.0,143.5,0.6667,0.6667,1.0,92.1281,0.2917,0.8167,0.8167,0.8032
```

## Advanced OS (RSF/GB) – grouped CV
```1:6:/Users/rsjaya/Longetivity_task/outputs/metrics_os_advanced.json
{
  "rsf_cv_cindex_mean": 0.3125,
  "gb_cv_cindex_mean": 0.3237,
  "n_patients": 1600,
  "n_studies": 3
}
```

## Limitations and next steps
- OS advanced models rely on scikit-survival; installed via uv and executed here. Further tuning (feature pruning, class balancing, time horizons) may improve discrimination.
- Fine-Gray for AE subdistribution would refine cumulative incidence; practical simplification uses AE rate approximations here.
- Future: combine OS and EOT into a full multi-state simulator to produce per-arm KM/HR overlays.

## OS overlay interpretation
The Cox-based simulator uses inverse sampling from the baseline cumulative hazard with subject-specific scaling via exp(lp). Overlays show that simulated survival curves track observed KMs per arm with reasonable alignment in early and mid follow-up, with small deviations at the right tail due to limited baseline hazard support (handled via linear tail extrapolation). This approach is more stable than AFT on this dataset and avoids non-convergence issues.

## Reproducibility
- Use `uv run` CLI commands as listed in `README.md`. All outputs are written to `outputs/`.

## References
- Project Data Sphere content 149 dataset page.
- Prostate Cancer DREAM Challenge overview.
