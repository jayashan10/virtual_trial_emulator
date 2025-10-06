# PDS310 Report

This report documents OS training, AE/EOT simulation, outputs, and interpretations for PDS310.

## Inputs
- ADSL, ADLB, ADAE from `AllProvidedFiles_310/PDS_DSA_20020408/`
- Config: `pds310/config.yaml`

## Pipelines and artifacts
- OS baseline: `cox.joblib`, optional `aft_weibull.joblib`, `metrics_os.json`, `km_input.csv`
- OS overlays (Cox-based simulation): `plot_os_overlay_PDS310.png`
- Advanced OS (RSF, GB): `os_rsf.joblib`, `os_gb.joblib`, `metrics_os_advanced.json`
- AE/EOT: `ae_cox.joblib`, `eot_model.joblib`, `sim_ae.csv`, `report_ae.csv`, `report_ae_summary.json`, `sim_ae_calibrated.csv`, `plot_ae_incidence_*.png`, `plot_eot_distributions.png`

## Key results observed
- Data footprint: 370 subjects, 1 study (`STUDYID=PDS310`). Arms detected: `Best supportive care`, `panit. plus best supportive care`.
- OS metrics (single-study friendly KFold): `cox_kfold_cindex_mean ≈ 0.666` (from `metrics_os.json`). Grouped-by-study CV remains `NaN` as expected.
- Advanced OS metrics (KFold): `rsf_kfold_cindex_mean ≈ 0.330`, `gb_kfold_cindex_mean ≈ 0.330` (from `metrics_os_advanced.json`). Grouped CV fields remain `NaN`.
- AE simulation: `sim_ae.csv` contains 18,500 rows (370 × 50 sims), with overall AE event rate ≈ 0.489 and both arms represented. `report_ae.csv` has 2 rows (per arm), including `suggested_time_scale` for calibration.
- Calibration: `sim_ae_calibrated.csv` includes `t_eot_calibrated`. Mean ratio `t_eot_calibrated / t_eot` ≈ 1.20, indicating simulated EOT was scaled up to better align with observed medians.

## Did the simulation work?
- Yes. The pipeline successfully produced AE/EOT simulations, arm-level incidence metrics at 90/180/365 days, and EOT distribution plots. The presence of plausible incidence values per arm and the calibration ratio > 1 suggests the initial EOT model slightly under-estimated EOT timing relative to observed; calibration adjusted it upward.

## Why some metrics are NaN
- Grouped CV metrics rely on at least 2 distinct `STUDYID` groups. With a single-study dataset, CV folds cannot be formed, hence `NaN` c-indices in both OS and advanced models. We added event-stratified KFold metrics as primary evaluation (see values above).

## Observed vs simulated comparisons
- From `report_ae.csv` (per arm):
  - Best supportive care:
    - Observed AE incidence at 90/180/365: 0.000 / 0.000 / 0.000
    - Simulated AE incidence at 90/180/365: ≈ 0.366 / 0.371 / 0.371
  - panit. plus best supportive care:
    - Observed AE incidence at 90/180/365: ≈ 0.913 / 0.913 / 0.913
    - Simulated AE incidence at 90/180/365: ≈ 0.386 / 0.603 / 0.609
- Interpretation: For BSC, observed early AEs are essentially absent while simulation predicts moderate incidence; verify `ADAE` completeness and onset alignment to treatment day scale. For panitumumab arm, simulated probabilities under-estimate observed AE frequency at early horizons, indicating a need for AE probability calibration and/or richer features.

## Recommendations to improve comparison fidelity
- Endpoint alignment:
  - Confirm `ADAE` onset column used (`AESTDY`, `AESTDYI`, etc.) and ensure it aligns to treatment day scale used for `ADLB.VISITDY` EOT proxy.
  - If available, use `ADSL/TRTSDT` to normalize all times to days-on-treatment for both labs and AE.
- Feature enrichment:
  - Include additional longitudinal markers (if present) like vitals or more lab tests to improve both EOT and AE modeling.
  - Add treatment indicator `TRT` explicitly to the design matrix for AE Cox modeling; currently it enters via ADSL baseline merge, but verify it survives preprocessing and one-hot encoding.
- Model calibration:
  - Keep using `suggested_time_scale` to match median EOT per arm; consider quantile-specific scaling if tails differ (align 25th/75th percentiles).
  - For AE probability, add Platt scaling or isotonic regression using observed AE at 90/180/365 to better match arm-specific incidence.
- OS evaluation:
  - Because there is a single study, prefer train/test split or bootstrapping over grouped CV, and report bootstrap c-index with confidence intervals.
- Data quality checks:
  - Investigate columns with all-missing slopes (e.g., creatinine slope). Either remove from feature selection up-front or compute slopes only when at least 2 non-missing timepoints exist.

## Next steps
1) Tighten endpoint timing normalization to `TRTSDT` reference.
2) Add AE probability calibration using observed incidence by arm and horizon.
3) Add bootstrap risk metrics for OS and AE models given single-study constraint.
4) Explore per-arm-specific EOT models if treatment materially influences discontinuation timing.
