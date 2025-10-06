# PDS310 (ADaM) Pipelines

This module processes PDS310 ADaM SAS tables (`ADSL`, `ADLB`, `ADAE`) to derive endpoints, train overall survival (OS) models, and simulate AE/EOT, producing overlays and reports.

## Data
Place the ADaM `.sas7bdat` files under:
```
AllProvidedFiles_310/PDS_DSA_20020408/
```
Required: `adsl_*.sas7bdat`, `adlb_*.sas7bdat`. Optional but recommended: `adae_*.sas7bdat` for AE/EOT.

## Config
`pds310/config.yaml` controls paths and endpoint mappings. Defaults point inside this repo.

## Run (uv)
```bash
uv sync

# OS baseline + overlays
uv run -m pds310.cli os --config pds310/config.yaml
uv run -m pds310.cli os-overlay --config pds310/config.yaml --sims 50

# AE/EOT pipeline
uv run -m pds310.cli ae --config pds310/config.yaml
uv run -m pds310.cli report-ae --config pds310/config.yaml
uv run -m pds310.cli calibrate-ae --config pds310/config.yaml
uv run -m pds310.cli plots --config pds310/config.yaml

# Advanced OS (RSF, GB)
uv run -m pds310.cli os-advanced --config pds310/config.yaml
```

Artifacts are written to `outputs/pds310/`.

## Outputs
- OS: `cox.joblib`, `aft_weibull.joblib` (if converged), `metrics_os.json`, `plot_os_overlay_PDS310.png`, `km_input.csv`
- AE/EOT: `ae_cox.joblib`, `eot_model.joblib`, `sim_ae.csv`, `report_ae.csv`, `report_ae_summary.json`, `sim_ae_calibrated.csv`, `plot_ae_incidence_*.png`, `plot_eot_distributions.png`
- Advanced OS: `os_rsf.joblib`, `os_gb.joblib`, `metrics_os_advanced.json`

## Quick validation summary
- Dataset: 370 subjects, 1 study (`STUDYID=PDS310`).
- AE sim: 50 simulations per subject; overall AE event rate ≈ 0.489.
- Arms detected from `ADSL`: `Best supportive care`, `panit. plus best supportive care`.
- OS CV metrics in `metrics_os.json`/`metrics_os_advanced.json` show `NaN` c-index because only a single study is present (grouped CV by study is not applicable).

## Notes
- OS event uses `ADSL.DTHX` (normalized to 0/1); time uses `ADSL.DTHDYX` with optional date fallback.
- ARM is taken from `ADSL.TRT` (or `ARM`/`ARMCD` when present).
- EOT is proxied from `ADLB.VISITDY` (last visit day); AE is earliest `ADAE` onset day if available.
- Some longitudinal features (e.g., creatinine slope) may be entirely missing in PDS310 and are safely dropped during preprocessing.
