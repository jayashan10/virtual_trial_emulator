# PDS149: Clinical Trial Modeling and Simulation (Project Data Sphere 149)

This repository implements a reproducible pipeline to learn from the Prostate Cancer DREAM Challenge mCRPC trials (Project Data Sphere content 149), validate against observed outcomes, and simulate trial endpoints.

## What this does
- Derives endpoints from the training split (`CoreTable_training.csv`):
  - Overall survival (OS): time `LKADT_P` and event flag `DEATH`.
  - On-treatment end-of-treatment (EOT) with AE competing risks via `ENDTRS_C` and `ENTRT_PC`.
- Builds feature views:
  - Baseline: demographics, ECOG, metastasis flags, labs, prior meds, comorbidities.
  - Longitudinal: windowed aggregates for key labs/vitals in baseline (−30–0 d) and early treatment (1–60 d).
- Models:
  - OS: Cox PH and Weibull AFT baselines (with grouped CV by `STUDYID`).
  - AE/EOT: cause-specific Cox for AE; robust AFT (Weibull/LogNormal with fallbacks) for all-cause EOT.
- Simulation:
  - Multi-state EOT simulation (sample EOT and AE indicator) and reporting.
- Reporting & plots:
  - Observed vs simulated AE incidence at 90/180/365 days.
  - EOT distribution overlays with density normalization and fair binning.
  - Suggested per-study/arm time-scaling for calibration.

## Data
Place the training CSVs under:
```
AllProvidedFiles_149/prostate_cancer_challenge_data_training/
```
The CLI uses the path in `config.yaml`.

## Quick start (uv)
1) Install deps and lock:
```bash
uv sync
```
2) Run pipelines:
```bash
uv run python -m pds149.cli ae --config config.yaml
uv run python -m pds149.cli report-ae --config config.yaml
uv run python -m pds149.cli calibrate-ae --config config.yaml
uv run python -m pds149.cli plots --config config.yaml
```
3) Optional OS baseline:
```bash
uv run python -m pds149.cli os --config config.yaml
```
Artifacts are written to `outputs/`.

### PDS310 (ADaM) pipelines

New, separate module `pds310/` processes the ADaM tables in `AllProvidedFiles_310/PDS_DSA_20020408`.

1) Configure paths in `pds310/config.yaml` (defaults are absolute paths under this repo).
2) Run OS baseline training and metrics:

```bash
uv run -m pds310.cli os --config pds310/config.yaml
```

3) Generate simulated vs observed OS overlays:

```bash
uv run -m pds310.cli os-overlay --config pds310/config.yaml --sims 50
```

## Code structure
- `pds149/io.py` – table loading with encoding/NA handling; ID normalization.
- `pds149/endpoints.py` – OS and AE/EOT label derivation.
- `pds149/features_baseline.py` – baseline feature view from core/med history/prior meds.
- `pds149/features_longitudinal.py` – lab/vital windowed aggregates (last/mean/slope/count).
- `pds149/model_os.py` – Cox/Weibull with robust preprocessing, adaptive CV.
- `pds149/model_ae.py` – cause-specific Cox for AE; robust AFT with fallbacks for EOT.
- `pds149/simulate.py` – EOT sampling + AE cause probability; aligned encoders.
- `pds149/reporting.py` – observed vs simulated AE metrics; suggested time scaling.
- `pds149/calibration.py` – apply time scaling to simulated EOT per study/arm.
- `pds149/plotting.py` – plots for AE incidence and EOT distributions with fair comparison.
- `pds149/cli.py` – subcommands: `os`, `ae`, `report-ae`, `calibrate-ae`, `plots`.

## Configuration
See `config.yaml` for data roots, endpoints mapping, CV options, and simulation seeds.

## Notes
- Leaderboard/final scoring sets have labels withheld; this pipeline focuses on training split for evaluation and simulation. You can still produce predictions for those splits by adapting the CLI to load alternate roots.

## License
MIT
