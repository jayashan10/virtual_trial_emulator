import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter

from .io import load_adam_tables
from .endpoints import derive_os_from_adsl, prepare_eot_competing_from_adam
from .features_baseline import assemble_baseline_feature_view
from .features_longitudinal import build_longitudinal_features

# Reuse generic modeling utilities from pds149
from pds149.model_os import fit_cox, fit_weibull_aft_robust, _transform  # type: ignore
from .os_simulate_cox import simulate_os_times_cox  # use PDS310-specific simulator
from .model_ae import fit_cause_specific_cox_ae, fit_eot_allcause
from .simulate import simulate_patients
from .reporting import write_report
from .plotting import plot_ae_incidence, plot_eot_distributions_from_tables
from .cv import cox_kfold_cindex


ID_COL = "SUBJID"
STUDY_COL = "STUDYID"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_baseline_os(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    out_root = cfg["outputs_root"]
    ensure_dir(out_root)

    tables = load_adam_tables(data_root)

    os_df = derive_os_from_adsl(
        tables["adsl"],
        death_flag_col=cfg["endpoints"]["os"].get("death_flag_col", "DTHX"),
        time_col=cfg["endpoints"]["os"].get("time_col", "DTHDYX"),
        trtsdt_col=cfg["endpoints"]["os"].get("trtsdt_col"),
        dthdt_col=cfg["endpoints"]["os"].get("dthdt_col"),
    )
    X_base = assemble_baseline_feature_view(tables)
    X_long = build_longitudinal_features(tables.get("adlb"))
    X_full = X_base.merge(X_long, on=[ID_COL, STUDY_COL], how="left")
    df = X_full.merge(os_df[[ID_COL, STUDY_COL, "time", "event"]], on=[ID_COL, STUDY_COL], how="inner")
    df = df[df["time"] > 0].reset_index(drop=True)

    cox_res = fit_cox(df, groups=df[STUDY_COL].astype(str))
    joblib.dump({"model": cox_res["model"], "preprocessor": cox_res["preprocessor"], "feature_names": cox_res["feature_names"]}, os.path.join(out_root, "cox.joblib"))

    aft_cv = float("nan")
    try:
        aft_res = fit_weibull_aft_robust(df, groups=df[STUDY_COL].astype(str))
        pre_obj = aft_res.get("preprocessor") or aft_res.get("pre")
        feat_names = aft_res.get("feature_names") or aft_res.get("feat_names")
        joblib.dump({"model": aft_res["model"], "pre": pre_obj, "feat_names": feat_names}, os.path.join(out_root, "aft_weibull.joblib"))
        aft_cv = float(aft_res.get("cv_cindex_mean", float("nan")))
    except Exception:
        # Proceed without AFT
        pass

    # Single-study friendly CV (KFold stratified by event) for Cox
    kfold_ci = cox_kfold_cindex(df, n_splits=min(5, max(2, int(len(df) ** 0.5))), random_state=42)
    metrics = {
        "cox_cv_cindex_mean": cox_res["cv_cindex_mean"],  # may be NaN with single study
        "cox_kfold_cindex_mean": kfold_ci,
        "aft_cv_cindex_mean": aft_cv,
        "n_patients": int(len(df)),
        "n_studies": int(df[STUDY_COL].nunique()),
    }
    with open(os.path.join(out_root, "metrics_os.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    X_pre_df = _transform(cox_res["preprocessor"], df, cox_res["feature_names"])
    risk = cox_res["model"].predict_partial_hazard(X_pre_df).values.ravel()
    df_km = df[["time", "event"]].copy()
    df_km["high_risk"] = (risk > pd.Series(risk).median()).astype(int)
    df_km.to_csv(os.path.join(out_root, "km_input.csv"), index=False)


def run_os_overlay(config_path: str, sims: int = 50) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    out_root = cfg["outputs_root"]
    ensure_dir(out_root)

    tables = load_adam_tables(data_root)
    os_df = derive_os_from_adsl(
        tables["adsl"],
        death_flag_col=cfg["endpoints"]["os"].get("death_flag_col", "DTHX"),
        time_col=cfg["endpoints"]["os"].get("time_col", "DTHDYX"),
        trtsdt_col=cfg["endpoints"]["os"].get("trtsdt_col"),
        dthdt_col=cfg["endpoints"]["os"].get("dthdt_col"),
    )
    X_base = assemble_baseline_feature_view(tables)
    X_long = build_longitudinal_features(tables.get("adlb"))
    X_full = X_base.merge(X_long, on=[ID_COL, STUDY_COL], how="left")
    df = X_full.merge(os_df[[ID_COL, STUDY_COL, "time", "event"]], on=[ID_COL, STUDY_COL], how="inner")
    df = df[df["time"] > 0].reset_index(drop=True)

    from pds149.model_os import fit_cox  # local import to avoid confusion
    cox_res = fit_cox(df, groups=df[STUDY_COL].astype(str))

    # Simulate OS times using Cox sampler; build ARM from ADSL
    adsl_for_arm = tables["adsl"]
    if "ARM" not in adsl_for_arm.columns and "ARMCD" not in adsl_for_arm.columns:
        adsl_for_arm = adsl_for_arm.copy()
        adsl_for_arm["ARM"] = "ARM"

    df_sim = simulate_os_times_cox(cox_res, df.drop(columns=["time", "event"]), os_df, adsl_for_arm, n_sim=sims)

    # Prepare observed df with ARM
    from .os_eval import observed_os_df
    df_obs = observed_os_df(tables["adsl"], os_df)

    # Overlay plots
    from pds149.plotting import plot_os_overlays  # reuse plotting
    paths = plot_os_overlays(df_obs, df_sim, out_root)
    print("OS overlays saved:")
    for p in paths:
        print(p)


def run_ae_and_sim(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    out_root = cfg["outputs_root"]
    ensure_dir(out_root)

    tables = load_adam_tables(data_root)

    X_base = assemble_baseline_feature_view(tables)
    X_long = build_longitudinal_features(tables.get("adlb"))
    X_full = X_base.merge(X_long, on=[ID_COL, STUDY_COL], how="left")

    eot = prepare_eot_competing_from_adam(tables.get("adsl"), tables.get("adlb"), tables.get("adae"))
    if eot is None or eot.empty:
        raise ValueError("Unable to derive EOT from ADLB; ensure ADLB with VISITDY is available.")

    df = X_full.merge(eot, on=[ID_COL, STUDY_COL], how="inner")
    df = df[df["time"] > 0].reset_index(drop=True)

    ae_res = fit_cause_specific_cox_ae(df)
    eot_res = fit_eot_allcause(df)

    joblib.dump(ae_res, os.path.join(out_root, "ae_cox.joblib"))
    joblib.dump(eot_res, os.path.join(out_root, "eot_model.joblib"))

    sim_df = simulate_patients(
        df_features=X_full,
        aft_allcause=eot_res,
        cox_ae=ae_res["model"],
        ohe=ae_res["ohe"],
        feature_cols=ae_res["feature_cols"],
        adsl=tables["adsl"],
        n_sim=cfg.get("simulation", {}).get("n_simulations", 100),
        seed=cfg.get("simulation", {}).get("seed", 42),
    )
    sim_path = os.path.join(out_root, "sim_ae.csv")
    sim_df.to_csv(sim_path, index=False)


def run_report_ae(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    tables = load_adam_tables(cfg["data_root"])
    out_dir = cfg["outputs_root"]
    sim_path = os.path.join(out_dir, "sim_ae.csv")
    report_path = write_report(tables.get("adsl"), tables.get("adlb"), tables.get("adae"), sim_path, out_dir)
    print(f"AE report written: {report_path}")


def run_plots(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["outputs_root"]
    rep = os.path.join(out_dir, "report_ae.csv")
    tables = load_adam_tables(cfg["data_root"])
    ae_plots = plot_ae_incidence(rep, out_dir)
    sim_csv = os.path.join(out_dir, "sim_ae_calibrated.csv") if os.path.exists(os.path.join(out_dir, "sim_ae_calibrated.csv")) else os.path.join(out_dir, "sim_ae.csv")
    eot_plot = plot_eot_distributions_from_tables(tables.get("adlb"), sim_csv, out_dir)
    print("Plots saved:")
    for p in ae_plots + [eot_plot]:
        print(p)


def run_os_advanced(config_path: str) -> None:
    from pds149.model_os_advanced import fit_rsf, fit_gb  # type: ignore
    from .cv import rsf_kfold_cindex, gb_kfold_cindex
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    out_root = cfg["outputs_root"]
    ensure_dir(out_root)

    tables = load_adam_tables(data_root)
    os_df = derive_os_from_adsl(
        tables["adsl"],
        death_flag_col=cfg["endpoints"]["os"].get("death_flag_col", "DTHX"),
        time_col=cfg["endpoints"]["os"].get("time_col", "DTHDYX"),
        trtsdt_col=cfg["endpoints"]["os"].get("trtsdt_col"),
        dthdt_col=cfg["endpoints"]["os"].get("dthdt_col"),
    )
    X_base = assemble_baseline_feature_view(tables)
    X_long = build_longitudinal_features(tables.get("adlb"))
    X_full = X_base.merge(X_long, on=[ID_COL, STUDY_COL], how="left")
    df = X_full.merge(os_df[[ID_COL, STUDY_COL, "time", "event"]], on=[ID_COL, STUDY_COL], how="inner")
    df = df[df["time"] > 0].reset_index(drop=True)

    rsf_res = fit_rsf(df, groups=df[STUDY_COL].astype(str))
    gb_res = fit_gb(df, groups=df[STUDY_COL].astype(str))

    joblib.dump(rsf_res, os.path.join(out_root, "os_rsf.joblib"))
    joblib.dump(gb_res, os.path.join(out_root, "os_gb.joblib"))

    # Single-study friendly KFold c-index for RSF/GB
    n_splits = min(5, max(2, int(len(df) ** 0.5)))
    metrics = {
        "rsf_cv_cindex_mean": rsf_res["cv_cindex_mean"],  # may be NaN
        "gb_cv_cindex_mean": gb_res["cv_cindex_mean"],    # may be NaN
        "rsf_kfold_cindex_mean": rsf_kfold_cindex(df, n_splits=n_splits, random_state=42),
        "gb_kfold_cindex_mean": gb_kfold_cindex(df, n_splits=n_splits, random_state=42),
        "n_patients": int(len(df)),
        "n_studies": int(df[STUDY_COL].nunique()),
    }
    with open(os.path.join(out_root, "metrics_os_advanced.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("OS advanced metrics written")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDS310 pipelines")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p1 = sub.add_parser("os", help="Run OS baseline pipeline for PDS310")
    p1.add_argument("--config", type=str, default="pds310/config.yaml")

    p2 = sub.add_parser("os-overlay", help="Simulated vs observed OS overlays for PDS310")
    p2.add_argument("--config", type=str, default="pds310/config.yaml")
    p2.add_argument("--sims", type=int, default=50)

    p3 = sub.add_parser("ae", help="Run AE competing risks + EOT simulation for PDS310")
    p3.add_argument("--config", type=str, default="pds310/config.yaml")

    p4 = sub.add_parser("report-ae", help="Generate AE observed vs simulated report for PDS310")
    p4.add_argument("--config", type=str, default="pds310/config.yaml")

    p5 = sub.add_parser("calibrate-ae", help="Apply EOT time scaling and regenerate sim_ae_calibrated.csv for PDS310")
    p5.add_argument("--config", type=str, default="pds310/config.yaml")

    p6 = sub.add_parser("plots", help="Generate AE incidence and EOT distribution plots for PDS310")
    p6.add_argument("--config", type=str, default="pds310/config.yaml")

    p7 = sub.add_parser("os-advanced", help="Train RSF and GB survival models for OS (PDS310)")
    p7.add_argument("--config", type=str, default="pds310/config.yaml")

    args = parser.parse_args()

    if args.cmd == "os-overlay":
        run_os_overlay(args.config, sims=args.sims)
    elif args.cmd == "ae":
        run_ae_and_sim(args.config)
    elif args.cmd == "report-ae":
        run_report_ae(args.config)
    elif args.cmd == "calibrate-ae":
        from pds149.calibration import apply_time_scaling  # reuse
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        out_dir = cfg["outputs_root"]
        sim_in = os.path.join(out_dir, "sim_ae.csv")
        rep = os.path.join(out_dir, "report_ae.csv")
        sim_out = os.path.join(out_dir, "sim_ae_calibrated.csv")
        path = apply_time_scaling(sim_in, rep, sim_out)
        print(f"Calibrated sim written: {path}")
    elif args.cmd == "plots":
        run_plots(args.config)
    elif args.cmd == "os-advanced":
        run_os_advanced(args.config)
    else:
        run_baseline_os(args.config)


