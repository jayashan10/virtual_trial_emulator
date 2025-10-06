import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from .io import load_training_tables, ensure_ids_are_string
from .endpoints import derive_os_from_core
from .features_baseline import assemble_baseline_feature_view
from .features_longitudinal import build_longitudinal_features
from .model_os import fit_cox, fit_weibull_aft, fit_weibull_aft_robust
from .model_ae import prepare_eot_competing, fit_cause_specific_cox_ae, fit_eot_allcause
from .simulate import simulate_patients
from .reporting import write_report
from .calibration import apply_time_scaling
from .plotting import plot_ae_incidence, plot_eot_distributions, plot_km_overlays, plot_os_overlays
from .os_eval import observed_os_df, km_curves_by_study_arm, hazard_ratios_by_study
from .os_simulate import simulate_os_times
from .os_simulate_cox import simulate_os_times_cox


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_baseline_os(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    out_root = cfg["outputs_root"]
    ensure_dir(out_root)

    tables = load_training_tables(data_root)
    for k in list(tables.keys()):
        tables[k] = ensure_ids_are_string(tables[k])

    os_df = derive_os_from_core(
        tables["core"],
        time_col=cfg["endpoints"]["os"]["time_col"],
        death_col=cfg["endpoints"]["os"]["death_flag_col"],
        time_unit_col=cfg["endpoints"]["os"]["time_unit_col"],
    )
    X_base = assemble_baseline_feature_view(tables)
    X_long = build_longitudinal_features(tables.get("labs"), tables.get("vitals"))
    X_full = X_base.merge(X_long, on=["RPT", "STUDYID"], how="left")
    df = X_full.merge(os_df[["RPT", "STUDYID", "time", "event"]], on=["RPT", "STUDYID"], how="inner")
    df = df[df["time"] > 0].reset_index(drop=True)

    cox_res = fit_cox(df, groups=df["STUDYID"].astype(str))
    joblib.dump({"model": cox_res["model"], "pre": cox_res["preprocessor"], "feat_names": cox_res["feature_names"]}, os.path.join(out_root, "cox.joblib"))

    aft_res = fit_weibull_aft(df, groups=df["STUDYID"].astype(str))
    joblib.dump({"model": aft_res["model"], "pre": aft_res["preprocessor"], "feat_names": aft_res["feature_names"]}, os.path.join(out_root, "aft_weibull.joblib"))

    metrics = {
        "cox_cv_cindex_mean": cox_res["cv_cindex_mean"],
        "aft_cv_cindex_mean": aft_res["cv_cindex_mean"],
        "n_patients": int(len(df)),
        "n_studies": int(df["STUDYID"].nunique()),
    }
    with open(os.path.join(out_root, "metrics_os.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    from .model_os import _transform
    X_pre_df = _transform(cox_res["preprocessor"], df, cox_res["feature_names"])
    risk = cox_res["model"].predict_partial_hazard(X_pre_df).values.ravel()
    df_km = df[["time", "event"]].copy()
    df_km["high_risk"] = (risk > pd.Series(risk).median()).astype(int)
    df_km.to_csv(os.path.join(out_root, "km_input.csv"), index=False)


def run_ae_and_sim(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    out_root = cfg["outputs_root"]
    ensure_dir(out_root)

    tables = load_training_tables(data_root)
    for k in list(tables.keys()):
        tables[k] = ensure_ids_are_string(tables[k])

    X_base = assemble_baseline_feature_view(tables)
    X_long = build_longitudinal_features(tables.get("labs"), tables.get("vitals"))
    X_full = X_base.merge(X_long, on=["RPT", "STUDYID"], how="left")

    eot = prepare_eot_competing(tables["core"], time_col=cfg["endpoints"]["ae_disc"]["time_col"], reason_col=cfg["endpoints"]["ae_disc"]["reason_col"])

    df = X_full.merge(eot, on=["RPT", "STUDYID"], how="inner")
    df = df[df["time"] > 0].reset_index(drop=True)

    ae_res = fit_cause_specific_cox_ae(df)
    eot_res = fit_eot_allcause(df)

    joblib.dump(ae_res, os.path.join(out_root, "ae_cox.joblib"))
    joblib.dump(eot_res, os.path.join(out_root, "eot_model.joblib"))

    df_features = X_full.copy()

    sim_df = simulate_patients(
        df_features=df_features,
        aft_allcause=eot_res,
        cox_ae=ae_res["model"],
        ohe=ae_res["ohe"],
        feature_cols=ae_res["feature_cols"],
        n_sim=cfg.get("simulation", {}).get("n_simulations", 100),
        seed=cfg.get("simulation", {}).get("seed", 42),
    )

    sim_path = os.path.join(out_root, "sim_ae.csv")
    sim_df.to_csv(sim_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDS149 pipelines")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p1 = sub.add_parser("os", help="Run OS baseline pipeline")
    p1.add_argument("--config", type=str, default="config.yaml")

    p2 = sub.add_parser("ae", help="Run AE competing risks + simulation")
    p2.add_argument("--config", type=str, default="config.yaml")

    p3 = sub.add_parser("report-ae", help="Generate AE observed vs simulated report")
    p3.add_argument("--config", type=str, default="config.yaml")

    p4 = sub.add_parser("os-advanced", help="Train RSF and GB survival models for OS")
    p4.add_argument("--config", type=str, default="config.yaml")

    p5 = sub.add_parser("calibrate-ae", help="Apply EOT time scaling from report_ae.csv and regenerate sim_ae_calibrated.csv")
    p5.add_argument("--config", type=str, default="config.yaml")

    p6 = sub.add_parser("plots", help="Generate plots for AE incidence and EOT distributions")
    p6.add_argument("--config", type=str, default="config.yaml")

    p7 = sub.add_parser("os-observed", help="Observed OS KM and HRs by study/arm")
    p7.add_argument("--config", type=str, default="config.yaml")

    p8 = sub.add_parser("os-overlay", help="Simulated vs observed OS overlays using AFT model")
    p8.add_argument("--config", type=str, default="config.yaml")
    p8.add_argument("--sims", type=int, default=50)

    args = parser.parse_args()

    if args.cmd == "ae":
        run_ae_and_sim(args.config)
    elif args.cmd == "report-ae":
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        tables = load_training_tables(cfg["data_root"])
        for k in list(tables.keys()):
            tables[k] = ensure_ids_are_string(tables[k])
        out_dir = cfg["outputs_root"]
        sim_path = os.path.join(out_dir, "sim_ae.csv")
        report_path = write_report(tables["core"], sim_path, out_dir)
        print(f"AE report written: {report_path}")
    elif args.cmd == "os-advanced":
        from .model_os_advanced import fit_rsf, fit_gb  # type: ignore
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        tables = load_training_tables(cfg["data_root"])
        for k in list(tables.keys()):
            tables[k] = ensure_ids_are_string(tables[k])
        os_df = derive_os_from_core(
            tables["core"],
            time_col=cfg["endpoints"]["os"]["time_col"],
            death_col=cfg["endpoints"]["os"]["death_flag_col"],
            time_unit_col=cfg["endpoints"]["os"]["time_unit_col"],
        )
        X_base = assemble_baseline_feature_view(tables)
        X_long = build_longitudinal_features(tables.get("labs"), tables.get("vitals"))
        X_full = X_base.merge(X_long, on=["RPT", "STUDYID"], how="left")
        df = X_full.merge(os_df[["RPT", "STUDYID", "time", "event"]], on=["RPT", "STUDYID"], how="inner")
        df = df[df["time"] > 0].reset_index(drop=True)
        rsf_res = fit_rsf(df, groups=df["STUDYID"].astype(str))
        gb_res = fit_gb(df, groups=df["STUDYID"].astype(str))
        out_dir = cfg["outputs_root"]
        joblib.dump(rsf_res, os.path.join(out_dir, "os_rsf.joblib"))
        joblib.dump(gb_res, os.path.join(out_dir, "os_gb.joblib"))
        metrics = {
            "rsf_cv_cindex_mean": rsf_res["cv_cindex_mean"],
            "gb_cv_cindex_mean": gb_res["cv_cindex_mean"],
            "n_patients": int(len(df)),
            "n_studies": int(df["STUDYID"].nunique()),
        }
        with open(os.path.join(out_dir, "metrics_os_advanced.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("OS advanced metrics written")
    elif args.cmd == "calibrate-ae":
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        out_dir = cfg["outputs_root"]
        sim_in = os.path.join(out_dir, "sim_ae.csv")
        rep = os.path.join(out_dir, "report_ae.csv")
        sim_out = os.path.join(out_dir, "sim_ae_calibrated.csv")
        path = apply_time_scaling(sim_in, rep, sim_out)
        print(f"Calibrated sim written: {path}")
    elif args.cmd == "plots":
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        out_dir = cfg["outputs_root"]
        rep = os.path.join(out_dir, "report_ae.csv")
        ae_plots = plot_ae_incidence(rep, out_dir)
        core_csv = os.path.join(cfg["data_root"], "CoreTable_training.csv")
        sim_csv = os.path.join(out_dir, "sim_ae_calibrated.csv") if os.path.exists(os.path.join(out_dir, "sim_ae_calibrated.csv")) else os.path.join(out_dir, "sim_ae.csv")
        eot_plot = plot_eot_distributions(core_csv, sim_csv, out_dir)
        print("Plots saved:")
        for p in ae_plots + [eot_plot]:
            print(p)
    elif args.cmd == "os-observed":
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        tables = load_training_tables(cfg["data_root"])
        for k in list(tables.keys()):
            tables[k] = ensure_ids_are_string(tables[k])
        os_df = derive_os_from_core(
            tables["core"],
            time_col=cfg["endpoints"]["os"]["time_col"],
            death_col=cfg["endpoints"]["os"]["death_flag_col"],
            time_unit_col=cfg["endpoints"]["os"]["time_unit_col"],
        )
        df_obs = observed_os_df(tables["core"], os_df)
        # Save HRs
        hrs = hazard_ratios_by_study(df_obs)
        out_dir = cfg["outputs_root"]
        hrs.to_csv(os.path.join(out_dir, "os_observed_hrs.csv"), index=False)
        # KM overlays per study
        times = pd.Series(df_obs["time"]).quantile([0.0, 1.0]).values
        t_grid = np.linspace(times[0], times[1], 200)
        curves = km_curves_by_study_arm(df_obs, t_grid)
        paths = plot_km_overlays(curves, out_dir)
        print("Observed OS artifacts:")
        for p in paths:
            print(p)
        print(os.path.join(out_dir, "os_observed_hrs.csv"))
    elif args.cmd == "os-overlay":
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        tables = load_training_tables(cfg["data_root"])
        for k in list(tables.keys()):
            tables[k] = ensure_ids_are_string(tables[k])
        # Fit Cox from baseline OS pipeline inputs
        os_df = derive_os_from_core(
            tables["core"],
            time_col=cfg["endpoints"]["os"]["time_col"],
            death_col=cfg["endpoints"]["os"]["death_flag_col"],
            time_unit_col=cfg["endpoints"]["os"]["time_unit_col"],
        )
        X_base = assemble_baseline_feature_view(tables)
        X_long = build_longitudinal_features(tables.get("labs"), tables.get("vitals"))
        X_full = X_base.merge(X_long, on=["RPT", "STUDYID"], how="left")
        df = X_full.merge(os_df[["RPT", "STUDYID", "time", "event"]], on=["RPT", "STUDYID"], how="inner")
        df = df[df["time"] > 0].reset_index(drop=True)
        cox_res = fit_cox(df, groups=df["STUDYID"].astype(str))
        # Simulate OS times with censoring from observed using Cox-based sampler
        df_sim = simulate_os_times_cox(cox_res, df.drop(columns=["time", "event"]), os_df, tables["core"], n_sim=args.sims)
        # Prepare observed df with ARM
        df_obs = observed_os_df(tables["core"], os_df)
        # Overlay plots
        out_dir = cfg["outputs_root"]
        paths = plot_os_overlays(df_obs, df_sim, out_dir)
        print("OS overlays saved:")
        for p in paths:
            print(p)
    else:
        run_baseline_os(args.config)
