from typing import Dict, Tuple, List
import os
import json
import numpy as np
import pandas as pd

ID_COL = "RPT"
STUDY_COL = "STUDYID"


def build_arm_identifier(core: pd.DataFrame) -> pd.Series:
    parts = []
    for c in ["TRT1_ID", "TRT2_ID", "TRT3_ID"]:
        if c in core.columns:
            parts.append(core[c].astype("string").fillna(""))
    if not parts:
        return pd.Series(["ARM" for _ in range(len(core))], index=core.index)
    arm = parts[0]
    for p in parts[1:]:
        arm = arm.where(p.str.len() == 0, arm + "+" + p)
    arm = arm.str.strip("+")
    arm = arm.replace({"": "UNSPEC"})
    return arm


def _observed_ae_metrics(core: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    df = core[[ID_COL, STUDY_COL, "ENTRT_PC", "ENDTRS_C", "TRT1_ID", "TRT2_ID", "TRT3_ID"]].copy()
    df.rename(columns={"ENTRT_PC": "time", "ENDTRS_C": "cause"}, inplace=True)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    df["cause"] = df["cause"].astype("string").str.strip()
    df["is_ae"] = df["cause"].isin(["AE", "possible_AE"]).astype(int)
    df["ARM"] = build_arm_identifier(df)

    rows = []
    for (study, arm), g in df.groupby([STUDY_COL, "ARM"]):
        median_eot = float(np.median(g["time"])) if len(g) > 0 else float("nan")
        iqr_eot = float(np.subtract(*np.percentile(g["time"], [75, 25]))) if len(g) > 0 else float("nan")
        rec = {"STUDYID": study, "ARM": arm, "obs_median_eot": median_eot, "obs_iqr_eot": iqr_eot}
        for h in horizons:
            rec[f"obs_incidence_ae_{h}"] = float(((g["time"] <= h) & (g["is_ae"] == 1)).mean()) if len(g) > 0 else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def _sim_ae_metrics(sim: pd.DataFrame, core: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    arm_map = core[[ID_COL, STUDY_COL, "TRT1_ID", "TRT2_ID", "TRT3_ID"]].copy()
    arm_map["ARM"] = build_arm_identifier(arm_map)
    arm_map = arm_map[[ID_COL, STUDY_COL, "ARM"]]

    s = sim.merge(arm_map, on=[ID_COL, STUDY_COL], how="left")

    rows = []
    for (study, arm, sim_id), g in s.groupby([STUDY_COL, "ARM", "sim"]):
        rec = {"STUDYID": study, "ARM": arm, "sim": int(sim_id)}
        rec["sim_median_eot"] = float(np.median(g["t_eot"])) if len(g) > 0 else float("nan")
        for h in horizons:
            rec[f"sim_incidence_ae_{h}"] = float(((g["t_eot"] <= h) & (g["event_ae"] == 1)).mean()) if len(g) > 0 else float("nan")
        rows.append(rec)
    df = pd.DataFrame(rows)
    agg = df.groupby(["STUDYID", "ARM"]).agg({
        "sim_median_eot": "mean",
        **{f"sim_incidence_ae_{h}": "mean" for h in horizons}
    }).reset_index()
    return agg


def compare_and_calibrate(core: pd.DataFrame, sim: pd.DataFrame, horizons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obs = _observed_ae_metrics(core, horizons)
    simagg = _sim_ae_metrics(sim, core, horizons)
    comp = obs.merge(simagg, on=["STUDYID", "ARM"], how="left")
    comp["suggested_time_scale"] = comp.apply(lambda r: (r["obs_median_eot"] / r["sim_median_eot"]) if (pd.notna(r["obs_median_eot"]) and pd.notna(r["sim_median_eot"]) and r["sim_median_eot"] > 0) else np.nan, axis=1)
    return comp, obs


def write_report(core: pd.DataFrame, sim_path: str, out_dir: str, horizons: List[int] = [90, 180, 365]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sim = pd.read_csv(sim_path)
    comp, obs = compare_and_calibrate(core, sim, horizons)
    out_csv = os.path.join(out_dir, "report_ae.csv")
    comp.to_csv(out_csv, index=False)
    summary = {
        "n_studies": int(comp["STUDYID"].nunique()),
        "n_arms": int(comp["ARM"].nunique()),
        "horizons": horizons,
    }
    with open(os.path.join(out_dir, "report_ae_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return out_csv
