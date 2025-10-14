from typing import Dict, List, Tuple
import os
import json
import numpy as np
import pandas as pd

from .endpoints import derive_eot_from_adlb, derive_ae_from_adae


ID_COL = "SUBJID"
STUDY_COL = "STUDYID"


def _arm_map_from_adsl(adsl: pd.DataFrame) -> pd.DataFrame:
    arm_col = None
    for c in ["ATRT", "TRT", "ARM", "ARMCD"]:
        if c in adsl.columns:
            arm_col = c
            break
    if arm_col is None:
        m = adsl[[ID_COL, STUDY_COL]].copy()
        m["ARM"] = "ARM"
    else:
        m = adsl[[ID_COL, STUDY_COL, arm_col]].copy().rename(columns={arm_col: "ARM"})
    return m


def _observed_ae_metrics(adsl: pd.DataFrame, adlb: pd.DataFrame, adae: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    eot = derive_eot_from_adlb(adlb)
    ae = derive_ae_from_adae(adae)
    arm_map = _arm_map_from_adsl(adsl)

    base = eot.merge(arm_map, on=[ID_COL, STUDY_COL], how="left")
    base = base.merge(ae[[ID_COL, STUDY_COL, "ae_time"]] if not ae.empty else pd.DataFrame(columns=[ID_COL, STUDY_COL, "ae_time"]),
                      on=[ID_COL, STUDY_COL], how="left")

    rows = []
    for (study, arm), g in base.groupby([STUDY_COL, "ARM"]):
        median_eot = float(np.median(g["time"])) if len(g) > 0 else float("nan")
        iqr_eot = float(np.subtract(*np.percentile(g["time"], [75, 25]))) if len(g) > 0 else float("nan")
        rec = {"STUDYID": study, "ARM": arm, "obs_median_eot": median_eot, "obs_iqr_eot": iqr_eot}
        for h in horizons:
            rec[f"obs_incidence_ae_{h}"] = float(((pd.to_numeric(g["ae_time"], errors="coerce") <= h)).fillna(False).mean()) if len(g) > 0 else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def _sim_ae_metrics(sim: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    rows = []
    if sim is None or sim.empty:
        return pd.DataFrame(columns=["STUDYID", "ARM"])  # empty
    for (study, arm, sim_id), g in sim.groupby([STUDY_COL, "ARM", "sim"]):
        rec = {"STUDYID": study, "ARM": arm, "sim": int(sim_id)}
        rec["sim_median_eot"] = float(np.median(pd.to_numeric(g["t_eot"], errors="coerce").dropna())) if len(g) > 0 else float("nan")
        for h in horizons:
            t = pd.to_numeric(g["t_eot"], errors="coerce")
            e = g["event_ae"].astype(int)
            rec[f"sim_incidence_ae_{h}"] = float(((t <= h) & (e == 1)).mean()) if len(g) > 0 else float("nan")
        rows.append(rec)
    df = pd.DataFrame(rows)
    agg = df.groupby(["STUDYID", "ARM"]).agg({
        "sim_median_eot": "mean",
        **{f"sim_incidence_ae_{h}": "mean" for h in horizons},
    }).reset_index()
    return agg


def write_report(adsl: pd.DataFrame, adlb: pd.DataFrame, adae: pd.DataFrame, sim_path: str, out_dir: str,
                 horizons: List[int] = [90, 180, 365]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sim = pd.read_csv(sim_path)

    obs = _observed_ae_metrics(adsl, adlb, adae, horizons)
    simagg = _sim_ae_metrics(sim, horizons)
    comp = obs.merge(simagg, on=["STUDYID", "ARM"], how="left")
    comp["suggested_time_scale"] = comp.apply(
        lambda r: (r["obs_median_eot"] / r["sim_median_eot"]) if (pd.notna(r["obs_median_eot"]) and pd.notna(r["sim_median_eot"]) and r["sim_median_eot"] > 0) else np.nan,
        axis=1,
    )

    out_csv = os.path.join(out_dir, "report_ae.csv")
    comp.to_csv(out_csv, index=False)

    summary = {
        "n_studies": int(comp["STUDYID"].nunique()) if not comp.empty else 0,
        "n_arms": int(comp["ARM"].nunique()) if not comp.empty else 0,
        "horizons": horizons,
    }
    with open(os.path.join(out_dir, "report_ae_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return out_csv


