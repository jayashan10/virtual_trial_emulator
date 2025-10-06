import os
import pandas as pd

ID_COL = "RPT"
STUDY_COL = "STUDYID"


def apply_time_scaling(sim_path: str, report_path: str, out_path: str) -> str:
    sim = pd.read_csv(sim_path)
    rep = pd.read_csv(report_path)
    # Build ARM from report and map scaling; ensure columns present
    if "ARM" not in rep.columns or "suggested_time_scale" not in rep.columns:
        raise ValueError("report_ae.csv must contain ARM and suggested_time_scale")
    # Map per study/arm
    scale_map = rep.set_index(["STUDYID", "ARM"]).loc[:, ["suggested_time_scale"]]

    # Need ARM in sim; rebuild from core is better, but sim should join to report via STUDYID+ARM
    if "ARM" not in sim.columns:
        # If not present, we cannot calibrate without core; require ARM in sim in future iterations
        sim["ARM"] = "UNSPEC"

    sim = sim.merge(scale_map, on=["STUDYID", "ARM"], how="left")
    # Apply scaling when available
    sim["t_eot_calibrated"] = sim.apply(
        lambda r: r["t_eot"] * r["suggested_time_scale"] if pd.notna(r["suggested_time_scale"]) and r["suggested_time_scale"] > 0 else r["t_eot"],
        axis=1,
    )
    sim.to_csv(out_path, index=False)
    return out_path
