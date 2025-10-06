from typing import List
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .endpoints import derive_eot_from_adlb


def plot_ae_incidence(report_csv: str, out_dir: str, horizons: List[int] = [90, 180, 365]) -> List[str]:
    # Reuse implementation from pds149
    from pds149.plotting import plot_ae_incidence as _plot
    return _plot(report_csv, out_dir, horizons)


def plot_eot_distributions_from_tables(adlb: pd.DataFrame, sim_csv: str, out_dir: str, n_bins: int = 50) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sim = pd.read_csv(sim_csv)

    # Observed EOT from ADLB
    eot = derive_eot_from_adlb(adlb)
    obs = pd.to_numeric(eot.get("time"), errors="coerce").dropna()
    obs = obs[obs > 0]

    sim_col = "t_eot_calibrated" if "t_eot_calibrated" in sim.columns else "t_eot"
    if "SUBJID" in sim.columns and "sim" in sim.columns:
        sim_agg = sim.groupby(["STUDYID", "SUBJID"]) [sim_col].median().reset_index()[sim_col]
    else:
        sim_agg = pd.to_numeric(sim[sim_col], errors="coerce")
    sim_agg = sim_agg.dropna()
    sim_agg = sim_agg[sim_agg > 0]

    lo = float(np.nanpercentile(pd.concat([obs, sim_agg]), 1)) if len(obs) and len(sim_agg) else 0
    hi = float(np.nanpercentile(pd.concat([obs, sim_agg]), 99)) if len(obs) and len(sim_agg) else (obs.max() if len(obs) else sim_agg.max())
    if hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, n_bins + 1)

    plt.figure(figsize=(10, 6))
    sns.histplot(obs, bins=bins, stat="density", color="steelblue", alpha=0.4, label="Observed EOT")
    sns.histplot(sim_agg, bins=bins, stat="density", color="orange", alpha=0.4, label="Simulated EOT")
    plt.legend()
    plt.title("End-of-Treatment Time Distribution: Observed vs Simulated (density)")
    plt.xlabel("Days")
    plt.ylabel("Density")
    plt.tight_layout()
    p = os.path.join(out_dir, "plot_eot_distributions.png")
    plt.savefig(p)
    plt.close()
    return p


