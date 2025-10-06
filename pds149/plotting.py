import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def plot_ae_incidence(report_csv: str, out_dir: str, horizons: List[int] = [90, 180, 365]) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(report_csv)
    paths = []
    for h in horizons:
        col_obs = f"obs_incidence_ae_{h}"
        col_sim = f"sim_incidence_ae_{h}"
        if col_obs not in df.columns or col_sim not in df.columns:
            continue
        p = os.path.join(out_dir, f"plot_ae_incidence_{h}.png")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=col_obs, y=col_sim, hue="ARM")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"AE Incidence at {h} days: Observed vs Simulated")
        plt.xlabel("Observed")
        plt.ylabel("Simulated")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths.append(p)
    return paths


def plot_eot_distributions(core_csv: str, sim_csv: str, out_dir: str, n_bins: int = 50) -> str:
    os.makedirs(out_dir, exist_ok=True)
    core = pd.read_csv(core_csv)
    sim = pd.read_csv(sim_csv)

    # Observed EOT (drop NAs and nonpositive)
    obs = pd.to_numeric(core.get("ENTRT_PC"), errors="coerce").dropna()
    obs = obs[obs > 0]

    # Simulated EOT: if multiple sims per patient, aggregate to median per patient for fair comparison
    sim_col = "t_eot_calibrated" if "t_eot_calibrated" in sim.columns else "t_eot"
    if "RPT" in sim.columns and "sim" in sim.columns:
        sim_agg = sim.groupby(["STUDYID", "RPT"])[sim_col].median().reset_index()[sim_col]
    else:
        sim_agg = pd.to_numeric(sim[sim_col], errors="coerce")
    sim_agg = sim_agg.dropna()
    sim_agg = sim_agg[sim_agg > 0]

    # Common bin edges using combined 1st-99th percentile range to avoid extreme tails dominating
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


def plot_km_overlays(curves: Dict[Tuple[str, str], pd.DataFrame], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    # Group by study and plot arms together
    studies = sorted({k[0] for k in curves.keys()})
    for study in studies:
        plt.figure(figsize=(10, 6))
        for (s, arm), df in curves.items():
            if s != study:
                continue
            plt.step(df["time"], df["survival"], where="post", label=arm)
        plt.title(f"Observed OS KM by arm - {study}")
        plt.xlabel("Days")
        plt.ylabel("Survival probability")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, f"plot_os_km_{study}.png")
        plt.savefig(p)
        plt.close()
        paths.append(p)
    return paths


def plot_os_overlays(df_obs: pd.DataFrame, df_sim: pd.DataFrame, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for study, g_obs_s in df_obs.groupby("STUDYID"):
        plt.figure(figsize=(10, 6))
        for arm in sorted(g_obs_s["ARM"].dropna().unique()):
            g_obs = g_obs_s[g_obs_s["ARM"] == arm]
            km = KaplanMeierFitter(label=f"Obs {arm}")
            km.fit(g_obs["time"].values, event_observed=g_obs["event"].values)
            km.plot(ci_show=False)
            # Simulated overlay (median over sims)
            g_sim = df_sim[(df_sim["STUDYID"].astype(str) == str(study)) & (df_sim["ARM"] == arm)]
            if not g_sim.empty:
                # Compute KM per sim then average survival at event times
                times = np.linspace(0, g_obs["time"].max(), 200)
                survs = []
                for sim_id, g in g_sim.groupby("sim"):
                    km_s = KaplanMeierFitter()
                    km_s.fit(g["time"].values, event_observed=g["event"].values)
                    s = km_s.survival_function_at_times(times).values.reshape(-1)
                    survs.append(s)
                mean_surv = np.nanmean(np.vstack(survs), axis=0) if survs else None
                if mean_surv is not None:
                    plt.step(times, mean_surv, where="post", label=f"Sim {arm}", linestyle="--")
        plt.title(f"OS KM Overlays - {study}")
        plt.xlabel("Days")
        plt.ylabel("Survival probability")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, f"plot_os_overlay_{study}.png")
        plt.savefig(p)
        plt.close()
        paths.append(p)
    return paths
