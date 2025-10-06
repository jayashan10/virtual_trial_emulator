"""
Profile Visualization for PDS310 Digital Twin System.

Provides visualization tools for exploring and comparing patient digital profiles.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .digital_profile import get_profile_feature_groups
from .io import ID_COL


def plot_profile_distribution(
    db: pd.DataFrame,
    output_dir: Optional[str] = None,
    figsize: tuple = (20, 15)
) -> None:
    """
    Visualize feature distributions across all patients.
    
    Creates multi-panel figure showing:
    - Demographic distributions
    - Lab value distributions
    - Tumor characteristics
    - Risk score distributions
    
    Args:
        db: Profile database
        output_dir: Directory to save plots (optional)
        figsize: Figure size
    """
    feature_groups = get_profile_feature_groups()
    
    fig = plt.figure(figsize=figsize)
    
    # Demographics
    ax_idx = 1
    
    # Age distribution
    if "AGE" in db.columns:
        plt.subplot(4, 4, ax_idx)
        ax_idx += 1
        db["AGE"].hist(bins=20, edgecolor="black")
        plt.xlabel("Age (years)")
        plt.ylabel("Count")
        plt.title("Age Distribution")
        plt.axvline(db["AGE"].median(), color="red", linestyle="--", label=f"Median: {db['AGE'].median():.1f}")
        plt.legend()
    
    # Sex distribution
    if "SEX" in db.columns:
        plt.subplot(4, 4, ax_idx)
        ax_idx += 1
        sex_counts = db["SEX"].value_counts()
        sex_counts.plot(kind="bar", color=["steelblue", "coral"])
        plt.xlabel("Sex")
        plt.ylabel("Count")
        plt.title("Sex Distribution")
        plt.xticks(rotation=0)
    
    # ECOG distribution
    if "B_ECOG" in db.columns:
        plt.subplot(4, 4, ax_idx)
        ax_idx += 1
        ecog_counts = db["B_ECOG"].value_counts().sort_index()
        ecog_counts.plot(kind="bar", color="steelblue")
        plt.xlabel("ECOG Performance Status")
        plt.ylabel("Count")
        plt.title("ECOG Distribution")
        plt.xticks(rotation=0)
    
    # Treatment arms
    if "TRT" in db.columns:
        plt.subplot(4, 4, ax_idx)
        ax_idx += 1
        trt_counts = db["TRT"].value_counts()
        trt_counts.plot(kind="barh", color="steelblue")
        plt.xlabel("Count")
        plt.ylabel("Treatment")
        plt.title("Treatment Distribution")
    
    # RAS status
    if "RAS_status" in db.columns:
        plt.subplot(4, 4, ax_idx)
        ax_idx += 1
        ras_counts = db["RAS_status"].value_counts()
        colors = {"WILD-TYPE": "green", "MUTANT": "red", "UNKNOWN": "gray"}
        ras_counts.plot(kind="bar", color=[colors.get(x, "steelblue") for x in ras_counts.index])
        plt.xlabel("RAS Status")
        plt.ylabel("Count")
        plt.title("RAS Mutation Status")
        plt.xticks(rotation=45, ha="right")
    
    # Baseline labs
    lab_features = ["baseline_HGB", "baseline_LDH", "baseline_ALP", "baseline_CEA"]
    for lab in lab_features:
        if lab in db.columns and ax_idx <= 16:
            plt.subplot(4, 4, ax_idx)
            ax_idx += 1
            db[lab].hist(bins=20, edgecolor="black")
            plt.xlabel(lab.replace("baseline_", ""))
            plt.ylabel("Count")
            plt.title(f"{lab.replace('baseline_', '')} Distribution")
            plt.axvline(db[lab].median(), color="red", linestyle="--", 
                       label=f"Median: {db[lab].median():.1f}")
            plt.legend(fontsize=8)
    
    # Risk scores
    risk_features = ["composite_risk_score", "lab_risk_score", "performance_risk", "tumor_burden_risk"]
    for risk in risk_features:
        if risk in db.columns and ax_idx <= 16:
            plt.subplot(4, 4, ax_idx)
            ax_idx += 1
            db[risk].hist(bins=20, edgecolor="black", color="coral")
            plt.xlabel(risk.replace("_", " ").title())
            plt.ylabel("Count")
            plt.title(f"{risk.replace('_', ' ').title()}")
            plt.axvline(db[risk].median(), color="darkred", linestyle="--",
                       label=f"Median: {db[risk].median():.2f}")
            plt.legend(fontsize=8)
    
    plt.tight_layout()
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / "profile_distributions.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved profile distributions to: {output_path}")
    
    plt.show()


def plot_profile_comparison(
    profile1: Dict[str, Any],
    profile2: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: tuple = (15, 10)
) -> None:
    """
    Compare two patient profiles side-by-side.
    
    Args:
        profile1: First patient profile
        profile2: Second patient profile
        output_path: Path to save plot (optional)
        figsize: Figure size
    """
    # Select features to compare
    features_to_compare = [
        "AGE", "SEX", "B_ECOG", "RAS_status",
        "baseline_HGB", "baseline_LDH", "baseline_ALP",
        "target_lesion_count", "sum_target_diameters",
        "composite_risk_score", "lab_risk_score",
        "best_response"
    ]
    
    # Filter to available features
    features_to_compare = [f for f in features_to_compare 
                          if f in profile1 and f in profile2]
    
    fig, axes = plt.subplots(len(features_to_compare), 1, figsize=figsize)
    
    if len(features_to_compare) == 1:
        axes = [axes]
    
    subj1_id = profile1.get(ID_COL, "Patient 1")
    subj2_id = profile2.get(ID_COL, "Patient 2")
    
    for idx, feature in enumerate(features_to_compare):
        ax = axes[idx]
        
        val1 = profile1[feature]
        val2 = profile2[feature]
        
        # Handle numeric vs categorical
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Numeric - bar chart
            ax.barh([subj1_id, subj2_id], [val1, val2], color=["steelblue", "coral"])
            ax.set_xlabel(feature)
            ax.set_title(f"{feature}: {val1:.2f} vs {val2:.2f}")
        else:
            # Categorical - text display
            ax.axis("off")
            ax.text(0.1, 0.5, f"{feature}:", fontsize=12, fontweight="bold")
            ax.text(0.4, 0.5, f"{subj1_id}: {val1}", fontsize=10, color="steelblue")
            ax.text(0.4, 0.3, f"{subj2_id}: {val2}", fontsize=10, color="coral")
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved profile comparison to: {output_path}")
    
    plt.show()


def plot_correlation_heatmap(
    db: pd.DataFrame,
    feature_subset: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 10)
) -> None:
    """
    Plot correlation heatmap for numeric features.
    
    Args:
        db: Profile database
        feature_subset: Specific features to include (default: all numeric)
        output_path: Path to save plot (optional)
        figsize: Figure size
    """
    # Select numeric features
    numeric_cols = db.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns
    numeric_cols = [c for c in numeric_cols if c not in [ID_COL, "STUDYID"]]
    
    if feature_subset:
        numeric_cols = [c for c in numeric_cols if c in feature_subset]
    
    # Limit to reasonable number
    if len(numeric_cols) > 40:
        # Prioritize key features
        priority_features = [
            "AGE", "B_ECOG", "B_WEIGHT",
            "baseline_HGB", "baseline_LDH", "baseline_ALP", "baseline_CEA",
            "sum_target_diameters", "target_lesion_count",
            "composite_risk_score", "lab_risk_score", "performance_risk",
            "DTHDYX", "PFSDYCR"
        ]
        numeric_cols = [c for c in priority_features if c in numeric_cols][:40]
    
    # Calculate correlations
    corr = db[numeric_cols].corr()
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved correlation heatmap to: {output_path}")
    
    plt.show()


def plot_subgroup_comparison(
    db: pd.DataFrame,
    subgroup_col: str,
    metric_col: str,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Compare metric distributions across subgroups.
    
    Args:
        db: Profile database
        subgroup_col: Column defining subgroups (e.g., "RAS_status", "TRT")
        metric_col: Metric to compare (e.g., "composite_risk_score", "DTHDYX")
        output_path: Path to save plot (optional)
        figsize: Figure size
    """
    if subgroup_col not in db.columns or metric_col not in db.columns:
        raise ValueError(f"Columns {subgroup_col} and/or {metric_col} not found in database")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Box plot
    ax = axes[0]
    subgroups = db[subgroup_col].dropna().unique()
    data_to_plot = [db[db[subgroup_col] == sg][metric_col].dropna().values 
                    for sg in subgroups]
    
    bp = ax.boxplot(data_to_plot, labels=subgroups, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
    
    ax.set_xlabel(subgroup_col.replace("_", " ").title())
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_title(f"{metric_col} by {subgroup_col}")
    ax.tick_params(axis='x', rotation=45)
    
    # Violin plot
    ax = axes[1]
    df_plot = db[[subgroup_col, metric_col]].dropna()
    
    parts = ax.violinplot(
        [df_plot[df_plot[subgroup_col] == sg][metric_col].values for sg in subgroups],
        positions=range(len(subgroups)),
        showmeans=True,
        showmedians=True
    )
    
    ax.set_xticks(range(len(subgroups)))
    ax.set_xticklabels(subgroups, rotation=45, ha="right")
    ax.set_xlabel(subgroup_col.replace("_", " ").title())
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_title(f"{metric_col} Distribution by {subgroup_col}")
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved subgroup comparison to: {output_path}")
    
    plt.show()


def export_profile_summary_table(
    db: pd.DataFrame,
    output_path: str,
    format: str = "csv"
) -> None:
    """
    Export human-readable summary table of key profile features.
    
    Args:
        db: Profile database
        output_path: Path to save table
        format: Output format ("csv" or "html")
    """
    # Select key features for summary
    summary_features = [
        ID_COL, "AGE", "SEX", "RACE", "B_ECOG", "TRT",
        "RAS_status", "baseline_HGB", "baseline_LDH",
        "sum_target_diameters", "tumor_burden_category",
        "composite_risk_score", "best_response",
        "DTHDYX", "DTHX"
    ]
    
    # Filter to available features
    summary_features = [f for f in summary_features if f in db.columns]
    
    summary_df = db[summary_features].copy()
    
    # Round numeric columns
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(2)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == "csv":
        summary_df.to_csv(output_path, index=False)
    elif format == "html":
        summary_df.to_html(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'html'")
    
    print(f"Profile summary table saved to: {output_path}")


def plot_feature_importance_from_profiles(
    db: pd.DataFrame,
    target_col: str = "DTHX",
    top_n: int = 20,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> None:
    """
    Calculate and visualize feature importance for predicting target outcome.
    
    Uses mutual information for feature importance.
    
    Args:
        db: Profile database
        target_col: Target outcome column
        top_n: Number of top features to show
        output_path: Path to save plot (optional)
        figsize: Figure size
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    
    if target_col not in db.columns:
        raise ValueError(f"Target column '{target_col}' not found in database")
    
    # Prepare features
    feature_cols = [c for c in db.columns if c not in [ID_COL, "STUDYID", target_col]]
    
    # Select numeric features only
    numeric_features = db[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = db[numeric_features].fillna(db[numeric_features].median())
    y = db[target_col].fillna(0)
    
    # Calculate mutual information
    if db[target_col].dtype in [np.float64, np.float32]:
        mi_scores = mutual_info_regression(X, y, random_state=42)
    else:
        mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "feature": numeric_features,
        "importance": mi_scores
    }).sort_values("importance", ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.barh(range(len(importance_df)), importance_df["importance"].values, color="steelblue")
    plt.yticks(range(len(importance_df)), importance_df["feature"].values)
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Features for Predicting {target_col}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved feature importance plot to: {output_path}")
    
    plt.show()
    
    return importance_df
