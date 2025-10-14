"""
Twin Diversity Metrics for PDS310.

Measures how well digital twin cohorts represent the real patient population.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def calculate_twin_diversity(
    twins: pd.DataFrame,
    real_profiles: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Measure how well twins represent real population.
    
    Metrics:
    - Coverage: % of feature space covered by twins
    - Distribution match: KS test per feature
    - Novelty: % of twins outside real data convex hull
    - Representativeness: Mean distance to nearest real patient
    
    Args:
        twins: DataFrame of digital twins
        real_profiles: DataFrame of real patients
        verbose: Print summary
    
    Returns:
        Dictionary with diversity metrics
    """
    if verbose:
        print(f"Calculating diversity metrics...")
        print(f"  Twins: {len(twins)}")
        print(f"  Real patients: {len(real_profiles)}")
    
    metrics = {}
    
    # 1. Feature space coverage
    metrics["coverage"] = _calculate_coverage(twins, real_profiles)
    
    # 2. Distribution matching
    metrics["distribution_match"] = _calculate_distribution_match(twins, real_profiles)
    
    # 3. Novelty score
    metrics["novelty"] = _calculate_novelty(twins, real_profiles)
    
    # 4. Representativeness
    metrics["representativeness"] = _calculate_representativeness(twins, real_profiles)
    
    # 5. Population balance
    metrics["population_balance"] = _calculate_population_balance(twins, real_profiles)
    
    # Overall diversity score (weighted average)
    weights = {
        "coverage_score": 0.25,
        "distribution_match_score": 0.30,
        "novelty_score": 0.15,
        "representativeness_score": 0.20,
        "balance_score": 0.10,
    }
    
    overall_score = sum(
        metrics[metric][metric] * weight 
        for metric, weight in weights.items()
        if metric in metrics
    )
    
    metrics["overall_diversity_score"] = overall_score
    
    if verbose:
        print(f"\nDiversity Metrics:")
        print(f"  Coverage: {metrics['coverage']['coverage_score']:.3f}")
        print(f"  Distribution Match: {metrics['distribution_match']['distribution_match_score']:.3f}")
        print(f"  Novelty: {metrics['novelty']['novelty_score']:.3f}")
        print(f"  Representativeness: {metrics['representativeness']['representativeness_score']:.3f}")
        print(f"  Population Balance: {metrics['population_balance']['balance_score']:.3f}")
        print(f"  Overall Score: {overall_score:.3f}")
    
    return metrics


def _calculate_coverage(
    twins: pd.DataFrame,
    real_profiles: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate what percentage of the feature space is covered by twins.
    """
    # Select numeric features
    numeric_cols = [c for c in real_profiles.select_dtypes(include=[np.number]).columns 
                   if c in twins.columns]
    
    if len(numeric_cols) == 0:
        return {"coverage_score": 0.0, "covered_features": 0, "total_features": 0}
    
    # For each feature, check if twin range covers real range
    covered_count = 0
    
    for col in numeric_cols:
        real_min = real_profiles[col].min()
        real_max = real_profiles[col].max()
        twin_min = twins[col].min()
        twin_max = twins[col].max()
        
        # Check if twin range overlaps significantly with real range
        # Consider covered if twin range spans at least 80% of real range
        if not pd.isna(real_min) and not pd.isna(real_max):
            real_range = real_max - real_min
            if real_range > 0:
                twin_coverage = (min(twin_max, real_max) - max(twin_min, real_min)) / real_range
                if twin_coverage >= 0.8:
                    covered_count += 1
    
    coverage_score = covered_count / len(numeric_cols) if len(numeric_cols) > 0 else 0.0
    
    return {
        "coverage_score": coverage_score,
        "covered_features": covered_count,
        "total_features": len(numeric_cols),
    }


def _calculate_distribution_match(
    twins: pd.DataFrame,
    real_profiles: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compare distributions using Kolmogorov-Smirnov test.
    """
    # Select numeric features
    numeric_cols = [c for c in real_profiles.select_dtypes(include=[np.number]).columns 
                   if c in twins.columns]
    
    if len(numeric_cols) == 0:
        return {"distribution_match_score": 0.0, "matching_features": 0}
    
    ks_results = {}
    matching_count = 0
    
    for col in numeric_cols:
        real_values = real_profiles[col].dropna()
        twin_values = twins[col].dropna()
        
        if len(real_values) > 0 and len(twin_values) > 0:
            # Perform KS test
            ks_stat, p_value = stats.ks_2samp(real_values, twin_values)
            ks_results[col] = {"statistic": ks_stat, "p_value": p_value}
            
            # Consider matching if p > 0.05 (distributions not significantly different)
            if p_value > 0.05:
                matching_count += 1
    
    match_score = matching_count / len(numeric_cols) if len(numeric_cols) > 0 else 0.0
    
    # Also compute mean p-value as alternative metric
    mean_p_value = np.mean([r["p_value"] for r in ks_results.values()]) if ks_results else 0.0
    
    return {
        "distribution_match_score": match_score,
        "matching_features": matching_count,
        "total_features": len(numeric_cols),
        "mean_p_value": mean_p_value,
        "ks_results": ks_results,
    }


def _calculate_novelty(
    twins: pd.DataFrame,
    real_profiles: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate percentage of twins that are novel (outside real data range).
    
    Novelty is good up to a point - too many novel twins = unrealistic.
    Target: 10-30% novelty.
    """
    # Select numeric features for PCA
    numeric_cols = [c for c in real_profiles.select_dtypes(include=[np.number]).columns 
                   if c in twins.columns]
    
    if len(numeric_cols) < 2:
        return {"novelty_score": 0.5, "novel_twins": 0, "novelty_percentage": 0.0}
    
    # Reduce to 2D using PCA for convex hull analysis
    X_real = real_profiles[numeric_cols].dropna()
    X_twins = twins[numeric_cols].dropna()
    
    if len(X_real) < 10 or len(X_twins) < 10:
        return {"novelty_score": 0.5, "novel_twins": 0, "novelty_percentage": 0.0}
    
    # Fit PCA on real data
    pca = PCA(n_components=min(2, len(numeric_cols)))
    pca.fit(X_real.fillna(X_real.median()))
    
    # Transform both real and twin data
    real_pca = pca.transform(X_real.fillna(X_real.median()))
    twin_pca = pca.transform(X_twins.fillna(X_real.median()))
    
    # Calculate distances from each twin to nearest real patient
    distances = cdist(twin_pca, real_pca, metric='euclidean')
    min_distances = distances.min(axis=1)
    
    # Define "novel" as being further than 95th percentile of real-to-real distances
    real_to_real = cdist(real_pca, real_pca, metric='euclidean')
    # Get upper triangle (exclude self-distances)
    triu_indices = np.triu_indices_from(real_to_real, k=1)
    real_distances = real_to_real[triu_indices]
    novelty_threshold = np.percentile(real_distances, 95)
    
    novel_count = (min_distances > novelty_threshold).sum()
    novelty_percentage = novel_count / len(twin_pca) * 100
    
    # Score novelty: optimal is 10-30%, penalize if too low or too high
    if 10 <= novelty_percentage <= 30:
        novelty_score = 1.0
    elif novelty_percentage < 10:
        novelty_score = novelty_percentage / 10
    else:  # > 30%
        novelty_score = max(0, 1.0 - (novelty_percentage - 30) / 70)
    
    return {
        "novelty_score": novelty_score,
        "novel_twins": int(novel_count),
        "novelty_percentage": float(novelty_percentage),
        "threshold_distance": float(novelty_threshold),
    }


def _calculate_representativeness(
    twins: pd.DataFrame,
    real_profiles: pd.DataFrame
) -> Dict[str, Any]:
    """
    Measure how well twins represent real patients.
    
    Lower mean distance to nearest real patient = more representative.
    """
    # Select numeric features
    numeric_cols = [c for c in real_profiles.select_dtypes(include=[np.number]).columns 
                   if c in twins.columns]
    
    if len(numeric_cols) < 2:
        return {"representativeness_score": 0.5, "mean_distance": None}
    
    # Prepare data
    X_real = real_profiles[numeric_cols].fillna(real_profiles[numeric_cols].median())
    X_twins = twins[numeric_cols].fillna(real_profiles[numeric_cols].median())
    
    if len(X_real) < 2 or len(X_twins) < 2:
        return {"representativeness_score": 0.5, "mean_distance": None}
    
    # Normalize features (0-1 scale) for fair distance comparison
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_real)
    
    X_real_scaled = scaler.transform(X_real)
    X_twins_scaled = scaler.transform(X_twins)
    
    # Calculate distances from each twin to nearest real patient
    distances = cdist(X_twins_scaled, X_real_scaled, metric='euclidean')
    min_distances = distances.min(axis=1)
    mean_distance = min_distances.mean()
    
    # Also calculate mean distance among real patients for reference
    real_to_real = cdist(X_real_scaled, X_real_scaled, metric='euclidean')
    triu_indices = np.triu_indices_from(real_to_real, k=1)
    real_mean_distance = real_to_real[triu_indices].mean()
    
    # Score: lower is better, normalize by real-to-real distance
    relative_distance = mean_distance / real_mean_distance if real_mean_distance > 0 else 1.0
    
    # Good if twins are as close to real as real patients are to each other
    # Score = 1.0 if relative_distance ≈ 1.0
    representativeness_score = np.exp(-abs(relative_distance - 1.0))
    
    return {
        "representativeness_score": representativeness_score,
        "mean_distance": float(mean_distance),
        "real_mean_distance": float(real_mean_distance),
        "relative_distance": float(relative_distance),
    }


def _calculate_population_balance(
    twins: pd.DataFrame,
    real_profiles: pd.DataFrame
) -> Dict[str, Any]:
    """
    Check if twin cohort maintains population balance for key subgroups.
    """
    balance_features = ["SEX", "RAS_status", "ATRT", "B_ECOG"]
    
    differences = []
    
    for feature in balance_features:
        if feature in twins.columns and feature in real_profiles.columns:
            # Get proportions
            real_props = real_profiles[feature].value_counts(normalize=True)
            twin_props = twins[feature].value_counts(normalize=True)
            
            # Calculate maximum absolute difference
            all_values = set(real_props.index).union(set(twin_props.index))
            max_diff = max(
                abs(real_props.get(val, 0) - twin_props.get(val, 0))
                for val in all_values
            )
            differences.append(max_diff)
    
    if len(differences) == 0:
        return {"balance_score": 0.5, "mean_difference": None}
    
    mean_difference = np.mean(differences)
    
    # Score: 1.0 if perfect balance, decrease linearly
    balance_score = max(0, 1.0 - mean_difference * 2)  # Penalize differences
    
    return {
        "balance_score": balance_score,
        "mean_difference": float(mean_difference),
        "max_difference": float(max(differences)),
    }


def plot_twin_vs_real_comparison(
    twins: pd.DataFrame,
    real_profiles: pd.DataFrame,
    features: List[str] = None,
    output_path: str = None
):
    """
    Create visualization comparing twin and real distributions.
    
    Args:
        twins: DataFrame of digital twins
        real_profiles: DataFrame of real patients
        features: List of features to plot (default: key features)
        output_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if features is None:
        features = ["AGE", "B_WEIGHT", "baseline_HGB", "baseline_LDH", 
                   "composite_risk_score", "sum_target_diameters"]
        features = [f for f in features if f in twins.columns and f in real_profiles.columns]
    
    n_features = len(features)
    if n_features == 0:
        print("No common features to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features[:6]):
        ax = axes[idx]
        
        real_values = real_profiles[feature].dropna()
        twin_values = twins[feature].dropna()
        
        # Plot histograms
        ax.hist(real_values, bins=20, alpha=0.5, label='Real', color='steelblue', density=True)
        ax.hist(twin_values, bins=20, alpha=0.5, label='Twins', color='coral', density=True)
        
        # KS test
        ks_stat, p_value = stats.ks_2samp(real_values, twin_values)
        
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'{feature}\nKS p-value: {p_value:.3f}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    
    plt.close()


def export_diversity_report(
    metrics: Dict[str, Any],
    output_path: str
):
    """
    Export diversity metrics to human-readable report.
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DIGITAL TWIN COHORT DIVERSITY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Overall Diversity Score: {metrics['overall_diversity_score']:.3f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("COVERAGE\n")
        f.write("-" * 80 + "\n")
        cov = metrics["coverage"]
        f.write(f"Covered Features: {cov['covered_features']}/{cov['total_features']}\n")
        f.write(f"Coverage Score: {cov['coverage_score']:.3f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("DISTRIBUTION MATCHING\n")
        f.write("-" * 80 + "\n")
        dm = metrics["distribution_match"]
        f.write(f"Matching Features: {dm['matching_features']}/{dm['total_features']}\n")
        f.write(f"Mean KS p-value: {dm['mean_p_value']:.3f}\n")
        f.write(f"Distribution Match Score: {dm['distribution_match_score']:.3f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("NOVELTY\n")
        f.write("-" * 80 + "\n")
        nov = metrics["novelty"]
        f.write(f"Novel Twins: {nov['novel_twins']} ({nov['novelty_percentage']:.1f}%)\n")
        f.write(f"Novelty Score: {nov['novelty_score']:.3f}\n")
        f.write(f"Target: 10-30% novelty for optimal diversity\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("REPRESENTATIVENESS\n")
        f.write("-" * 80 + "\n")
        rep = metrics["representativeness"]
        f.write(f"Mean Distance to Nearest Real: {rep['mean_distance']:.4f}\n")
        f.write(f"Real-to-Real Mean Distance: {rep['real_mean_distance']:.4f}\n")
        f.write(f"Relative Distance: {rep['relative_distance']:.3f}\n")
        f.write(f"Representativeness Score: {rep['representativeness_score']:.3f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("POPULATION BALANCE\n")
        f.write("-" * 80 + "\n")
        bal = metrics["population_balance"]
        f.write(f"Mean Proportion Difference: {bal['mean_difference']:.3f}\n")
        f.write(f"Max Proportion Difference: {bal['max_difference']:.3f}\n")
        f.write(f"Balance Score: {bal['balance_score']:.3f}\n\n")
    
    print(f"Diversity report saved to: {output_path}")
