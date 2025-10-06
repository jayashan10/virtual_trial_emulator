"""PDS310: Digital Twin & AI Simulation for Colorectal Cancer Clinical Trials.

Based on study 20020408 (Panitumumab + BSC vs BSC alone).

This package implements the CAMP methodology for digital twin generation and
virtual clinical trial simulation for colorectal cancer patients.
"""

__version__ = "0.2.0"

# Core data loading
from .io import load_adam_tables, audit_tables

# Digital profile system
from .digital_profile import (
    create_complete_digital_profile,
    get_profile_feature_groups,
)

from .profile_database import (
    create_profile_database,
    load_profile_database,
    get_profile_by_id,
    get_profiles_by_criteria,
    get_database_summary,
    export_database_summary,
    split_train_test,
)

from .profile_viz import (
    plot_profile_distribution,
    plot_profile_comparison,
    plot_correlation_heatmap,
    plot_subgroup_comparison,
    export_profile_summary_table,
    plot_feature_importance_from_profiles,
)

# Feature engineering
from .features_baseline import assemble_baseline_feature_view
from .features_longitudinal import build_longitudinal_features

# Endpoints
from .endpoints import (
    derive_os_from_adsl,
    derive_eot_from_adlb,
    derive_ae_from_adae,
    prepare_eot_competing_from_adam,
)

__all__ = [
    # Data loading
    "load_adam_tables",
    "audit_tables",
    # Digital profiles
    "create_complete_digital_profile",
    "get_profile_feature_groups",
    "create_profile_database",
    "load_profile_database",
    "get_profile_by_id",
    "get_profiles_by_criteria",
    "get_database_summary",
    "export_database_summary",
    "split_train_test",
    # Visualization
    "plot_profile_distribution",
    "plot_profile_comparison",
    "plot_correlation_heatmap",
    "plot_subgroup_comparison",
    "export_profile_summary_table",
    "plot_feature_importance_from_profiles",
    # Features
    "assemble_baseline_feature_view",
    "build_longitudinal_features",
    # Endpoints
    "derive_os_from_adsl",
    "derive_eot_from_adlb",
    "derive_ae_from_adae",
    "prepare_eot_competing_from_adam",
]


