"""
features/columns.py
===================
Lista canonica de features que entram nos modelos.
Separada do builder para facilitar ablation studies
(remover/adicionar features sem tocar na logica de engenharia).
"""

# Features presentes no CSV original que entram diretamente
ORIGINAL_COLS = [
    "area_first_ha",
    "log1p_area_first",
    "area_growth_rate_ha_per_h",
    "area_growth_rel_0_5h",
    "log1p_growth",
    "log_area_ratio_0_5h",
    "radial_growth_rate_m_per_h",
    "centroid_speed_m_per_h",
    "centroid_displacement_m",
    "dist_min_ci_0_5h",
    "dist_std_ci_0_5h",
    "dist_slope_ci_0_5h",
    "closing_speed_m_per_h",
    "closing_speed_abs_m_per_h",
    "projected_advance_m",
    "dist_accel_m_per_h2",
    "dist_fit_r2_0_5h",
    "alignment_abs",
    "alignment_cos",
    "along_track_speed",
    "cross_track_component",
    "spread_bearing_sin",
    "spread_bearing_cos",
    "num_perimeters_0_5h",
    "dt_first_last_0_5h",
    "low_temporal_resolution_0_5h",
    "event_start_hour",
    "event_start_month",
    "event_start_dayofweek",
]

# Features criadas pelo pipeline de engenharia (features/builder.py)
ENGINEERED_COLS = [
    "dist_km",
    "log_dist",
    "eta_hours",
    "area_over_dist2",
    "radial_speed_over_dist",
    "advance_ratio",
    "dist_per_sqrt_area",
    "rel_closing_speed",
    "threat_score",
    "growth_rate_norm",
    "already_close",
    "already_very_close",
    "is_peak_hour",
    "is_high_season",
    "has_dynamics",
    "dynamics_x_closing",
]

# Lista final combinada — unica fonte de verdade para os modelos
FEATURE_COLS = ORIGINAL_COLS + ENGINEERED_COLS
