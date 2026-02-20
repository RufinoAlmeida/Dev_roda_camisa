"""
features/builder.py
===================
Toda a logica de engenharia de features em um unico lugar.
Cada transformacao e documentada com sua justificativa fisica.
"""

import numpy as np
import pandas as pd
from config.settings import (
    EPS,
    ETA_CLIP_UPPER,
    AREA_DIST2_CLIP,
    RADIAL_DIST_CLIP,
    THREAT_CLIP,
    GROWTH_NORM_CLIP,
    ADVANCE_CLIP,
    REL_SPEED_CLIP,
    CLOSE_THRESHOLD,
    VERY_CLOSE_THRESHOLD,
    PEAK_HOUR_START,
    PEAK_HOUR_END,
    HIGH_SEASON_MONTHS,
)
from features.columns import FEATURE_COLS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica engenharia de features ao DataFrame e retorna uma copia enriquecida.

    Nao modifica o DataFrame original (opera sobre df.copy()).

    Parameters
    ----------
    df : pd.DataFrame — dados brutos (treino ou teste)

    Returns
    -------
    pd.DataFrame — com colunas originais + features engenhadas
    """
    X = df.copy()

    # --- Features de distancia -------------------------------------------

    # Distancia em km (escala menor facilita alguns otimizadores)
    X["dist_km"] = X["dist_min_ci_0_5h"] / 1000.0

    # Log da distancia: comprime a cauda longa (range de 307m a 757km)
    X["log_dist"] = np.log1p(X["dist_min_ci_0_5h"])

    # ETA: so faz sentido fisico quando o fogo esta se aproximando (speed > 0)
    # Velocidade negativa => fogo se afastando => ETA indefinido => clip(lower=0)
    closing_positive = X["closing_speed_m_per_h"].clip(lower=0) + EPS
    X["eta_hours"] = (X["dist_min_ci_0_5h"] / closing_positive).clip(upper=ETA_CLIP_UPPER)

    # Razao area/dist^2: inspirada na lei do inverso do quadrado
    # Fogo grande e proximo e exponencialmente mais perigoso que fogo grande longe
    X["area_over_dist2"] = (
        X["area_first_ha"] / (X["dist_min_ci_0_5h"] ** 2 + EPS)
    ).clip(upper=AREA_DIST2_CLIP)

    # --- Features de dinamica --------------------------------------------

    # Velocidade de expansao radial relativa a distancia
    X["radial_speed_over_dist"] = (
        X["radial_growth_rate_m_per_h"] / (X["dist_min_ci_0_5h"] + EPS)
    ).clip(upper=RADIAL_DIST_CLIP)

    # Avanco projetado como fracao da distancia total
    X["advance_ratio"] = (
        X["projected_advance_m"] / (X["dist_min_ci_0_5h"] + EPS)
    ).clip(-ADVANCE_CLIP, ADVANCE_CLIP)

    # Distancia normalizada pelo raio efetivo do fogo (sqrt da area)
    X["dist_per_sqrt_area"] = X["dist_min_ci_0_5h"] / (
        np.sqrt(X["area_first_ha"]) + EPS
    )

    # Velocidade de fechamento relativa (proporcional ao risco imediato)
    X["rel_closing_speed"] = (
        X["closing_speed_m_per_h"] / (X["dist_min_ci_0_5h"] + EPS)
    ).clip(-REL_SPEED_CLIP, REL_SPEED_CLIP)

    # Score de ameaca: alinhamento * velocidade absoluta / distancia
    X["threat_score"] = (
        X["alignment_abs"] * X["closing_speed_abs_m_per_h"] / (X["dist_min_ci_0_5h"] + EPS)
    ).clip(upper=THREAT_CLIP)

    # Taxa de crescimento relativa (fogo que dobra de area por hora e critico)
    X["growth_rate_norm"] = (
        X["area_growth_rate_ha_per_h"] / (X["area_first_ha"] + EPS)
    ).clip(upper=GROWTH_NORM_CLIP)

    # --- Indicadores binarios de proximidade -----------------------------

    X["already_close"]      = (X["dist_min_ci_0_5h"] < CLOSE_THRESHOLD).astype(int)
    X["already_very_close"] = (X["dist_min_ci_0_5h"] < VERY_CLOSE_THRESHOLD).astype(int)

    # --- Features temporais ----------------------------------------------

    # Horario de pico: maior temperatura, menor umidade e ventos mais fortes
    X["is_peak_hour"] = (
        (X["event_start_hour"] >= PEAK_HOUR_START)
        & (X["event_start_hour"] <= PEAK_HOUR_END)
    ).astype(int)

    # Alta temporada de incendios
    X["is_high_season"] = X["event_start_month"].isin(HIGH_SEASON_MONTHS).astype(int)

    # --- Features de qualidade dos dados ---------------------------------

    # Dados dinamicos ricos disponíveis (multiplos perimetros em 5h)
    X["has_dynamics"] = (1 - X["low_temporal_resolution_0_5h"]).astype(int)

    # Interacao: velocidade de fechamento e confiavel apenas com dados ricos
    X["dynamics_x_closing"] = X["has_dynamics"] * X["closing_speed_m_per_h"]

    return X


def prepare_matrices(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica build_features e seleciona FEATURE_COLS para treino e teste.
    Preenche NaN e infinitos com zero.

    Returns
    -------
    (X_train, X_test) : tuple[pd.DataFrame, pd.DataFrame]
    """
    train_fe = build_features(train)
    test_fe  = build_features(test)

    X_train = train_fe[FEATURE_COLS].fillna(0).replace([float("inf"), float("-inf")], 0)
    X_test  = test_fe[FEATURE_COLS].fillna(0).replace([float("inf"), float("-inf")], 0)

    return X_train, X_test
