"""
evaluation/metrics.py
=====================
Implementacoes das metricas da competicao WiDS:
  - C-index de Harrell (concordancia no ranking de risco)
  - Brier Score ponderado por IPCW para dados censurados
  - Hybrid Score = 0.3 * C-index + 0.7 * (1 - Weighted Brier)
"""

from typing import Callable
import numpy as np

from config.settings import BRIER_WEIGHTS
from survival.horizon import make_horizon_data


def concordance_index(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """
    Calcula o C-index de Harrell.

    Um par (i, j) e permissivel se i teve evento observado em t_i < t_j.
    O par e concordante se risk_score[i] > risk_score[j].

    C-index = (concordantes + 0.5 * empates) / permissiveis

    Complexidade: O(n^2) — adequado para n~200 amostras.
    """
    n           = len(times)
    concordant  = 0
    discordant  = 0
    tied_risk   = 0
    permissible = 0

    for i in range(n):
        for j in range(i + 1, n):
            if events[i] == 0 and events[j] == 0:
                continue
            if events[i] == 1 and times[i] < times[j]:
                permissible += 1
                if   risk_scores[i] > risk_scores[j]: concordant += 1
                elif risk_scores[i] < risk_scores[j]: discordant += 1
                else:                                  tied_risk  += 1
            elif events[j] == 1 and times[j] < times[i]:
                permissible += 1
                if   risk_scores[j] > risk_scores[i]: concordant += 1
                elif risk_scores[j] < risk_scores[i]: discordant += 1
                else:                                  tied_risk  += 1

    if permissible == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / permissible


def brier_score_censored(
    times: np.ndarray,
    events: np.ndarray,
    probs: np.ndarray,
    horizon: int,
    G_fn: Callable[[float], float],
) -> float:
    """
    Brier Score ponderado por IPCW para dados censurados em um horizonte.

    BS = mean_w[ (y - p)^2 ]  com pesos IPCW normalizados.
    """
    labels, weights, mask = make_horizon_data(times, events, horizon, G_fn)
    y_h = labels[mask]
    p_h = probs[mask]
    w_h = weights[mask]
    w_h = w_h / w_h.mean()
    return float(np.average((y_h - p_h) ** 2, weights=w_h))


def hybrid_score(c_idx: float, weighted_brier: float) -> float:
    """Metrica composta da competicao: 0.3 * C-index + 0.7 * (1 - Brier)."""
    return 0.3 * c_idx + 0.7 * (1.0 - weighted_brier)


def evaluate_on_train(
    X_train: np.ndarray,
    y_time: np.ndarray,
    y_event: np.ndarray,
    horizons: list[int],
    models: dict,
    G_fn: Callable[[float], float],
) -> None:
    """
    Calcula e imprime todas as metricas sobre o conjunto de treino.

    Usa prob_48h como risk_score para o C-index (maior peso na metrica).
    """
    print("\nPASSO 11: Avaliacao no conjunto de treino...")

    train_preds = {h: models[h]["predict"](X_train) for h in horizons}

    c_idx = concordance_index(y_time, y_event, train_preds[48])
    print(f"  C-index (treino): {c_idx:.4f}")

    weighted_brier = 0.0
    for h in horizons:
        bs = brier_score_censored(y_time, y_event, train_preds[h], h, G_fn)
        w  = BRIER_WEIGHTS.get(h, 0.0)
        weighted_brier += w * bs
        print(f"  Brier Score @{h}h (treino): {bs:.4f}  [peso={w}]")

    hs = hybrid_score(c_idx, weighted_brier)
    print(f"\n  >> Hybrid Score estimado (treino): {hs:.4f}")
    print(f"     (referencia: 0.5 = aleatorio | 1.0 = perfeito)")
