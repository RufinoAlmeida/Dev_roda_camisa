"""
survival/horizon.py
===================
Converte dados de sobrevivencia em problemas de classificacao binaria
para cada horizonte temporal, aplicando pesos IPCW.

Regras de rotulagem para um horizonte H:
  evento=1 e t<=H  -> label=1, peso=1/G(t)    fogo atingiu dentro do prazo
  evento=1 e t>H   -> label=0, peso=1/G(H)    fogo atingiu, mas apos o prazo
  evento=0 e t>=H  -> label=0, peso=1/G(H)    nao atingiu, observado ate H
  evento=0 e t<H   -> EXCLUIDO                censura ambigua antes do prazo
"""

from typing import Callable
import numpy as np


def make_horizon_data(
    times: np.ndarray,
    events: np.ndarray,
    horizon: int,
    G_fn: Callable[[float], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constroi labels, pesos e mascara de inclusao para um horizonte.

    Parameters
    ----------
    times   : tempos de evento ou censura
    events  : indicador de evento (1/0)
    horizon : horizonte em horas
    G_fn    : funcao de censura G(t) estimada por compute_ipcw()

    Returns
    -------
    labels  : np.ndarray[int]   — rotulos binarios (0 ou 1)
    weights : np.ndarray[float] — pesos IPCW
    mask    : np.ndarray[bool]  — True para amostras incluidas no treino
    """
    labels:  list[int]   = []
    weights: list[float] = []
    mask:    list[bool]  = []

    for t, e in zip(times, events):
        if e == 1 and t <= horizon:          # evento dentro do prazo
            labels.append(1)
            weights.append(1.0 / G_fn(t))
            mask.append(True)
        elif e == 1 and t > horizon:         # evento apos o prazo
            labels.append(0)
            weights.append(1.0 / G_fn(horizon))
            mask.append(True)
        elif e == 0 and t >= horizon:        # censurado apos o horizonte
            labels.append(0)
            weights.append(1.0 / G_fn(horizon))
            mask.append(True)
        else:                                # censurado antes do horizonte — exclui
            labels.append(0)
            weights.append(0.0)
            mask.append(False)

    return np.array(labels), np.array(weights), np.array(mask)


def print_horizon_summary(
    times: np.ndarray,
    events: np.ndarray,
    horizons: list[int],
    G_fn: Callable[[float], float],
) -> None:
    """Imprime estatisticas de rotulagem para cada horizonte."""
    print("\nPASSO 5: Criando rotulos e pesos por horizonte...")
    for h in horizons:
        labels, _, mask = make_horizon_data(times, events, h, G_fn)
        n_used = mask.sum()
        n_pos  = labels[mask].sum()
        print(
            f"  Horizonte {h:2d}h: {n_used} obs usadas | "
            f"{n_pos} positivos ({n_pos / n_used * 100:.1f}%)"
        )
