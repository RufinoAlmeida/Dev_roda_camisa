"""
survival/ipcw.py
================
Implementacao dos pesos IPCW (Inverse Probability of Censoring Weights).

G(t) = P(C > t) — probabilidade de nao ser censurado ate o instante t.
Peso IPCW = 1/G(t) — compensa observacoes que tiveram oportunidade menor
de ser observadas, corrigindo o vies introduzido pela censura diferencial.
"""

from typing import Callable
import numpy as np
import pandas as pd


def compute_ipcw(times: np.ndarray, events: np.ndarray) -> Callable[[float], float]:
    """
    Estima G(t) = P(C > t) via Kaplan-Meier com a censura tratada como evento.

    Parameters
    ----------
    times  : tempos de ocorrencia ou censura
    events : indicador de evento real (1 = evento, 0 = censurado)

    Returns
    -------
    G_at : funcao G_at(h) -> float
        Retorna G(h), com piso em 1e-3 para evitar divisao por zero.
    """
    censored = 1 - events  # inverte: a censura se torna o "evento" a modelar

    df = pd.DataFrame({"t": times, "e": censored}).sort_values("t")
    unique_times = sorted(df[df["e"] == 1]["t"].unique())

    G = 1.0
    survival_G: dict[float, float] = {}

    for t in unique_times:
        at_risk     = len(df[df["t"] >= t])
        events_at_t = len(df[(df["t"] == t) & (df["e"] == 1)])
        if at_risk > 0:
            G *= 1 - events_at_t / at_risk
        survival_G[t] = G

    sorted_G = sorted(survival_G.items())

    def G_at(h: float) -> float:
        prev = 1.0
        for t, g in sorted_G:
            if t > h:
                break
            prev = g
        return max(prev, 1e-3)  # piso evita peso infinito

    return G_at
