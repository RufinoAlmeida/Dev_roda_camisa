"""
survival/kaplan_meier.py
========================
Implementacao do estimador de Kaplan-Meier para a funcao de sobrevivencia S(t).

S(t) = P(T > t) — probabilidade de o evento ainda nao ter ocorrido em t.
"""

from typing import Callable
import numpy as np
import pandas as pd


def kaplan_meier(times: np.ndarray, events: np.ndarray) -> Callable[[float], float]:
    """
    Estima S(t) = P(T > t) pelo metodo de Kaplan-Meier.

    Parameters
    ----------
    times  : tempos de ocorrencia ou censura (array 1-D)
    events : indicador de evento (1 = evento, 0 = censurado)

    Returns
    -------
    S_at : funcao S_at(h) -> float
        Retorna a probabilidade de sobrevivencia no instante h.
    """
    df = pd.DataFrame({"t": times, "e": events}).sort_values("t")
    unique_times = sorted(df[df["e"] == 1]["t"].unique())

    S = 1.0
    survival: dict[float, float] = {}

    for t in unique_times:
        at_risk     = len(df[df["t"] >= t])
        events_at_t = len(df[(df["t"] == t) & (df["e"] == 1)])
        S *= 1 - events_at_t / at_risk
        survival[t] = S

    # Pre-ordena uma vez; chamadas subsequentes a S_at sao O(k) no nr de eventos
    sorted_survival = sorted(survival.items())

    def S_at(h: float) -> float:
        """Retorna S(h) = P(T > h) usando o estimador de KM."""
        prev = 1.0
        for t, s in sorted_survival:
            if t > h:
                break
            prev = s
        return prev

    return S_at


def print_km_summary(km_fn: Callable[[float], float], horizons: list[int]) -> None:
    """Imprime a curva de sobrevivencia para cada horizonte."""
    print("\nPASSO 3: Calculando curva de Kaplan-Meier baseline...")
    for h in horizons:
        surv = km_fn(h)
        print(f"  KM S({h}h) = {surv:.3f}  ->  P(hit by {h}h) = {1 - surv:.3f}")
