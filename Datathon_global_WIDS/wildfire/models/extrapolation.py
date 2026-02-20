"""
models/extrapolation.py
=======================
Extrapola predicoes para horizontes degenerados (ex: 72h com 100% positivos)
usando a razao entre probabilidades marginais do Kaplan-Meier.

Logica: se KM diz P(hit by 72h) / P(hit by 48h) = 1.76, aplica esse fator
sobre as predicoes individualizadas do horizonte anterior.
"""

from typing import Callable
import numpy as np


def make_extrapolator(
    base_predict_fn: Callable[[np.ndarray], np.ndarray],
    ratio: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Cria uma funcao de predicao que escala as saidas de base_predict_fn pelo ratio KM.

    Parameters
    ----------
    base_predict_fn : funcao predict do horizonte anterior valido
    ratio           : km_prob[h] / km_prob[prev_h]

    Returns
    -------
    Callable que retorna probabilidades clampadas em [0, 1].
    """
    def extrap(X_new: np.ndarray) -> np.ndarray:
        return np.clip(base_predict_fn(X_new) * ratio, 0.0, 1.0)

    return extrap


def fill_degenerate_horizons(
    models: dict,
    horizons: list[int],
    km_fn: Callable[[float], float],
) -> dict:
    """
    Para cada horizonte sem modelo proprio (predict=None), busca o horizonte
    valido imediatamente anterior e cria um extrapolador pela razao KM.

    Parameters
    ----------
    models   : dicionario construido por build_all_models()
    horizons : lista ordenada de horizontes
    km_fn    : funcao de sobrevivencia Kaplan-Meier

    Returns
    -------
    models atualizado (in-place e retornado para facilitar encadeamento).
    """
    km_probs = {h: 1.0 - km_fn(h) for h in horizons}
    print(f"\n  KM probs: { {h: f'{p:.3f}' for h, p in km_probs.items()} }")

    for h in horizons:
        if models[h]["predict"] is not None:
            continue

        prev_h = max(
            (ph for ph in horizons if ph < h and models[ph]["predict"] is not None),
            default=None,
        )

        if prev_h is None:
            print(f"  [!] Horizonte {h}h sem modelo e sem horizonte anterior valido — ignorado")
            continue

        ratio = km_probs[h] / max(km_probs[prev_h], 1e-6)
        print(f"  >> Horizonte {h}h extrapolado de {prev_h}h (ratio KM = {ratio:.3f})")

        models[h] = {
            "predict": make_extrapolator(models[prev_h]["predict"], ratio),
            "gbm": None,
            "rf":  None,
            "scaler": None,
            "lr":  None,
        }

    return models
