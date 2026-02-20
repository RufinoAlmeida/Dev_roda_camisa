"""
models/calibration.py
=====================
Calibracao isotonica das probabilidades do ensemble via cross-validation OOF.

Por que calibrar?
  O ensemble pode produzir probabilidades sistematicamente enviesadas
  (ex: concentradas em 0.3-0.7 quando a realidade e bimodal).
  A regressao isotonica aprende um mapeamento monotono nao-parametrico
  diretamente sobre predicoes out-of-fold honestas.

Por que OOF e nao o conjunto de treino completo?
  Treinar e calibrar no mesmo conjunto superajusta o calibrador aos dados.
  OOF garante que cada predicao usada no ajuste foi feita por um modelo
  que nunca viu aquela amostra.
"""

from typing import Callable
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression

from config.settings import (
    GBM_PARAMS, RF_PARAMS, LR_PARAMS,
    ENSEMBLE_WEIGHTS, CV_N_SPLITS, CV_SHUFFLE, CV_SEED,
)
from survival.horizon import make_horizon_data


def _identity_calibrator() -> IsotonicRegression:
    """Calibrador identidade: passa as probabilidades sem alteracao."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    return iso


def calibrate_with_cv(
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    horizon: int,
    G_fn: Callable[[float], float],
) -> IsotonicRegression:
    """
    Gera predicoes OOF do ensemble completo e ajusta regressao isotonica.

    Parameters
    ----------
    X       : features de treino (numpy array)
    times   : tempos de sobrevivencia
    events  : indicadores de evento
    horizon : horizonte em horas
    G_fn    : funcao de censura G(t)

    Returns
    -------
    IsotonicRegression ajustado.
    """
    labels, weights, mask = make_horizon_data(times, events, horizon, G_fn)
    X_h = X[mask]
    y_h = labels[mask]
    w_h = weights[mask]
    w_h = w_h / w_h.mean()

    if len(np.unique(y_h)) < 2:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit([0.0, 1.0], [float(y_h[0]), float(y_h[0])])
        return iso

    skf       = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=CV_SHUFFLE, random_state=CV_SEED)
    oof_preds = np.zeros(len(y_h))

    w_gbm = ENSEMBLE_WEIGHTS["gbm"]
    w_rf  = ENSEMBLE_WEIGHTS["rf"]
    w_lr  = ENSEMBLE_WEIGHTS["lr"]

    for tr_idx, val_idx in skf.split(X_h, y_h):
        X_tr, X_val = X_h[tr_idx], X_h[val_idx]
        y_tr, w_tr  = y_h[tr_idx], w_h[tr_idx]

        if len(np.unique(y_tr)) < 2:
            oof_preds[val_idx] = float(y_tr.mean())
            continue

        gbm_f    = GradientBoostingClassifier(**GBM_PARAMS)
        rf_f     = RandomForestClassifier(**RF_PARAMS)
        scaler_f = StandardScaler()
        X_tr_sc  = scaler_f.fit_transform(X_tr)
        lr_f     = LogisticRegression(**LR_PARAMS)

        gbm_f.fit(X_tr,    y_tr, sample_weight=w_tr)
        rf_f.fit(X_tr,     y_tr, sample_weight=w_tr)
        lr_f.fit(X_tr_sc,  y_tr, sample_weight=w_tr)

        X_val_sc = scaler_f.transform(X_val)
        oof_preds[val_idx] = (
            w_gbm * gbm_f.predict_proba(X_val)[:, 1]
            + w_rf  * rf_f.predict_proba(X_val)[:, 1]
            + w_lr  * lr_f.predict_proba(X_val_sc)[:, 1]
        )

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_preds, y_h, sample_weight=w_h)
    return iso


def build_all_calibrators(
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    horizons: list[int],
    G_fn: Callable[[float], float],
    models: dict,
) -> dict:
    """
    Constroi um calibrador por horizonte.

    Horizontes degenerados ou extrapolados recebem um calibrador identidade.

    Returns
    -------
    calibrators : dict { horizonte -> IsotonicRegression }
    """
    print("\nPASSO 7: Calibrando probabilidades com validacao cruzada...")
    calibrators: dict = {}

    for h in horizons:
        labels, weights, mask = make_horizon_data(times, events, h, G_fn)
        only_one_class  = len(np.unique(labels[mask])) < 2
        is_extrapolated = models[h]["gbm"] is None and models[h]["predict"] is not None

        if only_one_class or is_extrapolated:
            calibrators[h] = _identity_calibrator()
            motivo = "unica classe" if only_one_class else "horizonte extrapolado"
            print(f"  >> Calibrador {h}h: identidade ({motivo})")
            continue

        calibrators[h] = calibrate_with_cv(X, times, events, h, G_fn)
        print(f"  [OK] Calibrador {h}h ajustado")

    return calibrators
