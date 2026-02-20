"""
models/ensemble.py
==================
Treina um ensemble GBM + Random Forest + Logistic Regression para um horizonte.

O ensemble heterogeneo combina tres perspectivas complementares:
  - GBM   (50%): captura nao-linearidades e interacoes entre features
  - RF    (30%): reduz overfitting por meio de bagging e aleatoriedade
  - LR    (20%): ancora regularizada; estavel mesmo com poucos dados
"""

from typing import Callable
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config.settings import GBM_PARAMS, RF_PARAMS, LR_PARAMS, ENSEMBLE_WEIGHTS
from survival.horizon import make_horizon_data


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    """Normaliza pesos IPCW para media = 1, evitando gradientes gigantes."""
    mean = w.mean()
    return w / mean if mean > 0 else w


def train_horizon_model(
    X: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> tuple[Callable, object, object, object, object]:
    """
    Treina o ensemble para um horizonte especifico.

    Parameters
    ----------
    X       : matriz de features (numpy array)
    labels  : rotulos binarios produzidos por make_horizon_data()
    weights : pesos IPCW produzidos por make_horizon_data()
    mask    : mascara de amostras incluidas no treino

    Returns
    -------
    (predict_fn, gbm, rf, scaler, lr)
        predict_fn e a funcao de ensemble que retorna probabilidades.
        Os objetos treinados sao retornados para permitir feature importance etc.
        Em caso degenerado (1 so classe), todos os objetos exceto predict_fn sao None.
    """
    X_h = X[mask]
    y_h = labels[mask]
    w_h = _normalize_weights(weights[mask])

    # Caso degenerado: apenas uma classe nos dados validos
    if len(np.unique(y_h)) < 2:
        const_val = float(y_h[0])

        def predict_constant(X_new: np.ndarray) -> np.ndarray:
            return np.full(len(X_new), const_val)

        return predict_constant, None, None, None, None

    # --- Treinamento dos tres modelos ------------------------------------
    gbm = GradientBoostingClassifier(**GBM_PARAMS)
    rf  = RandomForestClassifier(**RF_PARAMS)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_h)
    lr       = LogisticRegression(**LR_PARAMS)

    gbm.fit(X_h,     y_h, sample_weight=w_h)
    rf.fit(X_h,      y_h, sample_weight=w_h)
    lr.fit(X_scaled, y_h, sample_weight=w_h)

    w_gbm = ENSEMBLE_WEIGHTS["gbm"]
    w_rf  = ENSEMBLE_WEIGHTS["rf"]
    w_lr  = ENSEMBLE_WEIGHTS["lr"]

    def predict_proba_ensemble(X_new: np.ndarray) -> np.ndarray:
        p_gbm = gbm.predict_proba(X_new)[:, 1]
        p_rf  = rf.predict_proba(X_new)[:, 1]
        p_lr  = lr.predict_proba(scaler.transform(X_new))[:, 1]
        return w_gbm * p_gbm + w_rf * p_rf + w_lr * p_lr

    return predict_proba_ensemble, gbm, rf, scaler, lr


def build_all_models(
    X: np.ndarray,
    y_time: np.ndarray,
    y_event: np.ndarray,
    horizons: list[int],
    G_fn,
) -> dict:
    """
    Treina um modelo por horizonte e retorna o dicionario de modelos.

    Parameters
    ----------
    X        : features de treino (numpy array)
    y_time   : tempos de sobrevivencia
    y_event  : indicadores de evento
    horizons : lista de horizontes em horas
    G_fn     : funcao de censura G(t)

    Returns
    -------
    models : dict { horizonte -> {'predict', 'gbm', 'rf', 'scaler', 'lr'} }
    """
    print("\nPASSO 6: Treinando modelos por horizonte...")
    models: dict = {}

    for h in horizons:
        labels, weights, mask = make_horizon_data(y_time, y_event, h, G_fn)

        if len(np.unique(labels[mask])) < 2:
            print(f"  [!] Horizonte {h}h: apenas 1 classe — sera extrapolado")
            models[h] = {"predict": None, "gbm": None, "rf": None, "scaler": None, "lr": None}
            continue

        predict_fn, gbm, rf, scaler, lr = train_horizon_model(X, labels, weights, mask)
        models[h] = {
            "predict": predict_fn,
            "gbm": gbm,
            "rf": rf,
            "scaler": scaler,
            "lr": lr,
        }
        print(f"  [OK] Modelo {h}h treinado")

    return models
