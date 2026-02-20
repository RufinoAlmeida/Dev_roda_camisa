"""
postprocessing/monotonicity.py
===============================
Garante que as probabilidades sejam monotonamente crescentes nos horizontes.

Justificativa: P(atingir zona ate t) e uma CDF — nunca pode decrescer.
Modelos treinados independentemente por horizonte podem violar levemente
essa restricao. A correcao e feita por acumulacao progressiva (O(n*k)).
"""

import numpy as np
import pandas as pd

from config.settings import PROB_CLIP_MIN, PROB_CLIP_MAX


def enforce_monotonicity(
    preds_dict: dict,
    horizons: list[int],
) -> dict:
    """
    Garante prob_12h <= prob_24h <= prob_48h <= prob_72h para cada amostra.

    Parameters
    ----------
    preds_dict : { horizonte -> np.ndarray de probabilidades }
    horizons   : lista ordenada de horizontes

    Returns
    -------
    dict com as mesmas chaves, probabilidades ajustadas.
    """
    n      = len(preds_dict[horizons[0]])
    result = {h: np.zeros(n) for h in horizons}

    for i in range(n):
        vals = [preds_dict[h][i] for h in horizons]
        for j in range(1, len(vals)):
            if vals[j] < vals[j - 1]:
                vals[j] = vals[j - 1]
        for j, h in enumerate(horizons):
            result[h][i] = vals[j]

    return result


def build_submission(
    event_ids: pd.Series,
    final_preds: dict,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Monta o DataFrame final de submissao com probabilidades clipadas em [0.01, 0.99].

    Parameters
    ----------
    event_ids   : coluna de IDs do conjunto de teste
    final_preds : saida de enforce_monotonicity()
    horizons    : lista de horizontes

    Returns
    -------
    pd.DataFrame pronto para .to_csv()
    """
    data = {"event_id": event_ids}
    for h in horizons:
        data[f"prob_{h}h"] = final_preds[h].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    return pd.DataFrame(data)


def validate_submission(submission: pd.DataFrame, test_ids: pd.Series) -> bool:
    """
    Executa verificacoes de sanidade no DataFrame de submissao.

    Retorna True se todas as verificacoes passarem; False caso contrario.
    """
    prob_cols = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]
    ok        = True

    checks = {
        "Colunas corretas":
            list(submission.columns) == ["event_id"] + prob_cols,
        "Numero de linhas correto":
            len(submission) == len(test_ids),
        "Probabilidades em [0, 1]":
            submission[prob_cols].min().min() >= 0
            and submission[prob_cols].max().max() <= 1,
        "Monotonicidade respeitada": (
            (submission["prob_12h"] <= submission["prob_24h"]).all()
            and (submission["prob_24h"] <= submission["prob_48h"]).all()
            and (submission["prob_48h"] <= submission["prob_72h"]).all()
        ),
        "event_id correspondem":
            submission["event_id"].isin(test_ids).all(),
    }

    print("\nValidacao do arquivo de submissao:")
    for descricao, passou in checks.items():
        status = "[OK]  " if passou else "[ERRO]"
        print(f"  {status} {descricao}")
        if not passou:
            ok = False

    return ok
