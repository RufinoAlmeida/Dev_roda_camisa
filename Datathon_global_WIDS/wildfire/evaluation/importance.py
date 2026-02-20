"""
evaluation/importance.py
========================
Relatorio de importancia de features a partir do GBM treinado.

O GBM do horizonte 48h e usado como referencia pois recebe o maior peso
na metrica Brier da competicao (0.4 vs 0.3 dos demais).
"""

import pandas as pd
from features.columns import FEATURE_COLS


def print_feature_importance(
    models: dict,
    top_n: int = 15,
    reference_horizon: int = 48,
) -> None:
    """
    Imprime as top_n features mais importantes do GBM de reference_horizon.

    Parameters
    ----------
    models            : dicionario de modelos por horizonte
    top_n             : numero de features a exibir
    reference_horizon : horizonte de referencia (default 48h)
    """
    print(f"\nPASSO 10: Top {top_n} features mais importantes (horizonte {reference_horizon}h):")

    gbm = models[reference_horizon]["gbm"]

    if gbm is None:
        print(f"  (modelo {reference_horizon}h extrapolado — importancia nao disponivel)")
        return

    feat_imp = (
        pd.Series(gbm.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
    )

    for feat, imp in feat_imp.head(top_n).items():
        bar = "#" * int(imp * 200)
        print(f"  {feat:<35} {imp:.4f} {bar}")
