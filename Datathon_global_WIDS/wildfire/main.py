"""
main.py
=======
Ponto de entrada do pipeline WiDS Wildfire.

Orquestra os modulos em ordem sem conter logica de negocio propria.
Para alterar comportamento, edite os modulos correspondentes ou config/settings.py.

Uso:
    python main.py
    (train.csv e test.csv devem estar na mesma pasta que main.py)
"""

import warnings
import sys

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Compatibilidade de encoding no terminal Windows (cp1252)
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Imports dos modulos do projeto
# ---------------------------------------------------------------------------
from config.settings import (
    TRAIN_PATH, TEST_PATH, OUTPUT_PATH, HORIZONS,
)
from data.loader import load_data, report_summary
from features.builder import prepare_matrices
from features.columns import FEATURE_COLS
from survival.kaplan_meier import kaplan_meier, print_km_summary
from survival.ipcw import compute_ipcw
from survival.horizon import print_horizon_summary
from models.ensemble import build_all_models
from models.extrapolation import fill_degenerate_horizons
from models.calibration import build_all_calibrators
from postprocessing.monotonicity import enforce_monotonicity, build_submission, validate_submission
from evaluation.importance import print_feature_importance
from evaluation.metrics import evaluate_on_train


def main() -> None:
    # ── 1. CARREGAR DADOS ───────────────────────────────────────────────────
    train, test = load_data(TRAIN_PATH, TEST_PATH)
    report_summary(train, test)

    # ── 2. ENGENHARIA DE FEATURES ───────────────────────────────────────────
    print("\nPASSO 2: Engenharia de features...")
    X_train, X_test = prepare_matrices(train, test)
    print(f"Features utilizadas: {len(FEATURE_COLS)}")

    y_time  = train["time_to_hit_hours"].values
    y_event = train["event"].values

    # ── 3. KAPLAN-MEIER ─────────────────────────────────────────────────────
    km_fn = kaplan_meier(y_time, y_event)
    print_km_summary(km_fn, HORIZONS)

    # ── 4. IPCW ─────────────────────────────────────────────────────────────
    print("\nPASSO 4: Calculando pesos IPCW...")
    G_fn = compute_ipcw(y_time, y_event)

    # ── 5. ROTULOS POR HORIZONTE ─────────────────────────────────────────────
    print_horizon_summary(y_time, y_event, HORIZONS, G_fn)

    # ── 6. TREINAMENTO ──────────────────────────────────────────────────────
    models = build_all_models(X_train.values, y_time, y_event, HORIZONS, G_fn)

    # ── 6b. EXTRAPOLACAO HORIZONTES DEGENERADOS ──────────────────────────────
    models = fill_degenerate_horizons(models, HORIZONS, km_fn)

    # ── 7. CALIBRACAO ───────────────────────────────────────────────────────
    calibrators = build_all_calibrators(
        X_train.values, y_time, y_event, HORIZONS, G_fn, models
    )

    # ── 8. PREDICAO NO TESTE ─────────────────────────────────────────────────
    print("\nPASSO 8: Gerando predicoes no conjunto de teste...")
    calibrated_preds = {}
    for h in HORIZONS:
        raw                = models[h]["predict"](X_test.values)
        calibrated_preds[h] = calibrators[h].predict(raw)

    # ── 9. MONOTONICIDADE ────────────────────────────────────────────────────
    print("\nPASSO 9: Garantindo monotonicidade das probabilidades...")
    final_preds = enforce_monotonicity(calibrated_preds, HORIZONS)

    # ── 10. IMPORTANCIA DE FEATURES ──────────────────────────────────────────
    print_feature_importance(models)

    # ── 11. AVALIACAO NO TREINO ──────────────────────────────────────────────
    evaluate_on_train(X_train.values, y_time, y_event, HORIZONS, models, G_fn)

    # ── 12. SUBMISSAO ────────────────────────────────────────────────────────
    print("\nPASSO 12: Gerando arquivo de submissao...")
    submission = build_submission(test["event_id"], final_preds, HORIZONS)

    is_valid = validate_submission(submission, test["event_id"])

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Submissao salva em: {OUTPUT_PATH}")
    print(f"     {len(submission)} incendios | {len(submission.columns)} colunas")
    print("\nPrimeiras 10 predicoes:")
    print(submission.head(10).to_string(index=False))

    if not is_valid:
        raise RuntimeError("Arquivo de submissao com problemas — verifique os erros acima.")


if __name__ == "__main__":
    main()
