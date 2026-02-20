"""
config/settings.py
==================
Unico ponto de verdade para todas as constantes do projeto.
Para alterar parametros, hiperparametros ou caminhos, edite apenas este arquivo.
"""

import os
import sys

# ---------------------------------------------------------------------------
# ENCODING — compatibilidade com terminais Windows (cp1252)
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# CAMINHOS
# Ao executar main.py, os CSVs devem estar na raiz do projeto (mesma pasta).
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH  = os.path.join(BASE_DIR, "train.csv")
TEST_PATH   = os.path.join(BASE_DIR, "test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "submission.csv")

# ---------------------------------------------------------------------------
# HORIZONTES TEMPORAIS (horas)
# ---------------------------------------------------------------------------
HORIZONS = [12, 24, 48, 72]

# ---------------------------------------------------------------------------
# HIPERPARAMETROS — GRADIENT BOOSTING
# ---------------------------------------------------------------------------
GBM_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=3,
    random_state=42,
)

# ---------------------------------------------------------------------------
# HIPERPARAMETROS — RANDOM FOREST
# ---------------------------------------------------------------------------
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

# ---------------------------------------------------------------------------
# HIPERPARAMETROS — LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
LR_PARAMS = dict(
    C=0.1,
    max_iter=1000,
    random_state=42,
)

# ---------------------------------------------------------------------------
# PESOS DO ENSEMBLE (devem somar 1.0)
# ---------------------------------------------------------------------------
ENSEMBLE_WEIGHTS = dict(gbm=0.50, rf=0.30, lr=0.20)

# ---------------------------------------------------------------------------
# CALIBRACAO — CROSS-VALIDATION
# ---------------------------------------------------------------------------
CV_N_SPLITS  = 5
CV_SHUFFLE   = True
CV_SEED      = 42

# ---------------------------------------------------------------------------
# SUBMISSAO
# ---------------------------------------------------------------------------
PROB_CLIP_MIN = 0.01   # evita certeza absoluta de 0
PROB_CLIP_MAX = 0.99   # evita certeza absoluta de 1

# ---------------------------------------------------------------------------
# METRICAS — pesos do Weighted Brier Score da competicao
# ---------------------------------------------------------------------------
BRIER_WEIGHTS = {24: 0.3, 48: 0.4, 72: 0.3}

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING — constantes fisicas
# ---------------------------------------------------------------------------
EPS               = 1e-6    # epsilon para evitar divisao por zero
ETA_CLIP_UPPER    = 500     # horas maximas de ETA
AREA_DIST2_CLIP   = 1e6
RADIAL_DIST_CLIP  = 1e4
THREAT_CLIP       = 1e4
GROWTH_NORM_CLIP  = 100
ADVANCE_CLIP      = 10      # valores absolutos
REL_SPEED_CLIP    = 1

CLOSE_THRESHOLD      = 5000   # metros — fogo "proximo"
VERY_CLOSE_THRESHOLD = 2000   # metros — fogo "muito proximo"
PEAK_HOUR_START      = 12     # hora de inicio do pico de risco
PEAK_HOUR_END        = 18     # hora de fim do pico de risco
HIGH_SEASON_MONTHS   = [6, 7, 8]  # junho, julho, agosto
