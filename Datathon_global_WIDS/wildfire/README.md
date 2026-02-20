# WiDS Datathon 2025 — Wildfire Evacuation Zone Threat Prediction

Previsao da probabilidade de um incendio florestal ameacar uma zona de evacuacao
em 12, 24, 48 e 72 horas usando apenas os primeiros 5h de dados.

**Metrica:** Hybrid Score = 0.3 * C-index + 0.7 * (1 - Weighted Brier Score)

---

## Como executar

```
wildfire/
├── train.csv        <- coloque aqui
├── test.csv         <- coloque aqui
└── main.py          <- execute este arquivo
```

```bash
cd wildfire
python main.py
```

O arquivo `submission.csv` sera gerado na mesma pasta.

---

## Estrutura do projeto

```
wildfire/
│
├── main.py                      # Orquestrador — sem logica de negocio
│
├── config/
│   └── settings.py              # UNICA FONTE DE VERDADE para constantes,
│                                # hiperparametros e caminhos
│
├── data/
│   └── loader.py                # Carregamento e validacao dos CSVs
│
├── features/
│   ├── columns.py               # Lista canonica de FEATURE_COLS
│   └── builder.py               # Toda a engenharia de features
│
├── survival/
│   ├── kaplan_meier.py          # Estimador KM: S(t) = P(T > t)
│   ├── ipcw.py                  # Pesos de censura: G(t) = P(C > t)
│   └── horizon.py               # Rotulos binarios por horizonte com IPCW
│
├── models/
│   ├── ensemble.py              # Treina GBM + RF + LR por horizonte
│   ├── extrapolation.py         # Extrapola horizontes degenerados via KM
│   └── calibration.py           # Calibracao isotonica OOF por horizonte
│
├── postprocessing/
│   └── monotonicity.py          # Garante prob_12h <= ... <= prob_72h
│                                # Gera e valida o DataFrame de submissao
│
└── evaluation/
    ├── metrics.py               # C-index, Brier censurado, Hybrid Score
    └── importance.py            # Feature importance do GBM
```

---

## Como alterar parametros

Todos os hiperparametros e constantes estao em **`config/settings.py`**.
Nao e necessario modificar nenhum outro arquivo para:

| O que mudar                        | Onde mudar              |
|------------------------------------|-------------------------|
| Hiperparametros do GBM/RF/LR       | `GBM_PARAMS`, `RF_PARAMS`, `LR_PARAMS` |
| Pesos do ensemble                  | `ENSEMBLE_WEIGHTS`      |
| Numero de folds na calibracao      | `CV_N_SPLITS`           |
| Limites de clip das probabilidades | `PROB_CLIP_MIN/MAX`     |
| Limiares de distancia (5km/2km)    | `CLOSE_THRESHOLD`       |
| Alta temporada                     | `HIGH_SEASON_MONTHS`    |
| Caminhos de entrada/saida          | `TRAIN_PATH`, `OUTPUT_PATH` |

---

## Como adicionar/remover features

1. Edite `features/builder.py` — adicione a nova transformacao em `build_features()`
2. Edite `features/columns.py` — adicione o nome da coluna em `ENGINEERED_COLS`

Nenhum outro arquivo precisa ser alterado.

---

## Abordagem

- **Analise de sobrevivencia:** converte o problema em classificacao binaria por horizonte
- **IPCW:** corrige o vies de censura diferencial via pesos 1/G(t)
- **Ensemble:** GBM (50%) + Random Forest (30%) + Logistic Regression (20%)
- **Calibracao isotonica:** ajustada sobre predicoes out-of-fold para evitar overfit
- **Extrapolacao KM:** horizontes degenerados (100% positivos) usam razao Kaplan-Meier
- **Monotonicidade:** prob_12h <= prob_24h <= prob_48h <= prob_72h garantida em pos-processamento
