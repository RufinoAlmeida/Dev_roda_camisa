"""
data/loader.py
==============
Responsavel por carregar, validar e reportar estatisticas basicas dos dados.
Nao faz nenhuma transformacao — apenas I/O e sanidade checks.
"""

import os
import pandas as pd


def _check_files(*paths: tuple[str, str]) -> None:
    """Verifica se todos os arquivos necessarios existem. Levanta FileNotFoundError caso contrario."""
    for path, nome in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n[ERRO] Arquivo '{nome}' nao encontrado em:\n   {path}\n\n"
                f"   Coloque train.csv e test.csv na raiz do projeto.\n"
            )


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os conjuntos de treino e teste a partir dos caminhos informados.

    Parameters
    ----------
    train_path : str  — caminho completo para train.csv
    test_path  : str  — caminho completo para test.csv

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]  — (train, test)
    """
    _check_files((train_path, "train.csv"), (test_path, "test.csv"))

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    return train, test


def report_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Imprime um resumo estatistico basico dos dados carregados."""
    print("=" * 60)
    print("PASSO 1: Carregando dados...")
    print("=" * 60)
    print(f"Treino: {train.shape[0]} linhas | Teste: {test.shape[0]} linhas")
    print(f"Eventos positivos: {train['event'].sum()} ({train['event'].mean()*100:.1f}%)")
    print(f"Censurados: {(1-train['event']).sum()} ({(1-train['event']).mean()*100:.1f}%)")
