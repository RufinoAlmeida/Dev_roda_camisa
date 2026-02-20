import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def gerar_dashboard_profissional(arquivo_csv):
    """
    Dashboard operacional avançado para análise de risco de incêndios.
    """

    # =========================================================
    # 1. CARREGAR DADOS
    # =========================================================
    df = pd.read_csv(arquivo_csv)

    required_cols = [
        "event_id", "prob_12h", "prob_24h", "prob_48h", "prob_72h"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    # =========================================================
    # 2. SCORE OPERACIONAL (alinhado à competição)
    # =========================================================
    df["risk_score"] = (
        0.3 * df["prob_24h"] +
        0.4 * df["prob_48h"] +
        0.3 * df["prob_72h"]
    )

    # =========================================================
    # 3. TRANSFORMAÇÃO LONG (para gráficos)
    # =========================================================
    df_melted = df.melt(
        id_vars=["event_id"],
        value_vars=["prob_12h", "prob_24h", "prob_48h", "prob_72h"],
        var_name="Horizonte_Tempo",
        value_name="Probabilidade"
    )

    df_melted["Horizonte_Tempo"] = (
        df_melted["Horizonte_Tempo"]
        .str.replace("prob_", "")
        .str.upper()
    )

    # ordem temporal correta (CRÍTICO)
    ordem = ["12H", "24H", "48H", "72H"]
    df_melted["Horizonte_Tempo"] = pd.Categorical(
        df_melted["Horizonte_Tempo"],
        categories=ordem,
        ordered=True
    )

    # =========================================================
    # 4. TOP INCIDENTES CRÍTICOS
    # =========================================================
    top_ids = df.nlargest(5, "risk_score")["event_id"]
    top_data = df_melted[df_melted["event_id"].isin(top_ids)]

    # =========================================================
    # 5. CONFIG VISUAL
    # =========================================================
    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    # =========================================================
    # PAINEL A — BOX PLOT GLOBAL
    # =========================================================
    ax1 = fig.add_subplot(gs[0, 0])

    sns.boxplot(
        data=df_melted,
        x="Horizonte_Tempo",
        y="Probabilidade",
        ax=ax1,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6,
        },
    )

    ax1.set_title("Distribuição Global de Risco")
    ax1.set_ylim(0, 1.05)

    # =========================================================
    # PAINEL B — HISTOGRAMA DE RISCO 48H
    # =========================================================
    ax2 = fig.add_subplot(gs[0, 1])

    sns.histplot(
        df["prob_48h"],
        bins=20,
        kde=True,
        ax=ax2,
    )

    ax2.set_title("Distribuição de Probabilidade (48h)")
    ax2.set_xlabel("Probabilidade")

    # =========================================================
    # PAINEL C — HEATMAP DE MÉDIAS
    # =========================================================
    ax3 = fig.add_subplot(gs[0, 2])

    heat_data = (
        df[["prob_12h", "prob_24h", "prob_48h", "prob_72h"]]
        .mean()
        .to_frame(name="Probabilidade Média")
        .T
    )

    sns.heatmap(
        heat_data,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax3,
        cbar=False,
    )

    ax3.set_title("Mapa de Risco Médio")

    # =========================================================
    # PAINEL D — TRAJETÓRIA DOS TOP 5
    # =========================================================
    ax4 = fig.add_subplot(gs[1, :2])

    sns.lineplot(
        data=top_data,
        x="Horizonte_Tempo",
        y="Probabilidade",
        hue="event_id",
        marker="o",
        linewidth=2,
        ax=ax4,
        palette="tab10",
    )

    ax4.set_title("Top 5 Incidentes Críticos")
    ax4.set_ylim(0, 1.05)
    ax4.legend(title="Event ID", bbox_to_anchor=(1.02, 1), loc="upper left")

    # =========================================================
    # PAINEL E — RANKING OPERACIONAL
    # =========================================================
    ax5 = fig.add_subplot(gs[1, 2])

    ranking = (
        df.sort_values("risk_score", ascending=False)
        .head(10)[["event_id", "risk_score"]]
        .iloc[::-1]
    )

    ax5.barh(ranking["event_id"].astype(str), ranking["risk_score"])
    ax5.set_title("Ranking de Prioridade")
    ax5.set_xlabel("Risk Score")

    # =========================================================
    # FINALIZAÇÃO
    # =========================================================
    plt.tight_layout()
    plt.savefig("dashboard_profissional_incendios.png", dpi=300)
    plt.show()

    print("Dashboard profissional gerado com sucesso.")


# =========================================================
# EXECUÇÃO
# =========================================================
gerar_dashboard_profissional("submission.csv")
