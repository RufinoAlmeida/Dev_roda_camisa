import os
import sys
import warnings

# ─── CORREÇÃO [1]: backend TkAgg DEVE ser definido antes de qualquer ─────────
# import de matplotlib.pyplot. Se pyplot já foi importado com outro backend,
# matplotlib.use() lança um RuntimeError e a janela nunca abre.
# ─────────────────────────────────────────────────────────────────────────────
_TKINTER_DISPONIVEL = True
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from tkinter import BOTH, BOTTOM, TOP, X, LEFT, RIGHT, W
    import matplotlib
    matplotlib.use("TkAgg")                          # <- antes do pyplot
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
except ImportError:
    _TKINTER_DISPONIVEL = False

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# Compatibilidade de encoding no terminal Windows (cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── CAMINHOS ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV  = os.path.join(BASE_DIR, "submission.csv")
OUTPUT_PNG = os.path.join(BASE_DIR, "dashboard_incendios.png")


# =============================================================================
# CORREÇÃO [8]: funções separadas por responsabilidade
# =============================================================================

def _preparar_dados(arquivo_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega, valida e prepara os dados para visualização.

    Returns
    -------
    (df, df_melted) : DataFrame wide + DataFrame long para os gráficos.
    """
    if not os.path.exists(arquivo_csv):
        raise FileNotFoundError(
            f"\n[ERRO] '{arquivo_csv}' não encontrado.\n"
            f"   Coloque submission.csv na mesma pasta que este script.\n"
        )

    df = pd.read_csv(arquivo_csv)

    required_cols = ["event_id", "prob_12h", "prob_24h", "prob_48h", "prob_72h"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    # Score operacional alinhado à métrica da competição
    df["risk_score"] = (
        0.3 * df["prob_24h"] +
        0.4 * df["prob_48h"] +
        0.3 * df["prob_72h"]
    )

    # ID abreviado — IDs de 8 dígitos são ilegíveis em legendas e eixo Y
    df["event_id_str"] = "ID-" + df["event_id"].astype(str).str[-6:]

    # Formato long para gráficos por horizonte
    df_melted = df.melt(
        id_vars=["event_id", "event_id_str"],
        value_vars=["prob_12h", "prob_24h", "prob_48h", "prob_72h"],
        var_name="Horizonte_Tempo",
        value_name="Probabilidade",
    )
    df_melted["Horizonte_Tempo"] = (
        df_melted["Horizonte_Tempo"]
        .str.replace("prob_", "")
        .str.upper()
    )
    ordem = ["12H", "24H", "48H", "72H"]
    df_melted["Horizonte_Tempo"] = pd.Categorical(
        df_melted["Horizonte_Tempo"], categories=ordem, ordered=True
    )

    return df, df_melted


def _build_figure(df: pd.DataFrame, df_melted: pd.DataFrame) -> plt.Figure:
    """
    Constrói e retorna a figura matplotlib SEM fechar nem salvar.

    CORREÇÃO [2]: a figura é retornada viva para ser embutida no canvas tkinter.
    plt.close() NÃO é chamado aqui — o chamador decide o que fazer com ela.
    """
    VERMELHO   = "#E63946"
    AZUL_DARK  = "#1D3557"
    AZUL_MID   = "#457B9D"
    PALETA_BOX = [AZUL_MID, AZUL_DARK, VERMELHO, "#F4A261"]

    sns.set_theme(style="whitegrid", font_scale=1.0)

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        "Dashboard Operacional — Risco de Incêndios Florestais",
        fontsize=16, fontweight="bold", y=1.005, color=AZUL_DARK,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    top_ids  = df.nlargest(5, "risk_score")["event_id"].values
    top_data = df_melted[df_melted["event_id"].isin(top_ids)].copy()

    # ── PAINEL A — Boxplot global ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(
        data=df_melted, x="Horizonte_Tempo", y="Probabilidade",
        hue="Horizonte_Tempo", palette=PALETA_BOX, legend=False, ax=ax1,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black",
                   "markeredgecolor": "black", "markersize": 5},
        linewidth=1.2,
    )
    ax1.set_title("Distribuição Global de Risco", fontweight="bold", pad=8)
    ax1.set_xlabel("Horizonte Temporal")
    ax1.set_ylabel("Probabilidade")
    ax1.set_ylim(-0.02, 1.08)
    ax1.axhline(0.7, color=VERMELHO, linestyle="--", linewidth=0.8,
                alpha=0.6, label="Limiar crítico (70%)")
    ax1.legend(fontsize=8, loc="upper left")

    # ── PAINEL B — Histograma 48h ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(df["prob_48h"], bins=20, kde=True, ax=ax2,
                 color=AZUL_MID, edgecolor="white")
    ax2.set_title("Distribuição de Probabilidade (48h)", fontweight="bold", pad=8)
    ax2.set_xlabel("Probabilidade")
    ax2.set_ylabel("Frequência")
    mediana = df["prob_48h"].median()
    ax2.axvline(mediana, color=VERMELHO, linestyle="--", linewidth=1.2,
                label=f"Mediana = {mediana:.2f}")
    ax2.legend(fontsize=8)

    # ── PAINEL C — Heatmap de médias ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    heat_data = (
        df[["prob_12h", "prob_24h", "prob_48h", "prob_72h"]]
        .mean()
        .rename({"prob_12h": "12h", "prob_24h": "24h",
                 "prob_48h": "48h", "prob_72h": "72h"})
        .to_frame(name="Probabilidade Média")
        .T
    )
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax3,
                cbar=False, linewidths=1,
                annot_kws={"size": 12, "weight": "bold"}, vmin=0, vmax=1)
    ax3.set_title("Probabilidade Média por Horizonte", fontweight="bold", pad=8)
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)

    # ── PAINEL D — Trajetória Top 5 ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    sns.lineplot(
        data=top_data, x="Horizonte_Tempo", y="Probabilidade",
        hue="event_id_str", marker="o", linewidth=2,
        ax=ax4, palette="tab10",
    )
    ax4.set_title("Trajetória de Risco — Top 5 Incidentes Críticos",
                  fontweight="bold", pad=8)
    ax4.set_xlabel("Horizonte Temporal")
    ax4.set_ylabel("Probabilidade")
    ax4.set_ylim(-0.02, 1.08)
    ax4.axhline(0.7, color=VERMELHO, linestyle="--", linewidth=0.8, alpha=0.6)
    ax4.legend(title="Event ID", loc="lower right", fontsize=8, title_fontsize=8)

    # ── PAINEL E — Ranking Top 10 ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ranking = (
        df.sort_values("risk_score", ascending=False)
        .head(10)[["event_id_str", "risk_score"]]
        .iloc[::-1]
    )
    cores = [VERMELHO if v >= 0.7 else AZUL_MID for v in ranking["risk_score"]]
    bars  = ax5.barh(ranking["event_id_str"], ranking["risk_score"],
                     color=cores, edgecolor="white", height=0.6)
    for bar, val in zip(bars, ranking["risk_score"]):
        ax5.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", ha="right",
                 fontsize=8, color="white", fontweight="bold")
    ax5.set_title("Top 10 — Ranking de Prioridade", fontweight="bold", pad=8)
    ax5.set_xlabel("Risk Score")
    ax5.set_xlim(0, 1.05)
    ax5.tick_params(axis="y", labelsize=8)

    return fig


def save_png(fig: plt.Figure, output_path: str) -> None:
    """Salva a figura como PNG sem fechar."""
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"[OK] Dashboard salvo em: {output_path}")


def show_window(fig: plt.Figure, output_path: str) -> None:
    """
    Exibe a figura em uma janela tkinter com toolbar e botão de salvar.

    CORREÇÃO [2]: recebe a figura viva — não chama plt.close() internamente.
    CORREÇÃO [3]: WM_DELETE_WINDOW destrói a janela e encerra corretamente.
    CORREÇÃO [4]: toolbar em frame dedicado com side=BOTTOM.
    CORREÇÃO [5]: canvas com fill=BOTH + expand=True para escala correta.
    CORREÇÃO [6]: root.geometry() define tamanho inicial adequado.
    CORREÇÃO [7]: botão "Salvar PNG" explícito com diálogo de arquivo.
    """
    root = tk.Tk()
    root.title("Dashboard Operacional — Risco de Incêndios Florestais")

    # CORREÇÃO [6]: tamanho inicial = 1400×800 px, redimensionável
    root.geometry("1400x800")
    root.resizable(True, True)
    root.configure(bg="#F0F4F8")

    # ── Barra de controles no topo ─────────────────────────────────────────
    control_frame = tk.Frame(root, bg="#1D3557", height=42)
    control_frame.pack(side=TOP, fill=X)
    control_frame.pack_propagate(False)

    tk.Label(
        control_frame,
        text="  Wildfire Risk Dashboard",
        bg="#1D3557", fg="white",
        font=("Helvetica", 12, "bold"),
    ).pack(side=LEFT, padx=8)

    # CORREÇÃO [7]: botão Salvar PNG com diálogo de arquivo
    def salvar_png_dialog():
        caminho = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            initialfile=os.path.basename(output_path),
            initialdir=os.path.dirname(output_path),
            title="Salvar dashboard como PNG",
        )
        if caminho:
            save_png(fig, caminho)
            messagebox.showinfo("Salvo", f"Dashboard salvo em:\n{caminho}")

    ttk.Button(
        control_frame, text="💾  Salvar PNG", command=salvar_png_dialog
    ).pack(side=RIGHT, padx=10, pady=6)

    # ── Frame principal do canvas ──────────────────────────────────────────
    canvas_frame = tk.Frame(root, bg="#F0F4F8")
    canvas_frame.pack(side=TOP, fill=BOTH, expand=True)

    # CORREÇÃO [5]: fill=BOTH + expand=True — canvas escala com a janela
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=4, pady=4)

    # CORREÇÃO [4]: toolbar em frame dedicado na base do canvas_frame
    toolbar_frame = tk.Frame(canvas_frame, bg="#EAECEF", height=30)
    toolbar_frame.pack(side=BOTTOM, fill=X)
    toolbar_frame.pack_propagate(False)

    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    # CORREÇÃO [3]: protocolo de fechamento limpo
    def on_close():
        plt.close(fig)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()


def gerar_dashboard(
    arquivo_csv: str = INPUT_CSV,
    output_path: str = OUTPUT_PNG,
    mostrar_janela: bool = True,
    salvar_png_auto: bool = True,
) -> None:
    """
    Ponto de entrada principal.

    Parameters
    ----------
    arquivo_csv    : caminho para o submission.csv
    output_path    : caminho de saída da imagem PNG
    mostrar_janela : exibe a janela tkinter (default True; False para modo headless)
    salvar_png_auto: salva o PNG automaticamente ao gerar (default True)
    """
    df, df_melted = _preparar_dados(arquivo_csv)
    fig           = _build_figure(df, df_melted)

    if salvar_png_auto:
        save_png(fig, output_path)

    # CORREÇÃO [9]: fallback headless se tkinter não estiver disponível
    if mostrar_janela:
        if not _TKINTER_DISPONIVEL:
            print("[!] tkinter não disponível neste ambiente.")
            print("    O dashboard foi salvo como PNG.")
            plt.close(fig)
            return
        show_window(fig, output_path)
    else:
        plt.close(fig)


# =============================================================================
if __name__ == "__main__":
    gerar_dashboard()