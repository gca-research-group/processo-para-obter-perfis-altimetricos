# visualization.py

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import logging
from typing import Optional

import matplotlib.pyplot as plt

# ===== FUNÇÕES PARA PLOTAGEM GRÁFICA =====
import pandas as pd


def plotar_perfis_altimetricos(
    df_calculado: pd.DataFrame,
    output_mde: str,
    texto_parada: str = "Parada",
) -> None:
    """Gera visualização profissional comparando dois perfis e opcionalmente gráficos individuais"""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")

        # Gráfico de comparação
        fig_comp, ax_comp = plt.subplots(figsize=(15, 3.5))

        # Dados processados (rota planejada)
        x_proc = df_calculado["Distancia_Acumulada"].values
        y_orig_proc = df_calculado["Altitude"].values
        y_suav_proc = df_calculado["Altitude_Suavizada"].values

      
        # Cálculo de limites globais (EIXO X e Y)
        min_x = min(x_proc) # referência MDE
        max_x = max(x_proc)

        min_y = min(y_orig_proc)
        max_y = max(y_orig_proc)

        # Plot dos dados processados (verde)
        ax_comp.plot(
            x_proc,
            y_orig_proc,
            color="#388E3C",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label="Elevações MDE",
        )
        ax_comp.plot(
            x_proc,
            y_suav_proc,
            color="#2C6B2F",
            linewidth=2,
            alpha=0.9,
            label="Elevações MDE - Suavizado",
        )
        ax_comp.fill_between(
            x_proc,
            y_suav_proc,
            min(y_orig_proc) - 10,
            color="#2C6B2F",
            alpha=0.1,
        )

        # Marcar paradas no perfil calculado
        if "Nome" in df_calculado.columns:
            paradas_calc = df_calculado[
                df_calculado["Nome"].str.contains(texto_parada, case=False, na=False)
            ]
            if not paradas_calc.empty:
                ax_comp.scatter(
                    paradas_calc["Distancia_Acumulada"],
                    paradas_calc["Altitude_Suavizada"],
                    color="#2E7D32",
                    s=60,
                    marker="o",
                    edgecolors="k",
                    zorder=5,
                    label="Paradas",
                )
        # Configuração dos eixos (unificados)
        ax_comp.set_xlim(min_x, max_x)
        ax_comp.set_ylim(min_y, max_y+20)
        ax_comp.set_xlabel("Distância Acumulada (m)", fontsize=14)
        ax_comp.set_ylabel("Altitude (m)", fontsize=14)
        ax_comp.tick_params(axis="x", labelsize=12)
        ax_comp.tick_params(axis="y", labelsize=12)
        ax_comp.set_title("Comparação de Perfis Altimétricos", pad=20, fontsize=16)
        ax_comp.grid(True, linestyle="--", alpha=0.4)

        ax_comp.legend()

        plt.tight_layout()
        plt.savefig(output_mde, dpi=300, bbox_inches="tight")
        plt.close(fig_comp)
        logging.info(f"Gráfico comparativo salvo em {output_mde}")


        
    except Exception as e:
        logging.error(f"Erro na geração do gráfico: {str(e)}")
        raise

    except Exception as e:
        logging.error(f"Erro ao gerar gráfico: {str(e)}")
        raise
