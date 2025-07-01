# 1 ===== IMPORTAÇÕES =====
import logging
import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from geopy import distance
from scipy.interpolate import interp1d, make_interp_spline
from scipy.signal import savgol_filter

from core.geo_processor import GeoProcessor



# ===== FUNÇÕES PARA CÁLCULO DE DISTÂNCIA =====


def calcular_distancia_acumulada_3d(
    df: pd.DataFrame, col_altitude: str, col_output: str
) -> pd.DataFrame:
    """Calcula distância acumulada 3D mantendo todas as colunas originais"""
    try:
        distances = [0.0]
        for i in range(len(df) - 1):
            lat1, lon1 = df["Latitude"].iloc[i], df["Longitude"].iloc[i]
            alt1 = df[col_altitude].iloc[i]

            lat2, lon2 = (
                df["Latitude"].iloc[i + 1],
                df["Longitude"].iloc[i + 1],
            )
            alt2 = df[col_altitude].iloc[i + 1]

            flat_dist = distance.distance((lat1, lon1), (lat2, lon2)).m
            elev_diff = abs(alt2 - alt1)
            distance_3d = math.sqrt(flat_dist**2 + elev_diff**2)

            distances.append(distances[-1] + distance_3d)

        df[col_output] = distances
        return df
    except Exception as e:
        logging.error(f"Erro no cálculo de distância 3D: {str(e)}")
        raise e


def filtrar_pontos_distancia_minima(df, distancia_minima=20.0):
    """
    Remove pontos consecutivos a menos de `distancia_minima` metros,
    exceto os com 'parada' ou 'ref' no nome.
    """
    if len(df) <= 1:
        return df.copy()

    indices_manter = [0]
    ultimo_index_mantido = 0

    for i in range(1, len(df)):
        nome_atual = df.loc[i, "Nome"].lower()

        if ("parada" in nome_atual) or ("ref" in nome_atual):
            indices_manter.append(i)
            ultimo_index_mantido = i
            continue

        lat1, lon1 = df.loc[ultimo_index_mantido, ["Latitude", "Longitude"]]
        lat2, lon2 = df.loc[i, ["Latitude", "Longitude"]]
        dist = distance.distance((lat1, lon1), (lat2, lon2)).m

        if dist >= distancia_minima:
            indices_manter.append(i)
            ultimo_index_mantido = i

    return df.loc[sorted(indices_manter)].reset_index(drop=True)


# ===== FUNÇÕES PARA SUAVIZAÇÃO =====


def suavizar_altitudes(
    df: pd.DataFrame,
    col_distancia: str,  # Coluna com distâncias acumuladas
    col_altitude: str,  # Coluna com altitudes
    col_output: str,  # Coluna de saída para altitudes suavizadas
    inclinacao_maxima: float = 8.0,  # Inclinação máxima permitida
    window_size_percent: float = 0.15,  # Tamanho da janela de suavização (em %)
    polyorder: int = 3,  # Ordem do polinômio para suavização
) -> pd.DataFrame:
    """Suaviza altitudes mantendo inclinação controlada e todas as colunas originais"""
    try:
        # Limita ordem do polinômio entre 3 e 4
        polyorder = max(2, min(3, polyorder))

        altitudes = df[col_altitude].values
        distancias = df[col_distancia].values

        # Configuração da janela
        min_window = 5
        max_window = 101  # Tamanho máximo da janela em pontos
        window_length = max(
            min(int(len(altitudes) * window_size_percent), max_window),
            min_window,
        )
        window_length = window_length if window_length % 2 else window_length - 1

        # Suavização principal
        if len(altitudes) > window_length:
            altitudes_suavizadas = savgol_filter(
                altitudes,
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",
            )
        else:
            altitudes_suavizadas = altitudes.copy()

        # Cálculo de inclinações
        diff_alt = np.diff(altitudes_suavizadas)
        diff_dist = np.diff(distancias)
        valid_diff = diff_dist > 0
        inclinacoes = np.zeros_like(diff_alt)
        inclinacoes[valid_diff] = (diff_alt[valid_diff] / diff_dist[valid_diff]) * 100

        # Correção de inclinações excessivas
        exceed_mask = np.abs(inclinacoes) > inclinacao_maxima
        if np.any(exceed_mask):
            correction_factors = inclinacao_maxima / np.abs(inclinacoes[exceed_mask])
            exceed_indices = np.where(exceed_mask)[0]

            for idx, i in enumerate(exceed_indices):
                if i + 1 < len(altitudes_suavizadas):
                    delta = altitudes_suavizadas[i + 1] - altitudes_suavizadas[i]
                    altitudes_suavizadas[i + 1] = altitudes_suavizadas[i] + (
                        delta * correction_factors[idx]
                    )

        df[col_output] = altitudes_suavizadas
        return df

    except Exception as e:
        logging.error(f"Erro na suavização de dados: {str(e)}", exc_info=True)
        raise


# ===== FUNÇÕES PARA CÁLCULO DE MÉTRICAS =====
def interpolar_para_comparar(
    df1: pd.DataFrame,
    dist_col1: str,
    alt_col1: str,
    df2: pd.DataFrame,
    dist_col2: str,
    alt_col2: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpola dois conjuntos de dados para pontos comuns de distância, evitando extrapolação incorreta"""

    # Definir faixa comum com segurança
    dist_min = max(df1[dist_col1].min(), df2[dist_col2].min())
    dist_max = min(df1[dist_col1].max(), df2[dist_col2].max())

    # Definir resolução média com base no primeiro DataFrame
    resolucao = np.mean(np.diff(df1[dist_col1].values))
    n_pontos = int((dist_max - dist_min) / resolucao)

    # Gera pontos uniformes apenas dentro da faixa comum
    distancias_comuns = np.linspace(dist_min, dist_max, n_pontos)

    # Interpolação sem extrapolação
    f1 = interp1d(
        df1[dist_col1],
        df1[alt_col1],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    f2 = interp1d(
        df2[dist_col2],
        df2[alt_col2],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    alt1_interp = f1(distancias_comuns)
    alt2_interp = f2(distancias_comuns)

    # Remove pontos com NaN (fora do domínio)
    mask = ~np.isnan(alt1_interp) & ~np.isnan(alt2_interp)
    return alt1_interp[mask], alt2_interp[mask], distancias_comuns[mask]


def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def calcular_metricas_suavizacao(altitudes_originais, altitudes_suavizadas) -> Dict[str, float]:
    """Calcula métricas de qualidade da suavização"""
    altitudes_originais = np.array(altitudes_originais)
    altitudes_suavizadas = np.array(altitudes_suavizadas)

    return {
        "RMSE (m)": np.sqrt(mean_squared_error(altitudes_originais, altitudes_suavizadas)),
        "MAE (m)": mean_absolute_error(altitudes_originais, altitudes_suavizadas),
        "MaxError (m)": np.max(np.abs(altitudes_originais - altitudes_suavizadas)),
        'StdDev_Diff_comparacao': (np.std(altitudes_originais - altitudes_suavizadas)),
        'Reducao_Ruido (%)': (1 - (np.std(altitudes_suavizadas) / np.std(altitudes_originais)))*100
    }

def analisar_inclinacoes(distancias, altitudes) -> Dict[str, float]:
    """Analisa as características das inclinações"""
    distancias = np.array(distancias)
    altitudes = np.array(altitudes)

    diff_alt = np.diff(altitudes)
    diff_dist = np.diff(distancias)

    with np.errstate(divide='ignore', invalid='ignore'):
        inclinacoes = np.degrees(np.arctan(diff_alt / diff_dist))
        inclinacoes = inclinacoes[~np.isnan(inclinacoes) & ~np.isinf(inclinacoes)]

    return {
        "Media_Inclinacao (°)": np.mean(np.abs(inclinacoes)),
        "Max_Inclinacao (°)": np.max(np.abs(inclinacoes)),
        "Min_Inclinacao (°)": np.min(np.abs(inclinacoes)),
        "Derivada_Media (°)": np.mean(np.abs(np.diff(inclinacoes))),
    }

#====== FUNÇÕES DE GERAÇÃO DE MÉTRICAS =====

def gerar_metricas_finais(df_calculado: pd.DataFrame) -> pd.DataFrame:
    """Gera DataFrame com métricas para os três casos de comparação"""
    metricas = []

    # 1. Processado: Original e Suavizada
    metricas.append({
        "Tipo": "Processado",
        "Comparacao": "Original e Suavizada",
        **calcular_metricas_suavizacao(df_calculado["Altitude"], df_calculado["Altitude_Suavizada"]),
        **analisar_inclinacoes(df_calculado["Distancia_Acumulada"], df_calculado["Altitude_Suavizada"]),
    })

   
    return pd.DataFrame(metricas)

# ===== FUNÇÕES DE PROCESSAMENTO PRINCIPAL =====


def processar_dados_calculados(
    input_csv: str,
    geotiff_path: str,
    output_csv: str,
    inclinacao_maxima: float = 12.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Processa dados brutos e gera CSV com elevações calculadas do GeoTIFF"""
    try:
        logging.info(f"Processando arquivo: {input_csv}")

        df = pd.read_csv(input_csv)

        if not all(col in df.columns for col in ["Latitude", "Longitude"]):
            raise ValueError("CSV deve conter colunas 'Latitude' e 'Longitude'")

        with GeoProcessor(geotiff_path) as processor:
            df["Altitude"] = df.apply(
                lambda row: processor.get_elevation(
                    row["Latitude"], row["Longitude"], "bilinear"
                )
                or 0.0,
                axis=1,
            )

        df = calcular_distancia_acumulada_3d(
            df, "Altitude", "Distancia_Acumulada"
        )
        df = suavizar_altitudes(
            df,
            "Distancia_Acumulada",
            "Altitude",
            "Altitude_Suavizada",
            inclinacao_maxima,
        )

        # Ordem preferencial das colunas
        colunas_ordenadas = [
            "Nome",
            "Latitude",
            "Longitude",
            "Altitude",
            "Distancia_Acumulada",
            "Altitude_Suavizada",
        ]

        # Mantém apenas colunas existentes
        colunas_finais = [col for col in colunas_ordenadas if col in df.columns]

        # Adiciona quaisquer outras colunas que possam existir
        outras_colunas = [col for col in df.columns if col not in colunas_finais]
        colunas_finais.extend(outras_colunas)

        df = df[colunas_finais]

        df.to_csv(output_csv, index=False)
        logging.info(f"Dados calculados processados salvos em: {output_csv}")

        # Retorna também métricas básicas
        df_metricas = pd.DataFrame(
            [
                {
                    "Tipo": "Calculado",
                    **calcular_metricas_suavizacao(
                        df["Altitude"], df["Altitude_Suavizada"]
                    ),
                }
            ]
        )

        return df, df_metricas
    except Exception as e:
        logging.error(f"Erro no processamento de dados altimétricos: {str(e)}")
        raise


# ===== FUNÇÕES DE MODELAGEM =====

def criar_modelo_completo(
    df: pd.DataFrame, tipo: str, spline_degree: int = 3
) -> Dict[str, Any]:
    """Cria modelos spline para dados calculados"""
    if tipo not in ["calculado"]:
        raise ValueError("Tipo deve ser 'calculado'")

    if df is None:
        return None

    x_col = (
        "Distancia_Acumulada"
        
    )
    alt_col = (
        "Altitude_Suavizada"
        
    )

    x = df[x_col].values
    return {
        "altitude": make_interp_spline(x, df[alt_col].values, k=spline_degree),
        "latitude": make_interp_spline(x, df["Latitude"].values, k=spline_degree),
        "longitude": make_interp_spline(x, df["Longitude"].values, k=spline_degree),
        "tipo": tipo,
    }
