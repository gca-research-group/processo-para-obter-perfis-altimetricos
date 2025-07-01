"""
Conversor de KML para CSV com Geração de Pontos Intermediários

Este script converte um arquivo KML contendo pontos geográficos
para CSV, ordena os pontos por proximidade e gera pontos intermediários
com distâncias aleatórias entre 8 e 40 metros ao longo da rota.

Autor: Luciana Machado Cardoso
Data: 26/05/2025
"""

import csv

# ===== Imports =====
import os
import xml.etree.ElementTree as ET

import geopy.distance
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def kml_para_lista(kml_path):
    """Converte um arquivo KML em uma lista de pontos com nome, longitude e latitude.

    Args:
        kml_path (str): Caminho do arquivo KML.

    Returns:
        list of tuple: Lista de tuplas no formato (nome, longitude, latitude).
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    placemarks = root.findall(".//{http://www.opengis.net/kml/2.2}Placemark")
    pontos = []

    for placemark in placemarks:
        name = placemark.find(".//{http://www.opengis.net/kml/2.2}name")
        point = placemark.find(".//{http://www.opengis.net/kml/2.2}Point")
        if name is None:
            name = placemark.find(
                './/{http://www.opengis.net/kml/2.2}SimpleData[@name="Nome"]'
            )
        if point is not None:
            coordinates = point.find(".//{http://www.opengis.net/kml/2.2}coordinates")
            if coordinates is not None:
                longitude, latitude, *_ = map(
                    float, coordinates.text.strip().split(",")
                )
                nome_ponto = name.text if name is not None else "Sem Nome"
                pontos.append((nome_ponto, latitude, longitude))
    return pontos


def ordenar_pontos(pontos):
    """Ordena os pontos com base no número inicial do nome.

    Args:
        pontos (list of tuple): Lista de tuplas (nome, longitude, latitude).

    Returns:
        list of tuple: Lista ordenada de pontos.
    """

    def extrair_ordem(nome):
        try:
            return int(nome.split()[0])
        except (IndexError, ValueError):
            return float("inf")  # Coloca no fim se o nome estiver fora do padrão

    pontos_ordenados = sorted(pontos, key=lambda p: extrair_ordem(p[0]))
    return pontos_ordenados


def salvar_pontos_csv(pontos, csv_path):
    """Salva uma lista de pontos em um arquivo CSV.

    Args:
        pontos (list of tuple): Lista de tuplas (nome, longitude, latitude).
        csv_path (str): Caminho do arquivo CSV de saída.
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Nome", "Latitude", "Longitude"])
        for ponto in pontos:
            writer.writerow(ponto)


def gerar_pontos_intermediarios(csv_path, output_csv):
    """Adiciona pontos intermediários entre os pontos existentes a cada 20 metros.

    Args:
        csv_path (str): Caminho do CSV com os pontos ordenados.
        output_csv (str): Caminho para salvar o CSV com pontos intermediários.
    """
    df = pd.read_csv(csv_path)

    # Calcula distâncias acumuladas entre pontos consecutivos
    cum_distances = [0] + list(
        np.cumsum(
            [
                geopy.distance.geodesic(
                    (df["Latitude"][i], df["Longitude"][i]),
                    (df["Latitude"][i + 1], df["Longitude"][i + 1]),
                ).meters
                for i in range(len(df) - 1)
            ]
        )
    )

    route_length = cum_distances[-1]

    # Gera distâncias para pontos intermediários a cada 20 metros
    new_distances = np.arange(20, route_length, 20).tolist()

    # Interpolação de latitude e longitude
    interp_lat = interp1d(cum_distances, df["Latitude"], kind="linear")
    interp_lon = interp1d(cum_distances, df["Longitude"], kind="linear")

    interpolated_lons = interp_lon(new_distances)
    interpolated_lats = interp_lat(new_distances)
    new_names = ["intermediario"] * len(new_distances)

    # Mescla os pontos originais com os intermediários
    merged_lons = []
    merged_lats = []
    merged_names = []

    # Ordena os pontos intermediários por distância (por segurança)
    sorted_indices = np.argsort(new_distances)
    sorted_lons = interpolated_lons[sorted_indices]
    sorted_lats = interpolated_lats[sorted_indices]
    sorted_distances = [new_distances[i] for i in sorted_indices]

    current_distance_index = 0
    for i in range(len(df) - 1):
        # Adiciona o ponto original
        merged_lons.append(df["Longitude"][i])
        merged_lats.append(df["Latitude"][i])
        merged_names.append(df["Nome"][i])

        # Adiciona pontos intermediários entre este ponto e o próximo
        while (current_distance_index < len(sorted_distances)) and (
            sorted_distances[current_distance_index] <= cum_distances[i + 1]
        ):
            merged_lons.append(sorted_lons[current_distance_index])
            merged_lats.append(sorted_lats[current_distance_index])
            merged_names.append("intermediario")
            current_distance_index += 1

    # Adiciona o último ponto original
    merged_lons.append(df["Longitude"].iloc[-1])
    merged_lats.append(df["Latitude"].iloc[-1])
    merged_names.append(df["Nome"].iloc[-1])

    # Cria o DataFrame final
    new_df = pd.DataFrame(
        {"Nome": merged_names, "Latitude": merged_lats, "Longitude": merged_lons}
    )

    new_df.to_csv(output_csv, index=False)
    print(f"Arquivo CSV com pontos intermediários (20m) salvo em: {output_csv}")


def converter_arquivo(kml_path: str, ordered_csv: str, final_csv: str):
    """Executa o processo completo de conversão KML → CSV com pontos intermediários."""

    # Garante que o diretório existe
    os.makedirs(os.path.dirname(ordered_csv), exist_ok=True)

    pontos = kml_para_lista(kml_path)
    pontos_ordenados = ordenar_pontos(pontos)
    salvar_pontos_csv(pontos_ordenados, ordered_csv)
    gerar_pontos_intermediarios(ordered_csv, final_csv)

    print(f"Processo concluído! Arquivo final: {final_csv}")
    return final_csv
