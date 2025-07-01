import csv
import glob
import os
from typing import Optional, Tuple

import pandas as pd
import rasterio
import requests
from dotenv import load_dotenv
from pyproj import Transformer
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

load_dotenv()

API_KEY_OPEN_TOPOGRAPHY = os.getenv("API_KEY_OPEN_TOPOGRAPHY")

MAX_SIZE_DEGREES = 0.2  # Aproximadamente 22 km x 22 km

mde_SOURCES = [
    ("AW3D30", "ALOS_World_3D_30m"),
    ("AW3D30_E", "ALOS_World_3D_30m_Elipsoidal"),
    ("SRTMGL1", "SRTM_30m"),
    ("SRTMGL1_E", "SRTM_30m_Elipsoidal"),
]


def identificar_ponto_central(csv_path: str) -> Tuple[float, float]:
    """Lê o CSV e retorna latitude e longitude do primeiro ponto."""
    df = pd.read_csv(csv_path)
    primeiro_ponto = df.iloc[0]
    return primeiro_ponto["Latitude"], primeiro_ponto["Longitude"]


def create_max_bbox(
    lat: float, lon: float, size_degrees: float = MAX_SIZE_DEGREES
) -> Tuple[float, float, float, float]:
    """Cria uma bounding box quadrada centrada no ponto, limitada às coordenadas válidas."""
    half_size = size_degrees / 2
    south = max(lat - half_size, -90)
    north = min(lat + half_size, 90)
    west = max(lon - half_size, -180)
    east = min(lon + half_size, 180)
    if south > north:
        south, north = north, south
    if west > east:
        west, east = east, west
    return south, north, west, east


def arquivo_ja_baixado(caminho: str) -> bool:
    """Verifica se o arquivo já existe para evitar download repetido."""
    return os.path.exists(caminho)


def load_coordinates(csv_path: str):
    coordinates = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Skip rows that don't have valid coordinates
            try:
                lon = float(row["Longitude"])
                lat = float(row["Latitude"])
                coordinate = (lat,lon)
                coordinates.append(coordinate)
            except (ValueError, KeyError):
                continue

    return coordinates


def get_city_name(lat: float, lon: float) -> str | None:
    """
    Returns the city name for a given latitude and longitude using Nominatim API.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
    
    Returns:
        str | None: City name, or None if not found
    """
    try:
        geolocator = Nominatim(user_agent="geoapi")  # You can customize the user agent
        location = geolocator.reverse((lat, lon), exactly_one=True, language='en')
        if location and location.raw and "address" in location.raw:
            address = location.raw["address"]
            return address.get("city") or address.get("town") or address.get("village") or address.get("hamlet")
        else:
            return None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error: {e}")
        return None



def load_local_tiff(origin_folder: str, coordinates: list[tuple[float, float]]):

    tif_files = glob.glob(os.path.join(origin_folder, "*.tif"))

    if not tif_files:
        return

    for file in tif_files:
        # === Input ===

        # === Load TIF bounds and CRS ===
        with rasterio.open(file) as src:
            bounds = src.bounds  # (left, bottom, right, top)
            raster_crs = src.crs

        # === Transform coordinates to raster CRS ===
        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        points_projected = [transformer.transform(lat, lon) for lat, lon in coordinates]

        # === Check if each point is within the TIF bounds ===
        left, bottom, right, top = bounds

        def is_point_inside(x, y):
            return left <= x <= right and bottom <= y <= top

        inside_flags = [is_point_inside(x, y) for x, y in points_projected]
        all_inside = all(inside_flags)

        if all_inside:
            return file

def download_melhor_mde(
    south: float,
    north: float,
    west: float,
    east: float,
    folder: str,
    location_name: str,
) -> Optional[str]:
    """Tenta baixar o melhor mde disponível na ormde definida, retornando o caminho do arquivo salvo."""
    for dem_type, mde_label in mde_SOURCES:
        filename = os.path.join(
            folder, f"mde_{mde_label}_{location_name.replace(' ', '_')}.tif"
        )
        if arquivo_ja_baixado(filename):
            print(f"Arquivo já existe: {filename}")
            return filename

        url = (
            f"https://portal.opentopography.org/API/globaldem?"
            f"demtype={dem_type}&"
            f"south={south}&north={north}&"
            f"west={west}&east={east}&"
            f"outputFormat=GTiff&"
            f"API_Key={API_KEY_OPEN_TOPOGRAPHY}"
        )

        print(f"URL do mde: {url}")

        try:
            print(f"Tentando baixar mde: {mde_label}")
            response = requests.get(url, stream=True, timeout=30)

            if response.status_code == 200:
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Baixado com sucesso: {mde_label}")
                return filename
            else:
                raise Exception(f"{mde_label} falhou (HTTP {response.status_code}).")

        except Exception as e:
            print(f"Erro ao baixar {mde_label}: {e}")

    print("Nenhum mde disponível para a área especificada ou a API não respondeu")
    return None
