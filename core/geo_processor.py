# 1 ===== IMPORTAÇÕES =====
import logging
import os

import numpy as np
import rasterio
from rasterio.warp import transform


# 2 ===== CLASSE GeoProcessor =====
# Esta classe processa arquivos GeoTIFF para extração de elevações com interpolação bilinear precisa.
class GeoProcessor:
    """Processa arquivos GeoTIFF para extração de elevações com interpolação bilinear precisa."""

    def __init__(self, geotiff_path):
        if not os.path.exists(geotiff_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {geotiff_path}")
        self.dataset = rasterio.open(geotiff_path)
        logging.info(f"GeoTIFF carregado. Resolução: {self.dataset.res[0]:.2f}m")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_elevation(self, lat, lon, method="bilinear"):
        """Obtém a elevação com método de interpolação especificado.

        Args:
            lat: Latitude em graus decimais
            lon: Longitude em graus decimais
            method: Método de interpolação (apenas 'bilinear' suportado atualmente)
        """
        if method != "bilinear":
            raise ValueError("Apenas interpolação bilinear é suportada atualmente")

        try:
            # Converte coordenadas para o CRS do GeoTIFF
            lon_t, lat_t = transform("EPSG:4326", self.dataset.crs, [lon], [lat])
            x, y = lon_t[0], lat_t[0]

            # Coordenadas fracionárias
            col_f, row_f = self.dataset.index(x, y)
            px, py = self.dataset.xy(row_f, col_f)
            row_f += (py - y) / self.dataset.res[1]
            col_f += (x - px) / self.dataset.res[0]

            row0, col0 = int(np.floor(row_f)), int(np.floor(col_f))
            row1, col1 = row0 + 1, col0 + 1

            if not (
                0 <= row0 < self.dataset.height - 1
                and 0 <= col0 < self.dataset.width - 1
            ):
                return None

            # Lê os 4 pixels ao redor
            window = ((row0, row1 + 1), (col0, col1 + 1))
            values = self.dataset.read(1, window=window, boundless=True)

            if np.any(values == self.dataset.nodata):
                return None

            # Interpolação bilinear
            dx = col_f - col0
            dy = row_f - row0
            top = (1 - dx) * values[0, 0] + dx * values[0, 1]
            bottom = (1 - dx) * values[1, 0] + dx * values[1, 1]
            interpolated = (1 - dy) * top + dy * bottom

            return float(interpolated)

        except Exception as e:
            logging.error(f"Erro ao obter elevação: {str(e)}")
            return None

    def close(self):
        if hasattr(self, "dataset") and not self.dataset.closed:
            self.dataset.close()
