import logging
import os

from dotenv import load_dotenv

from config import config
from core.data_processing import (
    gerar_metricas_finais,
    processar_dados_calculados,
)
from core.geotiff import (
    create_max_bbox,
    download_melhor_mde,
    get_city_name,
    load_coordinates,
    load_local_tiff,
)
from core.kml import converter_arquivo
from core.visualization import (
    plotar_perfis_altimetricos,
)
import re

load_dotenv()
config()

def sanitize_filename(name):
    """ Remove special characters, replace spaces with underscores, and lowercase """
    name = re.sub(r"[^\w\s]", "", name)
    name = name.replace(" ", "_")
    return name.lower()

def main():
    try:
        # Diretórios de trabalho
        saida = "assets/saida"
        dados_entrada = "assets/dados_entrada"
        geotiff_urbanos = "assets/geotiff_urbanos"


        os.makedirs(saida, exist_ok=True)
        os.makedirs(dados_entrada, exist_ok=True)
        os.makedirs(geotiff_urbanos, exist_ok=True)

        logging.info("Iniciando o pipeline de processamento de perfis altimétricos...")

        # 1. Converter KML para CSV com pontos intermediários
        converter_arquivo(
            kml_path="assets/dados_entrada/linha.kml",
            ordered_csv="assets/saida/saida_ordenada.csv",
            final_csv="assets/saida/saida_com_pontos_intermediarios.csv",
        )

        INPUT_CSV = f"{saida}/saida_com_pontos_intermediarios.csv"
       

        coordinates = load_coordinates(INPUT_CSV)
        GEOTIFF_PATH = load_local_tiff("assets/geotiff_urbanos", coordinates)

        if not GEOTIFF_PATH:
            geotiff_folder = "assets/geotiff_urbanos"

            lat, lon = coordinates[0]
            logging.info(f"Coordenadas do ponto central: Latitude {lat}, Longitude {lon}")

            city_name = sanitize_filename(get_city_name(lat, lon))

            print(f"Nome da cidade: {city_name}")

            # 2. Download do mde baseado no ponto central do CSV
            logging.info("Obtendo dados de elevação...")


            south, north, west, east = create_max_bbox(lat, lon)
            logging.info(
                f"Bounding box criada: South {south}, North {north}, West {west}, East {east}"
            )

            GEOTIFF_PATH = download_melhor_mde(
                south, north, west, east, geotiff_folder, city_name
            )

        if not GEOTIFF_PATH:
            raise Exception(
                "Não foi possível baixar os dados de elevação para a área especificada"
            )
        
        logging.info(f"Dados mde disponíveis em: {GEOTIFF_PATH}")

        # 3. Processamento dos dados altimétricos calculados
        logging.info("Processando dados altimétricos calculados...")
        df_calculado, _ = processar_dados_calculados(
            INPUT_CSV,
            GEOTIFF_PATH,
            f"{saida}/dados_processados.csv",
        )

        logging.info("Gerando visualizações dos perfis altimétricos...")
        
        plotar_perfis_altimetricos(
                df_calculado=df_calculado,
                output_mde=f"{saida}/comparacao_perfis.png",
          
        )
        logging.info(
            f"Visualizações salvas em: {saida}/comparacao_perfis.png"
        )
        # 4. Gerar métricas e salvar
        logging.info("Gerando métricas de qualidade...")

        df_metricas_calculado = gerar_metricas_finais(df_calculado)
        df_metricas_calculado.to_csv(f"{saida}/metricas.csv", index=False)
        logging.info(f"Métricas calculadas salvas em: {saida}/metricas.csv")

        logging.info("Processo concluído com sucesso!")

    except Exception as e:
        logging.error(f"Erro no pipeline principal: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
