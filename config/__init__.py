import logging
import os
from logging.handlers import TimedRotatingFileHandler


def config():

    if not os.path.exists("logs"):
        os.mkdir("logs")

    formatter = logging.Formatter(
        "%(name)s | [%(asctime)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = TimedRotatingFileHandler(
        os.path.join(os.getcwd(), "logs", "project.log"),
        when="midnight",
        interval=1,
        backupCount=7,
    )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, logging.StreamHandler()],
    )
