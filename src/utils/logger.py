from loguru import logger
import os
from pathlib import Path

LOG_PATH = os.getenv("LOG_PATH", "logs/app.log")
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

logger.add(LOG_PATH, rotation="10 MB", backtrace=True, diagnose=True)

def get_logger(name: str):
    return logger.bind(module=name)