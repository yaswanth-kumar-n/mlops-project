from pathlib import Path
import pandas as pd

from .data_loader import load_raw_data
from .utils.logger import get_logger
from .utils.config_loader import load_config

logger = get_logger(__name__)

def preprocess():
    config = load_config()
    processed_path = config["data"]["processed_path"]

    df = load_raw_data()

    # Simple preprocessing – later we improve
    df = df.dropna()

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Saved processed data to {processed_path}")
    return processed_path

if __name__ == "__main__":
    preprocess()