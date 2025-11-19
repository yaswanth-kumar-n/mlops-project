import pandas as pd
from .utils.logger import get_logger
from .utils.config_loader import load_config

logger = get_logger(__name__)

def load_raw_data():
    config = load_config()
    raw_path = config["data"]["raw_path"]
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded data with shape {df.shape}")
    return df