import logging
import logging.config
from pathlib import Path

import yaml


CONFIG_PATH = Path("config") / "logging.yaml"


def get_logger(name: str, config_path: Path = CONFIG_PATH) -> logging.Logger:
    """[summary]

    Args:
        name (str): loggerの名前 (通常は呼び出し元の __name__)
        config_path (Path, optional): 設定ファイル (yaml) のパス

    Returns:
        logging.Logger: 設定済みのlogger
    """
    with open(config_path, "r") as f:
        log_config = yaml.safe_load(f)

    logging.config.dictConfig(log_config)

    return logging.getLogger(name)
