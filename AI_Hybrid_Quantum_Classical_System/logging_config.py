# logging_config.py

import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_file: str = 'system.log', log_level: int = logging.DEBUG):
    """
    Setup logging configuration for the system.

    Parameters:
        log_file (str): The path to the log file.
        log_level (int): The logging level to use (default is DEBUG).
    """
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    handler = RotatingFileHandler(log_file, maxBytes=100000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
