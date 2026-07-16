import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name="RBO_Engine"):
    """
    Sets up a dual-output logger:
    1. Console gets INFO level and above (clean output)
    2. File 'system_debug.log' gets DEBUG level and above, rotating at 10MB (max 5 backups)
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, avoid adding duplicates
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # 2. File Handler (Rotating)
    log_file = "system_debug.log"
    # Max bytes = 10MB (10 * 1024 * 1024)
    file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Create a default global instance to import easily
logger = setup_logger()
