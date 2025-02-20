import logging
import logging.config
from pathlib import Path
from typing import Optional

def setup_logging(config_file: Optional[str] = "configs/logging_conf.ini") -> tuple[logging.Logger, logging.Logger]:
    """
    Set up logging configuration and return commonly used loggers.
    
    Args:
        config_file: Path to the logging configuration file. Defaults to "configs/logging_conf.ini"
    
    Returns:
        tuple: (logger_plugins, logger_main) - The two most commonly used loggers
    """
    logging.config.fileConfig(config_file)
    logger_plugins = logging.getLogger("plugins")
    logger_main = logging.getLogger("main")
    return logger_plugins, logger_main

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name. If logging hasn't been configured yet, this will configure it
    with the default configuration file.
    
    Args:
        name: Name of the logger to retrieve
    
    Returns:
        logging.Logger: The requested logger
    """
    # Check if logging is configured
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name) 