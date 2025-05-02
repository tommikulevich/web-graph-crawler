import os
import logging

_file_logger_init = False


def init_file_logger(log_file: str) -> None:
    """Initialize file logging: add a FileHandler to the root logger."""
    
    global _file_logger_init
    if _file_logger_init:
        return
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    
    _file_logger_init = True

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger (console + propagation to file)."""
    
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger
