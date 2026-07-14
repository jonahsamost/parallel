import os
import logging

def get_logger(name: str, log_level: str | None = None):
    if log_level is None:
        log_level = os.environ.get("PARALLEL_LOG_LEVEL", None)
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())
    return logger
