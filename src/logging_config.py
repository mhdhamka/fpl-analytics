"""Central logging configuration.

Replaces scattered `print()` calls with proper leveled logging that writes to
both the console and a rotating log file, so pipeline runs (especially in
CI/cron) leave a debuggable trail.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from src.config import LOG_DIR, LOG_LEVEL

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring handlers on first use."""
    global _CONFIGURED
    if not _CONFIGURED:
        os.makedirs(LOG_DIR, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(LOG_LEVEL)

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, "pipeline.log"), maxBytes=2_000_000, backupCount=3
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

        _CONFIGURED = True

    return logging.getLogger(name)
