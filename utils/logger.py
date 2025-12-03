"""
Logging configuration using loguru.
Provides structured logging with file and console output.
"""

import sys
from pathlib import Path
from loguru import logger as _logger

# Remove default handler
_logger.remove()

# Add console handler with custom format
_logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True
)

# Create logs directory
log_dir = Path(__file__).parent.parent / ".logs"
log_dir.mkdir(exist_ok=True)

# Add file handler for debug logs
_logger.add(
    log_dir / "debug.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip"
)

# Add file handler for error logs
_logger.add(
    log_dir / "error.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="ERROR",
    rotation="10 MB",
    retention="30 days",
    compression="zip"
)

# Export logger
logger = _logger