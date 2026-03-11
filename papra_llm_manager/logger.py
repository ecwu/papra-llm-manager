"""Centralized logging configuration using loguru."""

import sys
import os
from loguru import logger

# Remove default handler
logger.remove()

# Get log level from environment variable, default to INFO
# This means DEBUG messages will be hidden by default
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Add custom handler with formatted output
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level=LOG_LEVEL,
    colorize=True,
)

# Export logger instance
__all__ = ["logger"]
