"""Logging utilities for Neo Guardrail Hub.

This module provides structured logging using structlog
with consistent formatting across the framework.
"""

import logging
import sys
from typing import Any, Optional

try:
    import structlog
    from structlog.types import Processor

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for the framework.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Whether to output JSON formatted logs
        log_file: Optional file path for log output
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )

    if STRUCTLOG_AVAILABLE:
        # Configure structlog
        processors: list[Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]

        if json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


class SimpleLogger:
    """Simple logger fallback when structlog is not available."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        message = f"{event} | {extra}" if extra else event
        self._logger.log(level, message)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def bind(self, **kwargs: Any) -> "SimpleLogger":
        return self


def get_logger(name: str) -> Any:
    """Get a logger instance.

    Uses structlog if available, otherwise falls back to
    standard logging with a similar interface.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return SimpleLogger(name)


# Default logger for the package
logger = get_logger("neo_guardrail_hub")
