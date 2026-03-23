"""Structured logging configuration using structlog.

Provides console (colored, human-readable), JSON, and clean output formats,
integrated with stdlib logging for third-party library compatibility.
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog
from rich.console import Console

from agent.config import LoggingConfig

_console = Console(stderr=True)

# Third-party loggers to suppress at WARNING level
_NOISY_LOGGERS = (
    "httpx",
    "litellm",
    "httpcore",
    "uvicorn.access",
    "chromadb",
    "apscheduler",
)


def _clean_renderer(_logger: Any, _method: str, event_dict: MutableMapping[str, Any]) -> str:
    """Render log events as clean human-readable messages.

    At INFO+, shows only the event text.
    At DEBUG, includes key=value context for debugging.
    """
    event = event_dict.get("event", "")
    level = event_dict.get("level", "info")

    # Format the event name: replace underscores with spaces, title case
    message = str(event).replace("_", " ")

    if level == "debug":
        # Include context for debug messages
        extras = {
            k: v
            for k, v in event_dict.items()
            if k not in ("event", "level", "timestamp", "logger", "logger_name")
        }
        if extras:
            parts = [f"{k}={v}" for k, v in extras.items()]
            message = f"{message}  ({', '.join(parts)})"

    # Add level prefix for warnings and errors
    if level in ("warning", "error", "critical"):
        message = f"[{level.upper()}] {message}"

    return message


def setup_logging(config: LoggingConfig) -> None:
    """Configure structlog and stdlib logging.

    Args:
        config: Logging configuration with level and format settings.
               Supported formats: "console" (default), "json", "clean".
    """
    log_level = getattr(logging, config.level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if config.format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    elif config.format == "clean":
        renderer = _clean_renderer
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Ensure stderr uses UTF-8 on Windows to avoid UnicodeEncodeError
    # with non-ASCII characters (emojis, CJK, etc.)
    stderr = sys.stderr
    if hasattr(stderr, "reconfigure"):
        with contextlib.suppress(Exception):
            stderr.reconfigure(encoding="utf-8", errors="replace")

    handler = logging.StreamHandler(stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Optional file-based logging with rotation
    if config.log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_file,
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backup_count,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.setLevel(log_level)

    # Quiet down noisy third-party loggers
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the given module name.

    Args:
        name: Module name, typically __name__.

    Returns:
        A bound structlog logger instance.
    """
    return structlog.get_logger(name)
