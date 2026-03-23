"""
Structured logging configuration using structlog.
Produces JSON logs in production, pretty-printed in development.
"""

import sys

import structlog

from config.settings import settings


def setup_logging():
    """Call once at startup to configure structlog."""

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_level == "DEBUG":
        # Pretty console output for development
        renderer = structlog.dev.ConsoleRenderer()
    else:
        # JSON for production / log aggregation
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            _level_to_int(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def _level_to_int(level: str) -> int:
    mapping = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }
    return mapping.get(level.upper(), 20)
