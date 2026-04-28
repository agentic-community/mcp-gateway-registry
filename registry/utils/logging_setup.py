"""Shared logging configuration for registry and auth-server.

Configures three output destinations:
1. Console (stdout/stderr) - always enabled
2. RotatingFileHandler - rotated log file
3. MongoDBLogHandler - optional, writes to MongoDB application_logs collection
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s"


def setup_logging(
    service_name: str,
    log_file: Path | None = None,
) -> Path | None:
    """Configure root logger with console, file, and optional MongoDB handlers.

    Args:
        service_name: Identifies this process in MongoDB log documents
            (e.g. "registry", "auth-server").
        log_file: Explicit log file path. When ``None`` the path is derived
            from settings (``settings.log_dir / f"{service_name}.log"``).

    Returns:
        The resolved log file path, or None if file logging was skipped.
    """
    from ..core.config import settings

    level = getattr(logging, settings.app_log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)

    # 1. Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # 2. RotatingFileHandler
    resolved_log_file: Path | None = None
    if log_file is not None:
        resolved_log_file = log_file
    else:
        resolved_log_file = settings.log_dir / f"{service_name}.log"

    if resolved_log_file is not None:
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(resolved_log_file),
            maxBytes=settings.app_log_max_bytes,
            backupCount=settings.app_log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # 3. Centralized log handler (optional, writes to MongoDB/DocumentDB)
    if settings.app_log_centralized_enabled and settings.storage_backend in (
        "documentdb",
        "mongodb-ce",
    ):
        try:
            from .mongodb_log_handler import MongoDBLogHandler

            excluded = frozenset(
                name.strip()
                for name in settings.app_log_excluded_loggers.split(",")
                if name.strip()
            )
            mongo_handler = MongoDBLogHandler(
                service_name=service_name,
                buffer_size=settings.app_log_mongodb_buffer_size,
                flush_interval=settings.app_log_mongodb_flush_interval_seconds,
                ttl_days=settings.app_log_centralized_ttl_days,
                excluded_loggers=excluded,
            )
            mongo_handler.setLevel(level)
            mongo_handler.setFormatter(formatter)
            root.addHandler(mongo_handler)
        except Exception as exc:
            root.warning(f"Failed to initialize MongoDB log handler: {exc}")

    return resolved_log_file
