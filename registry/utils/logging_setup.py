"""Shared logging configuration for registry and auth-server.

Configures three output destinations:
1. Console (stdout/stderr) - always enabled, human-readable format
2. RotatingFileHandler - rotated log file (JSONL by default, text if configured)
3. MongoDBLogHandler - optional, writes to MongoDB application_logs collection

The file format is controlled by settings.app_log_file_format:
- "json" (default): JSON Lines per docs/logging-standard.md
- "text": legacy comma-separated format (same as console)

Console format always stays human-readable so `docker logs <container>` is
skimmable without a JSON parser.
"""

import json
import logging
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s"


class JsonlFormatter(logging.Formatter):
    """Formats log records as JSON Lines (one JSON object per line).

    Schema is documented in docs/logging-standard.md. Every record carries
    eight mandatory fields; exception records add three more. Output is a
    single line (newlines inside message text are JSON-escaped).
    """

    def __init__(
        self,
        service_name: str,
    ) -> None:
        super().__init__()
        self._service_name = service_name

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "service": self._service_name,
            "level": record.levelname,
            "logger": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
            "process_id": record.process,
            "message": record.getMessage(),
        }

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exc_type"] = exc_type.__name__ if exc_type else "Unknown"
            payload["exc_message"] = str(exc_value) if exc_value else ""
            payload["stack_trace"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(
    service_name: str,
    log_file: Path | None = None,
) -> Path | None:
    """Configure root logger with console, file, and optional MongoDB handlers.

    Args:
        service_name: Identifies this process in log records and in the
            MongoDB application_logs collection (e.g. "registry", "auth-server").
        log_file: Explicit log file path. When ``None`` the path is derived
            from settings (``settings.log_dir / f"{service_name}.log"``).

    Returns:
        The resolved log file path, or None if file logging was skipped
        (typically due to a PermissionError on the target directory).
    """
    from ..core.config import MONGODB_BACKENDS, settings

    level = getattr(logging, settings.app_log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Console always uses the human-readable text format so `docker logs`
    # stays skimmable without a JSON parser.
    console_formatter = logging.Formatter(LOG_FORMAT)

    # File format is selected by settings (default JSONL, per issue #987).
    file_formatter: logging.Formatter
    if settings.app_log_file_format == "json":
        file_formatter = JsonlFormatter(service_name=service_name)
    else:
        file_formatter = console_formatter

    # 1. Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(console_formatter)
    root.addHandler(console)

    # 2. RotatingFileHandler
    resolved_log_file: Path | None = None
    if log_file is not None:
        resolved_log_file = log_file
    else:
        resolved_log_file = settings.log_dir / f"{service_name}.log"

    if resolved_log_file is not None:
        try:
            resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                filename=str(resolved_log_file),
                maxBytes=settings.app_log_max_bytes,
                backupCount=settings.app_log_backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            root.addHandler(file_handler)
        except (PermissionError, OSError) as exc:
            root.error(
                "Cannot write to log file %s: %s. "
                "Fix on host: sudo mkdir -p %s && sudo chown -R 1000:1000 %s. "
                "Continuing with console-only logging.",
                resolved_log_file,
                exc,
                resolved_log_file.parent,
                resolved_log_file.parent,
            )
            resolved_log_file = None

    # 3. Centralized log handler (optional, writes to MongoDB/DocumentDB)
    if settings.app_log_centralized_enabled and settings.storage_backend in MONGODB_BACKENDS:
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
            # MongoDB handler writes structured documents via its own emit();
            # the formatter set here is only used if emit() fails and falls
            # back to default formatting. Plain text is fine for that case.
            mongo_handler.setFormatter(console_formatter)
            root.addHandler(mongo_handler)
        except Exception as exc:
            root.warning(f"Failed to initialize MongoDB log handler: {exc}")

    return resolved_log_file
