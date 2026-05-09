"""Logging setup for the mcpgw container (issue #987).

Provides console + file handlers with the same JSONL format as registry and
auth-server. Does not depend on registry/* modules or MongoDB; configuration
comes from environment variables (APP_LOG_DIR, APP_LOG_FILE_FORMAT, etc.).

Why duplicate instead of importing registry.utils.logging_setup?
The mcpgw image is built from docker/Dockerfile.mcp-server and does not copy
the registry/ source tree. It also does not carry a MongoDB client. A small
standalone module with the same schema keeps the container image minimal and
avoids cross-service import paths.
"""

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

CONSOLE_FORMAT = (
    "%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s"
)
DEFAULT_LOG_DIR = "/var/log/containers/ai-registry"
SERVICE_NAME = "ai-registry-tools"


class JsonlFormatter(logging.Formatter):
    """JSONL formatter matching docs/logging-standard.md.

    Kept in sync with registry.utils.logging_setup.JsonlFormatter. Both emit
    the same 8 mandatory fields plus 3 exception fields.
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


def _resolve_log_dir() -> Path:
    """Read APP_LOG_DIR from env, validate, fall back to default."""
    raw = os.getenv("APP_LOG_DIR", "").strip()
    if not raw:
        return Path(DEFAULT_LOG_DIR)
    if not raw.startswith("/"):
        # Invalid config: log a warning and use the default instead of crashing.
        logging.getLogger(__name__).warning(
            "APP_LOG_DIR=%r is not absolute; falling back to %s",
            raw,
            DEFAULT_LOG_DIR,
        )
        return Path(DEFAULT_LOG_DIR)
    if ".." in Path(raw).parts:
        logging.getLogger(__name__).warning(
            "APP_LOG_DIR=%r contains '..'; falling back to %s",
            raw,
            DEFAULT_LOG_DIR,
        )
        return Path(DEFAULT_LOG_DIR)
    return Path(raw)


def _resolve_file_format() -> str:
    """Read APP_LOG_FILE_FORMAT from env; accept only 'json' or 'text'."""
    raw = os.getenv("APP_LOG_FILE_FORMAT", "json").strip().lower()
    if raw not in ("json", "text"):
        logging.getLogger(__name__).warning(
            "APP_LOG_FILE_FORMAT=%r is invalid; falling back to 'json'",
            raw,
        )
        return "json"
    return raw


def setup_mcpgw_logging() -> Path | None:
    """Configure root logger for mcpgw.

    Returns:
        The resolved log file path, or None if file logging could not be
        initialized (PermissionError, missing host directory, etc.). In the
        None case the process continues with console-only logging.
    """
    log_dir = _resolve_log_dir()
    log_file = log_dir / f"{SERVICE_NAME}.log"
    log_level = getattr(
        logging,
        os.getenv("APP_LOG_LEVEL", "INFO").upper(),
        logging.INFO,
    )
    max_bytes = int(os.getenv("APP_LOG_MAX_BYTES", str(50 * 1024 * 1024)))
    backup_count = int(os.getenv("APP_LOG_BACKUP_COUNT", "5"))
    file_format = _resolve_file_format()

    root = logging.getLogger()
    root.setLevel(log_level)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    console_formatter = logging.Formatter(CONSOLE_FORMAT)
    file_formatter: logging.Formatter
    if file_format == "json":
        file_formatter = JsonlFormatter(service_name=SERVICE_NAME)
    else:
        file_formatter = console_formatter

    # Console handler (always on, human-readable)
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(console_formatter)
    root.addHandler(console)

    # Rotating file handler (JSONL by default)
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)
        return log_file
    except (PermissionError, OSError) as exc:
        root.error(
            "Cannot write to log file %s: %s. "
            "Fix on host: sudo mkdir -p %s && sudo chown -R 1000:1000 %s. "
            "Continuing with console-only logging.",
            log_file,
            exc,
            log_dir,
            log_dir,
        )
        return None
