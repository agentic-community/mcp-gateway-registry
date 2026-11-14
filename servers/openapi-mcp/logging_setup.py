"""Logging for the openapi-mcp container, in the format of docs/logging-standard.md.

The JSON line format comes from ``registry.utils.logging_setup.JsonlFormatter``,
the class the registry and the auth server use, so the fields cannot drift.
The registry's ``setup_logging()`` reads the registry ``Settings`` (which need
``SECRET_KEY`` and MongoDB), and this container never loads those, so the
handlers are set up here from ``ContainerSettings``. The behavior matches
``servers/mcpgw/logging_setup.py``: a console handler, and a rotating file
handler that falls back to console-only when the directory is not writable.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from settings import (
    ContainerSettings,
    LogFormat,
)

from registry.utils.logging_setup import (
    LOG_FORMAT,
    JsonlFormatter,
)

SERVICE_NAME: str = "openapi-mcp"
DEFAULT_LOG_DIR: Path = Path("/var/log/containers/ai-registry")

logger = logging.getLogger(__name__)


def _log_dir(
    raw: str,
) -> Path:
    """APP_LOG_DIR as a path; the default when empty, relative, or containing '..'."""
    value = raw.strip()
    if not value:
        return DEFAULT_LOG_DIR
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        logger.warning(
            f"APP_LOG_DIR={value!r} must be absolute without '..'; using {DEFAULT_LOG_DIR}"
        )
        return DEFAULT_LOG_DIR
    return path


def _formatter(
    kind: LogFormat,
) -> logging.Formatter:
    if kind == "json":
        return JsonlFormatter(service_name=SERVICE_NAME)
    return logging.Formatter(LOG_FORMAT)


def _add_file_handler(
    root: logging.Logger,
    settings: ContainerSettings,
    level: int,
) -> Path | None:
    """Add the rotating file handler, or log why not and return None."""
    log_dir = _log_dir(settings.app_log_dir)
    log_file = log_dir / f"{SERVICE_NAME}.log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=settings.app_log_max_bytes,
            backupCount=settings.app_log_backup_count,
            encoding="utf-8",
        )
    except OSError as e:
        root.warning(
            f"Cannot write to log file {log_file} ({type(e).__name__}: {e}); "
            f"logging to the console only. On the host: sudo mkdir -p {log_dir} && "
            f"sudo chown -R 1000:1000 {log_dir}"
        )
        return None
    handler.setLevel(level)
    handler.setFormatter(_formatter(settings.app_log_file_format))
    root.addHandler(handler)
    return log_file


def setup_container_logging(
    settings: ContainerSettings,
) -> Path | None:
    """Replace the root handlers with the standard console and file handlers.

    Returns:
        The log file path, or None when only console logging could be set up.
    """
    level = getattr(logging, settings.app_log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(_formatter(settings.app_log_console_format))
    root.addHandler(console)
    return _add_file_handler(root, settings, level)
