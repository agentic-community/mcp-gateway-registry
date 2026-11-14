"""Test runs must never write to the host's real application log directory.

servers/mcpgw/server.py sets up file logging when it is imported, so before
tests/conftest.py pointed APP_LOG_DIR at a temp directory, a test run on a host
with a writable /var/log/containers/ai-registry appended every later test's
records to the real ai-registry-tools.log.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

PRODUCTION_LOG_DIR: Path = Path("/var/log/containers/ai-registry")
MCPGW_DIR: Path = Path(__file__).resolve().parents[2] / "servers" / "mcpgw"


def _file_handler_paths() -> list[Path]:
    return [
        Path(h.baseFilename)
        for h in logging.getLogger().handlers
        if isinstance(h, RotatingFileHandler)
    ]


def test_app_log_dir_is_a_temp_directory():
    log_dir = Path(os.environ["APP_LOG_DIR"])

    assert log_dir.is_dir()
    assert PRODUCTION_LOG_DIR not in [log_dir, *log_dir.parents]


def test_mcpgw_logging_setup_writes_under_the_test_directory(monkeypatch):
    monkeypatch.syspath_prepend(str(MCPGW_DIR))
    logging_setup = pytest.importorskip("logging_setup")
    if not hasattr(logging_setup, "setup_mcpgw_logging"):
        pytest.skip("another logging_setup module is first on sys.path")
    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        log_file = logging_setup.setup_mcpgw_logging()

        assert log_file is not None
        assert log_file.parent == Path(os.environ["APP_LOG_DIR"])
        assert all(PRODUCTION_LOG_DIR not in p.parents for p in _file_handler_paths())
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        for handler in saved:
            root.addHandler(handler)
        sys.modules.pop("logging_setup", None)
