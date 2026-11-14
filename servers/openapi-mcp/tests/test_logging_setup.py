"""openapi-mcp logs in the same format as the other containers (docs/logging-standard.md)."""

import json
import logging

import logging_setup
import pytest
from settings import ContainerSettings

MANDATORY_FIELDS: set[str] = {
    "timestamp",
    "service",
    "level",
    "logger",
    "filename",
    "lineno",
    "process_id",
    "message",
}


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


def _settings(
    tmp_path,
    **overrides,
) -> ContainerSettings:
    return ContainerSettings(app_log_dir=str(tmp_path), **overrides)


def _log_and_read(
    tmp_path,
    message: str = "hub built",
) -> list[str]:
    logging.getLogger("hubs").info(message)
    for handler in logging.getLogger().handlers:
        handler.flush()
    return (tmp_path / "openapi-mcp.log").read_text().splitlines()


def test_file_lines_are_standard_json_records(tmp_path):
    log_file = logging_setup.setup_container_logging(_settings(tmp_path))

    record = json.loads(_log_and_read(tmp_path)[-1])

    assert log_file == tmp_path / "openapi-mcp.log"
    assert MANDATORY_FIELDS <= set(record)
    assert (record["service"], record["level"], record["logger"], record["message"]) == (
        "openapi-mcp",
        "INFO",
        "hubs",
        "hub built",
    )


def test_console_uses_the_same_json_format(tmp_path, capsys):
    logging_setup.setup_container_logging(_settings(tmp_path))

    logging.getLogger("server").warning("container started")

    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert (record["service"], record["level"]) == ("openapi-mcp", "WARNING")


def test_exception_records_carry_the_exception_fields(tmp_path):
    logging_setup.setup_container_logging(_settings(tmp_path))
    try:
        raise ValueError("bad spec")
    except ValueError:
        logging.getLogger("conversion").exception("conversion failed")

    record = json.loads(_log_and_read(tmp_path, "after")[-2])

    assert (record["exc_type"], record["exc_message"]) == ("ValueError", "bad spec")
    assert "Traceback" in record["stack_trace"]


def test_text_format_matches_the_legacy_line(tmp_path):
    logging_setup.setup_container_logging(_settings(tmp_path, app_log_file_format="text"))

    line = _log_and_read(tmp_path)[-1]

    assert line.endswith(",INFO,hub built") and ",p" in line and "{test_logging_setup.py:" in line


def test_unwritable_directory_falls_back_to_console(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")

    log_file = logging_setup.setup_container_logging(_settings(blocker / "logs"))

    assert log_file is None
    assert len(logging.getLogger().handlers) == 1


@pytest.mark.parametrize("raw", ["", "relative/logs", "/var/log/../etc"])
def test_bad_or_empty_log_dir_uses_the_default(raw):
    assert logging_setup._log_dir(raw) == logging_setup.DEFAULT_LOG_DIR


def test_invalid_format_is_refused_at_startup():
    with pytest.raises(ValueError):
        ContainerSettings(app_log_file_format="xml")
