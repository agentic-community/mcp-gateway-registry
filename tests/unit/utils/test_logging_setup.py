"""Unit tests for registry/utils/logging_setup.py and mongodb_log_handler.py."""

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from unittest.mock import patch

# =============================================================================
# LOGGING SETUP TESTS
# =============================================================================


class TestSetupLogging:
    """Test the shared setup_logging function."""

    def test_creates_console_handler(self, tmp_path):
        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.app_log_level = "INFO"
            mock_settings.app_log_max_bytes = 50 * 1024 * 1024
            mock_settings.app_log_backup_count = 5
            mock_settings.app_log_centralized_enabled = False
            mock_settings.log_dir = tmp_path

            from registry.utils.logging_setup import setup_logging

            setup_logging(service_name="test-service", log_file=tmp_path / "test.log")

            root = logging.getLogger()
            handler_types = [type(h) for h in root.handlers]
            assert logging.StreamHandler in handler_types

    def test_creates_rotating_file_handler(self, tmp_path):
        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.app_log_level = "INFO"
            mock_settings.app_log_max_bytes = 50 * 1024 * 1024
            mock_settings.app_log_backup_count = 5
            mock_settings.app_log_centralized_enabled = False
            mock_settings.log_dir = tmp_path

            from registry.utils.logging_setup import setup_logging

            log_path = setup_logging(
                service_name="test-service",
                log_file=tmp_path / "test.log",
            )

            assert log_path == tmp_path / "test.log"

            root = logging.getLogger()
            handler_types = [type(h) for h in root.handlers]
            assert RotatingFileHandler in handler_types

    def test_rotating_handler_uses_settings(self, tmp_path):
        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.app_log_level = "WARNING"
            mock_settings.app_log_max_bytes = 10 * 1024 * 1024
            mock_settings.app_log_backup_count = 3
            mock_settings.app_log_centralized_enabled = False
            mock_settings.log_dir = tmp_path

            from registry.utils.logging_setup import setup_logging

            setup_logging(service_name="test-service", log_file=tmp_path / "test.log")

            root = logging.getLogger()
            rotating_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
            assert len(rotating_handlers) == 1
            assert rotating_handlers[0].maxBytes == 10 * 1024 * 1024
            assert rotating_handlers[0].backupCount == 3

    def test_default_log_file_path(self, tmp_path):
        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.app_log_level = "INFO"
            mock_settings.app_log_max_bytes = 50 * 1024 * 1024
            mock_settings.app_log_backup_count = 5
            mock_settings.app_log_centralized_enabled = False
            mock_settings.log_dir = tmp_path

            from registry.utils.logging_setup import setup_logging

            log_path = setup_logging(service_name="registry")

            assert log_path == tmp_path / "registry.log"

    def test_mongodb_handler_not_added_when_disabled(self, tmp_path):
        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.app_log_level = "INFO"
            mock_settings.app_log_max_bytes = 50 * 1024 * 1024
            mock_settings.app_log_backup_count = 5
            mock_settings.app_log_centralized_enabled = False
            mock_settings.log_dir = tmp_path

            from registry.utils.logging_setup import setup_logging

            setup_logging(service_name="test", log_file=tmp_path / "test.log")

            root = logging.getLogger()
            from registry.utils.mongodb_log_handler import MongoDBLogHandler

            mongo_handlers = [h for h in root.handlers if isinstance(h, MongoDBLogHandler)]
            assert len(mongo_handlers) == 0

    def test_mongodb_handler_skipped_for_file_backend(self, tmp_path):
        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.app_log_level = "INFO"
            mock_settings.app_log_max_bytes = 50 * 1024 * 1024
            mock_settings.app_log_backup_count = 5
            mock_settings.app_log_centralized_enabled = True
            mock_settings.storage_backend = "file"
            mock_settings.log_dir = tmp_path

            from registry.utils.logging_setup import setup_logging

            setup_logging(service_name="test", log_file=tmp_path / "test.log")

            root = logging.getLogger()
            from registry.utils.mongodb_log_handler import MongoDBLogHandler

            mongo_handlers = [h for h in root.handlers if isinstance(h, MongoDBLogHandler)]
            assert len(mongo_handlers) == 0

    def test_clears_existing_handlers(self, tmp_path):
        root = logging.getLogger()
        dummy_handler = logging.StreamHandler()
        root.addHandler(dummy_handler)
        initial_count = len(root.handlers)

        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.app_log_level = "INFO"
            mock_settings.app_log_max_bytes = 50 * 1024 * 1024
            mock_settings.app_log_backup_count = 5
            mock_settings.app_log_centralized_enabled = False
            mock_settings.log_dir = tmp_path

            from registry.utils.logging_setup import setup_logging

            setup_logging(service_name="test", log_file=tmp_path / "test.log")

            # Should have exactly 2 handlers: console + file
            assert len(root.handlers) == 2


# =============================================================================
# JSONL FORMATTER TESTS (Issue #987)
# =============================================================================


class TestJsonlFormatter:
    """Tests for the JsonlFormatter added in issue #987."""

    def _make_record(
        self,
        name: str = "test.logger",
        level: int = logging.INFO,
        msg: str = "hello",
        args: tuple = (),
        exc_info=None,
    ) -> logging.LogRecord:
        return logging.LogRecord(
            name=name,
            level=level,
            pathname="/app/example.py",
            lineno=42,
            msg=msg,
            args=args,
            exc_info=exc_info,
        )

    def test_mandatory_fields_present(self):
        from registry.utils.logging_setup import JsonlFormatter

        formatter = JsonlFormatter(service_name="registry")
        rec = self._make_record(name="registry.core.config", msg="Starting on port %d", args=(8000,))
        out = formatter.format(rec)
        payload = json.loads(out)

        assert payload["service"] == "registry"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "registry.core.config"
        assert payload["filename"] == "example.py"
        assert payload["lineno"] == 42
        assert payload["message"] == "Starting on port 8000"
        assert "timestamp" in payload
        assert isinstance(payload["process_id"], int)
        assert {"exc_type", "exc_message", "stack_trace"}.isdisjoint(payload)

    def test_timestamp_iso_utc_with_offset(self):
        from registry.utils.logging_setup import JsonlFormatter

        formatter = JsonlFormatter(service_name="auth-server")
        out = formatter.format(self._make_record())
        payload = json.loads(out)
        dt = datetime.fromisoformat(payload["timestamp"])
        assert dt.tzinfo is not None

    def test_exception_fields_populated(self):
        from registry.utils.logging_setup import JsonlFormatter

        formatter = JsonlFormatter(service_name="registry")
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        rec = self._make_record(level=logging.ERROR, msg="oh no", exc_info=exc_info)
        out = formatter.format(rec)
        payload = json.loads(out)

        assert payload["exc_type"] == "ValueError"
        assert payload["exc_message"] == "boom"
        assert "Traceback" in payload["stack_trace"]

    def test_output_is_single_line(self):
        from registry.utils.logging_setup import JsonlFormatter

        formatter = JsonlFormatter(service_name="registry")
        rec = self._make_record(msg="line1\nline2")
        out = formatter.format(rec)
        # Raw output must be a single line; embedded newlines are JSON-escaped.
        assert "\n" not in out
        payload = json.loads(out)
        assert payload["message"] == "line1\nline2"

    def test_utf8_preserved(self):
        from registry.utils.logging_setup import JsonlFormatter

        formatter = JsonlFormatter(service_name="registry")
        rec = self._make_record(msg="café résumé 中文")
        out = formatter.format(rec)
        payload = json.loads(out)
        assert payload["message"] == "café résumé 中文"


class TestSetupLoggingFileFormatSelection:
    """Tests that APP_LOG_FILE_FORMAT selects the file handler formatter."""

    def test_json_format_uses_jsonl_formatter(self, tmp_path):
        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.app_log_level = "INFO"
            mock_settings.app_log_max_bytes = 50 * 1024 * 1024
            mock_settings.app_log_backup_count = 5
            mock_settings.app_log_centralized_enabled = False
            mock_settings.app_log_file_format = "json"
            mock_settings.log_dir = tmp_path

            from registry.utils.logging_setup import JsonlFormatter, setup_logging

            setup_logging(service_name="registry", log_file=tmp_path / "registry.log")

            root = logging.getLogger()
            rotating = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
            assert len(rotating) == 1
            assert isinstance(rotating[0].formatter, JsonlFormatter)

    def test_text_format_uses_plain_formatter(self, tmp_path):
        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.app_log_level = "INFO"
            mock_settings.app_log_max_bytes = 50 * 1024 * 1024
            mock_settings.app_log_backup_count = 5
            mock_settings.app_log_centralized_enabled = False
            mock_settings.app_log_file_format = "text"
            mock_settings.log_dir = tmp_path

            from registry.utils.logging_setup import JsonlFormatter, setup_logging

            setup_logging(service_name="registry", log_file=tmp_path / "registry.log")

            root = logging.getLogger()
            rotating = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
            assert len(rotating) == 1
            # Text mode must NOT use JsonlFormatter; it should be plain logging.Formatter.
            assert not isinstance(rotating[0].formatter, JsonlFormatter)
            assert isinstance(rotating[0].formatter, logging.Formatter)

    def test_jsonl_output_is_parseable_end_to_end(self, tmp_path):
        log_path = tmp_path / "registry.log"

        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.app_log_level = "INFO"
            mock_settings.app_log_max_bytes = 50 * 1024 * 1024
            mock_settings.app_log_backup_count = 5
            mock_settings.app_log_centralized_enabled = False
            mock_settings.app_log_file_format = "json"
            mock_settings.log_dir = tmp_path

            from registry.utils.logging_setup import setup_logging

            setup_logging(service_name="registry", log_file=log_path)

            logging.getLogger("registry.test").info("hello world")

            for handler in logging.getLogger().handlers:
                handler.flush()

        content = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(content) >= 1
        payload = json.loads(content[-1])
        assert payload["service"] == "registry"
        assert payload["message"] == "hello world"
        assert payload["level"] == "INFO"


# =============================================================================
# MONGODB LOG HANDLER TESTS
# =============================================================================


class TestMongoDBLogHandler:
    """Test the MongoDBLogHandler class."""

    def test_emit_buffers_record(self):
        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.documentdb_namespace = "test"
            mock_settings.documentdb_host = "localhost"
            mock_settings.documentdb_port = 27017
            mock_settings.documentdb_use_iam = False
            mock_settings.documentdb_username = None
            mock_settings.documentdb_password = None
            mock_settings.documentdb_use_tls = False
            mock_settings.documentdb_tls_ca_file = ""
            mock_settings.documentdb_direct_connection = True
            mock_settings.documentdb_database = "test_db"
            mock_settings.storage_backend = "mongodb-ce"

            from registry.utils.mongodb_log_handler import MongoDBLogHandler

            handler = MongoDBLogHandler(
                service_name="test-service",
                buffer_size=100,
                flush_interval=999,
                ttl_days=7,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="test message",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

            assert len(handler._buffer) == 1
            assert handler._buffer[0]["service"] == "test-service"
            assert handler._buffer[0]["level"] == "INFO"
            assert handler._buffer[0]["message"] == "test message"

            handler._closed = True

    def test_emit_ignored_when_closed(self):
        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.documentdb_namespace = "test"
            mock_settings.documentdb_host = "localhost"
            mock_settings.documentdb_port = 27017
            mock_settings.documentdb_use_iam = False
            mock_settings.documentdb_username = None
            mock_settings.documentdb_password = None
            mock_settings.documentdb_use_tls = False
            mock_settings.documentdb_tls_ca_file = ""
            mock_settings.documentdb_direct_connection = True
            mock_settings.documentdb_database = "test_db"
            mock_settings.storage_backend = "mongodb-ce"

            from registry.utils.mongodb_log_handler import MongoDBLogHandler

            handler = MongoDBLogHandler(
                service_name="test",
                buffer_size=100,
                flush_interval=999,
                ttl_days=7,
            )
            handler._closed = True

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="ignored",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

            assert len(handler._buffer) == 0

    def test_flush_triggers_at_buffer_size(self):
        with (
            patch("registry.core.config.settings") as mock_settings,
            patch("registry.utils.mongodb_log_handler.MongoDBLogHandler._flush") as mock_flush,
        ):
            mock_settings.documentdb_namespace = "test"
            mock_settings.documentdb_host = "localhost"
            mock_settings.documentdb_port = 27017
            mock_settings.documentdb_use_iam = False
            mock_settings.documentdb_username = None
            mock_settings.documentdb_password = None
            mock_settings.documentdb_use_tls = False
            mock_settings.documentdb_tls_ca_file = ""
            mock_settings.documentdb_direct_connection = True
            mock_settings.documentdb_database = "test_db"
            mock_settings.storage_backend = "mongodb-ce"

            from registry.utils.mongodb_log_handler import MongoDBLogHandler

            handler = MongoDBLogHandler(
                service_name="test",
                buffer_size=2,
                flush_interval=999,
                ttl_days=7,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))

            for i in range(2):
                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="test.py",
                    lineno=1,
                    msg=f"msg-{i}",
                    args=(),
                    exc_info=None,
                )
                handler.emit(record)

            mock_flush.assert_called()
            handler._closed = True
