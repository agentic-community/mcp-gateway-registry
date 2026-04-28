"""Unit tests for registry/utils/logging_setup.py and mongodb_log_handler.py."""

import logging
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
