"""Unit tests for registry/utils/mongodb_log_handler.py - MongoDB log handler."""

import logging
import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from registry.utils.mongodb_log_handler import EXCLUDED_LOGGERS_DEFAULT, MongoDBLogHandler


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.documentdb_namespace = "test"
    s.documentdb_database = "registry_test"
    return s


@pytest.fixture
def handler(mock_settings):
    with (
        patch("registry.utils.mongodb_log_handler.build_connection_string"),
        patch("registry.utils.mongodb_log_handler.build_client_options", return_value={}),
        patch("registry.utils.mongodb_log_handler.build_tls_kwargs", return_value={}),
        patch("registry.core.config.settings", mock_settings),
    ):
        h = MongoDBLogHandler.__new__(MongoDBLogHandler)
        logging.Handler.__init__(h)
        h._service_name = "test-service"
        h._hostname = "test-host"
        h._buffer = []
        h._buffer_lock = threading.Lock()
        h._buffer_size = 50
        h._flush_interval = 5.0
        h._ttl_days = 7
        h._excluded_loggers = EXCLUDED_LOGGERS_DEFAULT
        h._flush_failure_count = 0
        h._closed = False
        h._collection_name = "application_logs_test"
        h._client = None
        h._collection = None
        h._connect_error_logged = False
        h._flush_thread = threading.Thread(target=lambda: None, daemon=True)
        yield h
        h._closed = True


class TestExcludedLoggers:
    """Test recursion guard via _is_excluded."""

    def test_exact_match(self, handler):
        assert handler._is_excluded("pymongo") is True

    def test_child_logger_excluded(self, handler):
        assert handler._is_excluded("pymongo.collection") is True

    def test_unrelated_logger_allowed(self, handler):
        assert handler._is_excluded("registry.api.server_routes") is False

    def test_partial_name_not_excluded(self, handler):
        assert handler._is_excluded("pymongo_extra") is False

    def test_default_exclusions_present(self):
        assert "pymongo" in EXCLUDED_LOGGERS_DEFAULT
        assert "motor" in EXCLUDED_LOGGERS_DEFAULT
        assert "uvicorn.access" in EXCLUDED_LOGGERS_DEFAULT
        assert "httpx" in EXCLUDED_LOGGERS_DEFAULT
        assert "registry.utils.mongodb_log_handler" in EXCLUDED_LOGGERS_DEFAULT

    def test_custom_exclusions(self, handler):
        handler._excluded_loggers = frozenset({"myapp"})
        assert handler._is_excluded("myapp") is True
        assert handler._is_excluded("myapp.sub") is True
        assert handler._is_excluded("pymongo") is False


class TestEmit:
    """Test the emit method buffers records correctly."""

    def test_record_buffered(self, handler):
        record = logging.LogRecord(
            name="registry.api",
            level=logging.INFO,
            pathname="api.py",
            lineno=10,
            msg="Test message",
            args=None,
            exc_info=None,
        )

        handler.emit(record)
        assert len(handler._buffer) == 1
        doc = handler._buffer[0]
        assert doc["service"] == "test-service"
        assert doc["hostname"] == "test-host"
        assert doc["level"] == "INFO"
        assert doc["level_no"] == 20
        assert doc["message"] == "Test message"
        assert doc["process"] is not None
        assert isinstance(doc["timestamp"], datetime)
        assert isinstance(doc["created_at"], datetime)

    def test_excluded_logger_not_buffered(self, handler):
        record = logging.LogRecord(
            name="pymongo.collection",
            level=logging.INFO,
            pathname="collection.py",
            lineno=1,
            msg="Internal message",
            args=None,
            exc_info=None,
        )

        handler.emit(record)
        assert len(handler._buffer) == 0

    def test_closed_handler_no_buffer(self, handler):
        handler._closed = True
        record = logging.LogRecord(
            name="registry.api",
            level=logging.ERROR,
            pathname="api.py",
            lineno=5,
            msg="Should be ignored",
            args=None,
            exc_info=None,
        )

        handler.emit(record)
        assert len(handler._buffer) == 0

    def test_buffer_flush_on_size(self, handler):
        handler._buffer_size = 2
        mock_collection = MagicMock()
        handler._collection = mock_collection

        for i in range(2):
            record = logging.LogRecord(
                name="registry.api",
                level=logging.INFO,
                pathname="api.py",
                lineno=i,
                msg=f"Message {i}",
                args=None,
                exc_info=None,
            )
            handler.emit(record)

        mock_collection.insert_many.assert_called_once()
        assert len(handler._buffer) == 0


class TestFlush:
    """Test the _flush method."""

    def test_flush_empty_buffer_noop(self, handler):
        handler._flush()
        assert handler._collection is None

    def test_flush_failure_increments_counter(self, handler):
        from pymongo.errors import PyMongoError

        mock_collection = MagicMock()
        mock_collection.insert_many.side_effect = PyMongoError("write error")
        handler._collection = mock_collection
        handler._buffer = [{"message": "test"}]

        with patch("registry.core.metrics.APP_LOG_FLUSH_FAILURES") as mock_metric:
            handler._flush()

        assert handler._flush_failure_count == 1
        mock_metric.labels.assert_called_once_with(service="test-service")


class TestFlushFailureCount:
    """Test the flush_failure_count property."""

    def test_initial_count_zero(self, handler):
        assert handler.flush_failure_count == 0

    def test_count_reflects_failures(self, handler):
        handler._flush_failure_count = 5
        assert handler.flush_failure_count == 5


class TestDocumentSchema:
    """Test that emitted documents match the expected schema."""

    def test_document_has_all_fields(self, handler):
        record = logging.LogRecord(
            name="registry.main",
            level=logging.WARNING,
            pathname="main.py",
            lineno=42,
            msg="Test warning",
            args=None,
            exc_info=None,
        )

        handler.emit(record)
        doc = handler._buffer[0]

        expected_fields = {
            "timestamp",
            "hostname",
            "service",
            "level",
            "level_no",
            "logger",
            "filename",
            "lineno",
            "process",
            "message",
            "created_at",
        }
        assert set(doc.keys()) == expected_fields

    def test_level_no_matches_record(self, handler):
        for level, expected_no in [
            (logging.DEBUG, 10),
            (logging.INFO, 20),
            (logging.WARNING, 30),
            (logging.ERROR, 40),
            (logging.CRITICAL, 50),
        ]:
            handler._buffer.clear()
            record = logging.LogRecord(
                name="registry.test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="msg",
                args=None,
                exc_info=None,
            )
            handler.emit(record)
            assert handler._buffer[0]["level_no"] == expected_no


class TestClose:
    """Test handler close behavior."""

    def test_close_flushes_remaining(self, handler):
        mock_collection = MagicMock()
        handler._collection = mock_collection
        handler._client = MagicMock()
        handler._buffer = [{"message": "final"}]

        handler.close()

        mock_collection.insert_many.assert_called_once()
        handler._client.close.assert_called_once()
        assert handler._closed is True

    def test_double_close_safe(self, handler):
        handler._client = MagicMock()
        handler.close()
        handler.close()
        assert handler._closed is True
