"""Custom logging handler that writes log records to MongoDB.

Uses synchronous PyMongo in a background thread to avoid blocking the
async event loop. Records are buffered and flushed periodically or
when the buffer reaches a configurable size.
"""

import atexit
import logging
import socket
import threading
import time
from datetime import UTC, datetime
from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from .mongodb_connection import build_client_options, build_connection_string, build_tls_kwargs


class MongoDBLogHandler(logging.Handler):
    """Logging handler that buffers records and flushes them to MongoDB.

    A daemon thread periodically flushes the buffer. The handler also
    flushes when the buffer reaches ``buffer_size`` records.

    The target collection is ``application_logs_{namespace}`` with a TTL
    index on the ``timestamp`` field.
    """

    def __init__(
        self,
        service_name: str,
        buffer_size: int = 50,
        flush_interval: float = 5.0,
        ttl_days: int = 7,
    ):
        super().__init__()
        from ..core.config import settings

        self._service_name = service_name
        self._hostname = socket.gethostname()
        self._buffer: list[dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._ttl_days = ttl_days
        self._closed = False

        namespace = settings.documentdb_namespace
        self._collection_name = f"application_logs_{namespace}"

        self._client: MongoClient | None = None
        self._collection = None
        self._connect_error_logged = False

        self._flush_thread = threading.Thread(
            target=self._periodic_flush,
            daemon=True,
            name="mongodb-log-flusher",
        )
        self._flush_thread.start()

        atexit.register(self.close)

    def _ensure_connection(self) -> bool:
        """Lazily connect to MongoDB and ensure TTL index exists."""
        if self._collection is not None:
            return True

        try:
            from ..core.config import settings

            self._client = MongoClient(
                build_connection_string(),
                serverSelectionTimeoutMS=5000,
                **build_client_options(),
                **build_tls_kwargs(),
            )
            db = self._client[settings.documentdb_database]
            self._collection = db[self._collection_name]

            self._collection.create_index(
                "timestamp",
                expireAfterSeconds=self._ttl_days * 86400,
                background=True,
            )
            self._collection.create_index(
                [("service", 1), ("level", 1), ("timestamp", -1)],
                background=True,
            )
            self._connect_error_logged = False
            return True

        except Exception as exc:
            if not self._connect_error_logged:
                import sys

                print(
                    f"MongoDBLogHandler: failed to connect - {exc}",
                    file=sys.stderr,
                )
                self._connect_error_logged = True
            return False

    def emit(self, record: logging.LogRecord) -> None:
        if self._closed:
            return

        doc = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC),
            "hostname": self._hostname,
            "service": self._service_name,
            "level": record.levelname,
            "logger": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
            "message": self.format(record),
        }

        with self._buffer_lock:
            self._buffer.append(doc)
            should_flush = len(self._buffer) >= self._buffer_size

        if should_flush:
            self._flush()

    def _flush(self) -> None:
        """Flush buffered records to MongoDB."""
        with self._buffer_lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()

        if not self._ensure_connection():
            return

        try:
            self._collection.insert_many(batch, ordered=False)
        except PyMongoError:
            pass

    def _periodic_flush(self) -> None:
        """Background thread: flush buffer every ``flush_interval`` seconds."""
        while not self._closed:
            time.sleep(self._flush_interval)
            try:
                self._flush()
            except Exception:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._flush()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        super().close()
