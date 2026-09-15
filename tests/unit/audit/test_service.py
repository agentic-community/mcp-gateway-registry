"""
Unit tests for AuditLogger service.

Tests the MongoDB-only audit logging functionality.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.audit import (
    AuditLogger,
    Identity,
    RegistryApiAccessRecord,
    Request,
    Response,
)


def make_test_record(request_id: str = "test-123") -> RegistryApiAccessRecord:
    """Create a test audit record."""
    return RegistryApiAccessRecord(
        timestamp=datetime.now(UTC),
        request_id=request_id,
        identity=Identity(
            username="testuser",
            auth_method="oauth2",
            credential_type="bearer_token",
        ),
        request=Request(
            method="GET",
            path="/api/test",
            client_ip="127.0.0.1",
        ),
        response=Response(
            status_code=200,
            duration_ms=50.5,
        ),
    )


class TestAuditLoggerInit:
    """Tests for AuditLogger initialization."""

    def test_init_with_mongodb_enabled(self):
        """Logger initializes correctly with MongoDB enabled."""
        mock_repo = MagicMock()
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=mock_repo,
        )
        assert logger.mongodb_enabled is True
        assert logger.is_open is True
        assert logger.stream_name == "test-stream"

    def test_init_with_mongodb_disabled(self):
        """Logger initializes correctly with MongoDB disabled."""
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=False,
        )
        assert logger.mongodb_enabled is False
        assert logger.is_open is False

    def test_deprecated_params_accepted(self):
        """Deprecated parameters are accepted for backward compatibility."""
        logger = AuditLogger(
            log_dir="/tmp/test",
            rotation_hours=2,
            rotation_max_mb=50,
            local_retention_hours=48,
            stream_name="test-stream",
        )
        # Should not raise, deprecated params are ignored
        assert logger.stream_name == "test-stream"


class TestLogEvent:
    """Tests for log_event method."""

    async def test_log_event_writes_to_mongodb(self):
        """Event is written to MongoDB when enabled."""
        mock_repo = AsyncMock()
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=mock_repo,
        )

        record = make_test_record()
        await logger.log_event(record)

        mock_repo.insert.assert_called_once_with(record)

    async def test_log_event_skipped_when_disabled(self):
        """Event is skipped when MongoDB is disabled."""
        mock_repo = AsyncMock()
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=False,
            audit_repository=mock_repo,
        )

        await logger.log_event(make_test_record())

        mock_repo.insert.assert_not_called()

    async def test_log_event_handles_mongodb_error(self):
        """MongoDB errors are caught and logged, not raised."""
        mock_repo = AsyncMock()
        mock_repo.insert.side_effect = Exception("MongoDB connection failed")
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=mock_repo,
        )

        # Should not raise
        await logger.log_event(make_test_record())

    async def test_dropped_durable_write_is_logged_critical(self, caplog):
        """A dropped durable audit write must be loud (CRITICAL) and alertable.

        A durable audit trail must never lose a record *silently*: when the
        durable write fails, the request is not broken (avoids a self-inflicted
        DoS) but the loss is surfaced as a distinct CRITICAL 'AUDIT RECORD
        DROPPED' event carrying identifying context (not the full record).
        """
        import logging

        mock_repo = AsyncMock()
        mock_repo.insert.side_effect = Exception("MongoDB connection failed")
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=mock_repo,
        )

        with caplog.at_level(logging.CRITICAL, logger="registry.audit.service"):
            await logger.log_event(make_test_record("req-dropped"))

        dropped = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert dropped, "expected a CRITICAL log when a durable audit write is dropped"
        message = dropped[0].getMessage()
        assert "AUDIT RECORD DROPPED" in message
        assert "req-dropped" in message

    async def test_dropped_durable_write_increments_the_integrity_counter(self):
        """A CRITICAL log line cannot be alerted on as a RATE, so the drop also
        increments a bounded counter. Without it, "audit records are being lost
        right now" is invisible to monitoring."""
        mock_repo = AsyncMock()
        mock_repo.insert.side_effect = Exception("MongoDB connection failed")
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=mock_repo,
        )

        with patch("registry.observability.meters.audit_integrity_degraded_total") as counter:
            await logger.log_event(make_test_record("req-dropped"))

        counter.add.assert_called_once_with(1, {"reason": "record_dropped"})

    async def test_successful_write_does_not_increment_the_integrity_counter(self):
        """The healthy path must stay silent, or the alert is meaningless."""
        mock_repo = AsyncMock()
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=mock_repo,
        )

        with patch("registry.observability.meters.audit_integrity_degraded_total") as counter:
            await logger.log_event(make_test_record("req-ok"))

        counter.add.assert_not_called()

    async def test_counter_labels_carry_no_unbounded_dimension(self):
        """REGRESSION: no username / request id / tenant id may become a label --
        that is a cardinality DoS on the metrics backend."""
        mock_repo = AsyncMock()
        mock_repo.insert.side_effect = Exception("MongoDB connection failed")
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=mock_repo,
        )

        with patch("registry.observability.meters.audit_integrity_degraded_total") as counter:
            await logger.log_event(make_test_record("req-unbounded"))

        attributes = counter.add.call_args.args[1]
        assert set(attributes) == {"reason"}
        assert "req-unbounded" not in attributes.values()

    async def test_metrics_failure_does_not_break_the_request_path(self):
        """A broken meter must not turn a dropped audit record into a 500."""
        mock_repo = AsyncMock()
        mock_repo.insert.side_effect = Exception("MongoDB connection failed")
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=mock_repo,
        )

        with patch("registry.observability.meters.audit_integrity_degraded_total") as counter:
            counter.add.side_effect = RuntimeError("meter exploded")

            await logger.log_event(make_test_record("req-meter-down"))

    async def test_multiple_events_logged(self):
        """Multiple events can be logged sequentially."""
        mock_repo = AsyncMock()
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=mock_repo,
        )

        for i in range(3):
            await logger.log_event(make_test_record(f"request-{i}"))

        assert mock_repo.insert.call_count == 3


class TestClose:
    """Tests for close method."""

    async def test_close_is_safe(self):
        """Close method completes without error."""
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=AsyncMock(),
        )
        # Should not raise
        await logger.close()


class TestProperties:
    """Tests for logger properties."""

    def test_current_file_path_returns_none(self):
        """current_file_path returns None (no local files)."""
        logger = AuditLogger(stream_name="test-stream")
        assert logger.current_file_path is None

    def test_is_open_with_mongodb(self):
        """is_open returns True when MongoDB is enabled and repo is set."""
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=MagicMock(),
        )
        assert logger.is_open is True

    def test_is_open_without_mongodb(self):
        """is_open returns False when MongoDB is disabled."""
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=False,
        )
        assert logger.is_open is False

    def test_is_open_without_repo(self):
        """is_open returns False when MongoDB enabled but no repo."""
        logger = AuditLogger(
            stream_name="test-stream",
            mongodb_enabled=True,
            audit_repository=None,
        )
        assert logger.is_open is False


class TestEnforceDurableAuditSink:
    """Tests for the fail-closed durable-audit-sink guard.

    A non-durable audit trail (best-effort JSON log lines) is not a dependable
    record for a repudiation-sensitive deployment. The guard must refuse to
    start by default when no durable sink is available, and only permit a
    non-durable trail on an explicit operator opt-out.
    """

    def test_durable_available_is_a_noop(self) -> None:
        """When a durable sink is available, the guard does nothing (no raise)."""
        from registry.audit.service import enforce_durable_audit_sink

        # Must not raise regardless of the require_durable setting.
        enforce_durable_audit_sink(durable_sink_available=True, require_durable=True)
        enforce_durable_audit_sink(durable_sink_available=True, require_durable=False)

    def test_no_durable_sink_fails_closed_by_default(self) -> None:
        """No durable sink + require_durable True => refuse to start (fail closed).

        This is the property the vulnerable default lacked: audit silently
        degraded to non-durable log lines instead of failing closed.
        """
        from registry.audit.service import NonDurableAuditError, enforce_durable_audit_sink

        with pytest.raises(NonDurableAuditError):
            enforce_durable_audit_sink(durable_sink_available=False, require_durable=True)

    def test_no_durable_sink_allowed_on_explicit_optout(self) -> None:
        """No durable sink + require_durable False => allowed, with a warning."""
        from registry.audit.service import enforce_durable_audit_sink

        # Explicit dev opt-out: must not raise.
        enforce_durable_audit_sink(durable_sink_available=False, require_durable=False)

    def test_optout_emits_loud_warning(self, caplog) -> None:
        """The non-durable opt-out path must warn loudly so it is never silent."""
        import logging

        from registry.audit.service import enforce_durable_audit_sink

        with caplog.at_level(logging.WARNING, logger="registry.audit.service"):
            enforce_durable_audit_sink(durable_sink_available=False, require_durable=False)

        assert any("WITHOUT a durable sink" in record.message for record in caplog.records), (
            "expected a loud warning when running non-durable"
        )


class TestRecordInstanceAttribution:
    """The audit record must carry the producing replica's instance id."""

    def test_record_accepts_instance_id(self) -> None:
        """RegistryApiAccessRecord persists the instance_id for attribution."""
        record = make_test_record()
        assert hasattr(record, "instance_id")

        attributed = RegistryApiAccessRecord(
            timestamp=datetime.now(UTC),
            request_id="req-1",
            instance_id="registry-blue-3",
            identity=Identity(
                username="testuser",
                auth_method="oauth2",
                credential_type="bearer_token",
            ),
            request=Request(method="GET", path="/api/test", client_ip="127.0.0.1"),
            response=Response(status_code=200, duration_ms=1.0),
        )
        assert attributed.instance_id == "registry-blue-3"
