"""Tests for data retention and cleanup functionality."""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from app.core.retention import (
    RetentionPolicy,
    RetentionManager,
    ALLOWED_TABLE_NAMES,
    ALLOWED_TIMESTAMP_COLUMNS,
    MIN_RETENTION_DAYS,
    MAX_RETENTION_DAYS,
    _enable_test_tables,
    _disable_test_tables,
    _validate_table_name,
    _validate_timestamp_column,
    _validate_identifier,
    _validate_retention_days,
)
from app.storage.database import MetricsStorage
from app.storage.migrations import MigrationManager
import aiosqlite


@pytest.fixture(autouse=True)
def enable_test_tables():
    """Enable test table names for all tests in this module."""
    _enable_test_tables()
    yield
    _disable_test_tables()


@pytest_asyncio.fixture
async def temp_db():
    """Create temporary database for testing."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        # Initialize database with schema using migrations
        migration_manager = MigrationManager(db_path)
        await migration_manager.migrate_up()

        # Add sample data
        async with aiosqlite.connect(db_path) as db:
            # Create test tables
            await db.execute("""
                CREATE TABLE IF NOT EXISTS test_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    value REAL NOT NULL
                )
            """)

            # Insert test data with various timestamps
            now = datetime.now()
            await db.execute(
                "INSERT INTO test_metrics (created_at, value) VALUES (?, ?)",
                ((now - timedelta(days=100)).isoformat(), 1.0),
            )
            await db.execute(
                "INSERT INTO test_metrics (created_at, value) VALUES (?, ?)",
                ((now - timedelta(days=50)).isoformat(), 2.0),
            )
            await db.execute(
                "INSERT INTO test_metrics (created_at, value) VALUES (?, ?)",
                ((now - timedelta(days=10)).isoformat(), 3.0),
            )
            await db.execute(
                "INSERT INTO test_metrics (created_at, value) VALUES (?, ?)", (now.isoformat(), 4.0)
            )
            await db.commit()

        yield db_path
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


class TestRetentionPolicy:
    """Test retention policy functionality."""

    def test_policy_creation(self):
        """Test retention policy creation."""
        policy = RetentionPolicy(table_name="metrics", retention_days=90, is_active=True)

        assert policy.table_name == "metrics"
        assert policy.retention_days == 90
        assert policy.is_active is True
        assert policy.timestamp_column == "created_at"

    def test_custom_cleanup_query(self):
        """Test custom cleanup query."""
        custom_query = "DELETE FROM metrics WHERE timestamp < datetime('now', '-30 days')"
        policy = RetentionPolicy(
            table_name="metrics", retention_days=30, cleanup_query=custom_query
        )

        query, params = policy.get_cleanup_query()
        assert query == custom_query
        assert params == ()

    def test_default_cleanup_query(self):
        """Test default cleanup query generation."""
        policy = RetentionPolicy(table_name="metrics", retention_days=90)

        query, params = policy.get_cleanup_query()
        assert "DELETE FROM metrics" in query
        assert "created_at < ?" in query
        assert len(params) == 1
        # Verify param is an ISO formatted date string
        assert isinstance(params[0], str)

    def test_count_query(self):
        """Test count query generation."""
        policy = RetentionPolicy(table_name="metrics", retention_days=30)

        query, params = policy.get_count_query()
        assert "SELECT COUNT(*)" in query
        assert "FROM metrics" in query
        assert "created_at < ?" in query
        assert len(params) == 1
        # Verify param is an ISO formatted date string
        assert isinstance(params[0], str)


class TestRetentionManager:
    """Test retention manager functionality."""

    @pytest_asyncio.fixture
    async def manager(self, temp_db):
        """Create retention manager with temporary database."""
        manager = RetentionManager()
        manager.storage.db_path = temp_db
        return manager

    @pytest.mark.asyncio
    async def test_load_default_policies(self, manager):
        """Test loading default policies."""
        assert len(manager.policies) > 0
        assert "metrics" in manager.policies
        assert "auth_metrics" in manager.policies
        assert manager.policies["metrics"].retention_days == 90

    @pytest.mark.asyncio
    async def test_update_policy(self, manager):
        """Test updating retention policy."""
        await manager.update_policy("test_table", 60, True)

        assert "test_table" in manager.policies
        assert manager.policies["test_table"].retention_days == 60
        assert manager.policies["test_table"].is_active is True

    @pytest.mark.asyncio
    async def test_get_cleanup_preview(self, manager, temp_db):
        """Test cleanup preview functionality."""
        # Add test policy
        manager.policies["test_metrics"] = RetentionPolicy(
            table_name="test_metrics", retention_days=30
        )

        preview = await manager.get_cleanup_preview("test_metrics")

        assert "test_metrics" in preview
        preview_data = preview["test_metrics"]
        assert "retention_days" in preview_data
        assert "records_to_delete" in preview_data
        assert "total_records" in preview_data
        assert preview_data["retention_days"] == 30
        assert preview_data["total_records"] == 4  # From test data

        # Should have 2 records older than 30 days
        assert preview_data["records_to_delete"] == 2

    @pytest.mark.asyncio
    async def test_cleanup_table_dry_run(self, manager, temp_db):
        """Test table cleanup in dry run mode."""
        # Add test policy
        manager.policies["test_metrics"] = RetentionPolicy(
            table_name="test_metrics", retention_days=30
        )

        result = await manager.cleanup_table("test_metrics", dry_run=True)

        assert result["table"] == "test_metrics"
        assert result["status"] == "dry_run"
        assert result["records_would_delete"] == 2

        # Verify no records were actually deleted
        async with aiosqlite.connect(temp_db) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM test_metrics")
            count = (await cursor.fetchone())[0]
            assert count == 4

    @pytest.mark.asyncio
    async def test_cleanup_table_actual(self, manager, temp_db):
        """Test actual table cleanup."""
        # Add test policy
        manager.policies["test_metrics"] = RetentionPolicy(
            table_name="test_metrics", retention_days=30
        )

        result = await manager.cleanup_table("test_metrics", dry_run=False)

        assert result["table"] == "test_metrics"
        assert result["status"] == "completed"
        assert result["records_deleted"] == 2

        # Verify records were actually deleted
        async with aiosqlite.connect(temp_db) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM test_metrics")
            count = (await cursor.fetchone())[0]
            assert count == 2

    @pytest.mark.asyncio
    async def test_cleanup_inactive_policy(self, manager):
        """Test cleanup with inactive policy."""
        manager.policies["test_table"] = RetentionPolicy(
            table_name="test_table", retention_days=30, is_active=False
        )

        result = await manager.cleanup_table("test_table", dry_run=False)

        assert result["status"] == "skipped"
        assert result["reason"] == "policy_inactive"

    @pytest.mark.asyncio
    async def test_cleanup_no_policy(self, manager):
        """Test cleanup for table without policy."""
        with pytest.raises(ValueError, match="No retention policy found"):
            await manager.cleanup_table("nonexistent_table")

    @pytest.mark.asyncio
    async def test_cleanup_all_tables(self, manager, temp_db):
        """Test cleanup of all tables."""
        # Add test policy
        manager.policies["test_metrics"] = RetentionPolicy(
            table_name="test_metrics", retention_days=30
        )

        result = await manager.cleanup_all_tables(dry_run=True)

        assert result["operation"] == "dry_run"
        assert "test_metrics" in result["table_results"]
        assert result["table_results"]["test_metrics"]["status"] == "dry_run"

    @pytest.mark.asyncio
    async def test_get_table_stats(self, manager, temp_db):
        """Test getting table statistics."""
        stats = await manager.get_table_stats()

        # Check that we get stats for some tables
        assert len(stats) > 0

        # Check that real metrics tables have proper stats
        if "metrics" in stats and "error" not in stats["metrics"]:
            assert "record_count" in stats["metrics"]
            assert "has_retention_policy" in stats["metrics"]

        # For test table that might have errors, just verify it exists
        assert "test_metrics" in stats

    @pytest.mark.asyncio
    async def test_get_database_size(self, manager):
        """Test getting database size information."""
        size_info = await manager.get_database_size()

        assert "main_db_bytes" in size_info
        assert "main_db_mb" in size_info
        assert "total_bytes" in size_info
        assert "page_count" in size_info
        assert "page_size" in size_info
        assert size_info["main_db_bytes"] > 0

    @pytest.mark.asyncio
    async def test_save_and_load_policies(self, manager, temp_db):
        """Test saving and loading policies to/from database."""
        # Add custom policy
        manager.policies["custom_table"] = RetentionPolicy(
            table_name="custom_table", retention_days=120, is_active=True
        )

        # Save policies
        await manager.save_policies_to_database()

        # Create new manager and load policies
        new_manager = RetentionManager()
        new_manager.storage.db_path = temp_db
        await new_manager.load_policies_from_database()

        # Verify custom policy was loaded
        assert "custom_table" in new_manager.policies
        assert new_manager.policies["custom_table"].retention_days == 120
        assert new_manager.policies["custom_table"].is_active is True


class TestRetentionIntegration:
    """Integration tests for retention system."""

    @pytest.mark.asyncio
    async def test_end_to_end_cleanup(self, temp_db):
        """Test complete end-to-end cleanup process."""
        manager = RetentionManager()
        manager.storage.db_path = temp_db

        # Add metrics with old timestamps
        async with aiosqlite.connect(temp_db) as db:
            old_date = (datetime.now() - timedelta(days=120)).isoformat()
            recent_date = (datetime.now() - timedelta(days=10)).isoformat()

            await db.execute(
                "INSERT INTO metrics (request_id, service, metric_type, value, timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("req_1", "test", "auth_request", 1.0, old_date, old_date),
            )
            await db.execute(
                "INSERT INTO metrics (request_id, service, metric_type, value, timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("req_2", "test", "auth_request", 1.0, recent_date, recent_date),
            )
            await db.commit()

        # Preview cleanup
        preview = await manager.get_cleanup_preview("metrics")
        assert preview["metrics"]["records_to_delete"] == 1

        # Run actual cleanup
        result = await manager.cleanup_table("metrics", dry_run=False)
        assert result["records_deleted"] == 1

        # Verify only recent record remains
        async with aiosqlite.connect(temp_db) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM metrics")
            count = (await cursor.fetchone())[0]
            assert count == 1

            cursor = await db.execute("SELECT request_id FROM metrics")
            remaining_id = (await cursor.fetchone())[0]
            assert remaining_id == "req_2"


class TestSQLInjectionPrevention:
    """Security tests for SQL injection prevention.

    These tests verify that the retention system properly rejects
    malicious input that could be used for SQL injection attacks.
    CVE reference: Security vulnerability reported 2026-02-05.
    """

    def test_table_name_with_sql_injection_rejected(self):
        """Test that SQL injection attempts in table_name are rejected."""
        # Test cases from the security report
        malicious_table_names = [
            "metrics WHERE 1=1--",
            "metrics; DROP TABLE api_keys;--",
            "metrics UNION SELECT * FROM sqlite_master--",
            "metrics) INSERT INTO api_keys VALUES ('hack');--",
            "metrics' OR '1'='1",
            'metrics"; DELETE FROM metrics; --',
            "metrics`; DROP TABLE users; --",
        ]

        for malicious_name in malicious_table_names:
            with pytest.raises(ValueError, match="Invalid table name"):
                _validate_table_name(malicious_name)

    def test_table_name_with_spaces_rejected(self):
        """Test that table names with spaces are rejected."""
        with pytest.raises(ValueError, match="Invalid table name"):
            _validate_table_name("metrics table")

    def test_table_name_with_special_chars_rejected(self):
        """Test that table names with special characters are rejected."""
        invalid_names = [
            "metrics;",
            "metrics--",
            "metrics/*",
            "metrics'",
            'metrics"',
            "metrics`",
            "metrics(",
            "metrics)",
            "metrics=",
        ]

        for invalid_name in invalid_names:
            with pytest.raises(ValueError, match="Invalid table name"):
                _validate_table_name(invalid_name)

    def test_timestamp_column_with_sql_injection_rejected(self):
        """Test that SQL injection in timestamp_column is rejected."""
        malicious_columns = [
            "created_at; DROP TABLE--",
            "created_at UNION SELECT--",
            "created_at' OR '1'='1",
        ]

        for malicious_col in malicious_columns:
            with pytest.raises(ValueError, match="Invalid timestamp column"):
                _validate_timestamp_column(malicious_col)

    def test_identifier_validation_rejects_invalid_patterns(self):
        """Test that identifier validation rejects invalid SQL identifiers."""
        invalid_identifiers = [
            "",  # Empty string
            "123table",  # Starts with number
            "table name",  # Contains space
            "table;drop",  # Contains semicolon
            "table--comment",  # Contains SQL comment
            "table'quote",  # Contains quote
            'table"double',  # Contains double quote
            "table`backtick",  # Contains backtick
        ]

        for invalid_id in invalid_identifiers:
            with pytest.raises(ValueError, match="Invalid SQL identifier"):
                _validate_identifier(invalid_id)

    def test_identifier_validation_accepts_valid_patterns(self):
        """Test that identifier validation accepts valid SQL identifiers."""
        valid_identifiers = [
            "metrics",
            "auth_metrics",
            "table_123",
            "_private_table",
            "CamelCaseTable",
            "table_with_many_underscores_123",
        ]

        for valid_id in valid_identifiers:
            result = _validate_identifier(valid_id)
            assert result == valid_id

    def test_valid_table_names_accepted(self):
        """Test that valid table names from allowlist are accepted."""
        for table_name in ALLOWED_TABLE_NAMES:
            result = _validate_table_name(table_name)
            assert result == table_name

    def test_valid_timestamp_columns_accepted(self):
        """Test that valid timestamp columns from allowlist are accepted."""
        for column_name in ALLOWED_TIMESTAMP_COLUMNS:
            result = _validate_timestamp_column(column_name)
            assert result == column_name

    def test_retention_policy_rejects_invalid_table(self):
        """Test that RetentionPolicy constructor rejects invalid table names."""
        # Temporarily disable test tables to test production behavior
        _disable_test_tables()
        try:
            with pytest.raises(ValueError, match="Invalid table name"):
                RetentionPolicy(table_name="malicious; DROP TABLE--", retention_days=30)
        finally:
            _enable_test_tables()

    def test_retention_policy_rejects_invalid_timestamp_column(self):
        """Test that RetentionPolicy rejects invalid timestamp columns."""
        with pytest.raises(ValueError, match="Invalid timestamp column"):
            RetentionPolicy(
                table_name="metrics", retention_days=30, timestamp_column="invalid_column; DROP--"
            )

    @pytest.mark.asyncio
    async def test_update_policy_rejects_invalid_table(self, temp_db):
        """Test that update_policy rejects invalid table names."""
        manager = RetentionManager()
        manager.storage.db_path = temp_db

        # Temporarily disable test tables
        _disable_test_tables()
        try:
            with pytest.raises(ValueError, match="Invalid table name"):
                await manager.update_policy("metrics WHERE 1=1--", retention_days=30)
        finally:
            _enable_test_tables()

    @pytest.mark.asyncio
    async def test_get_cleanup_preview_rejects_invalid_table(self, temp_db):
        """Test that get_cleanup_preview rejects invalid table names."""
        manager = RetentionManager()
        manager.storage.db_path = temp_db

        # Temporarily disable test tables
        _disable_test_tables()
        try:
            with pytest.raises(ValueError, match="Invalid table name"):
                await manager.get_cleanup_preview("metrics; DROP TABLE--")
        finally:
            _enable_test_tables()

    @pytest.mark.asyncio
    async def test_cleanup_table_rejects_invalid_table(self, temp_db):
        """Test that cleanup_table rejects invalid table names via policy check."""
        manager = RetentionManager()
        manager.storage.db_path = temp_db

        # cleanup_table checks if table is in policies first
        # For a table not in policies, it raises ValueError
        with pytest.raises(ValueError, match="No retention policy found"):
            await manager.cleanup_table("nonexistent_malicious_table")


class TestAllowlistConfiguration:
    """Tests for allowlist configuration."""

    def test_allowlist_contains_expected_tables(self):
        """Test that the allowlist contains all expected production tables."""
        expected_tables = {
            "metrics",
            "auth_metrics",
            "discovery_metrics",
            "tool_metrics",
            "metrics_hourly",
            "metrics_daily",
            "api_key_usage_log",
        }

        assert ALLOWED_TABLE_NAMES == expected_tables

    def test_allowlist_contains_expected_timestamp_columns(self):
        """Test that the timestamp column allowlist contains expected values."""
        expected_columns = {
            "created_at",
            "timestamp",
            "updated_at",
        }

        assert ALLOWED_TIMESTAMP_COLUMNS == expected_columns

    def test_allowlist_is_immutable_set(self):
        """Test that the allowlist is a set (for O(1) lookups)."""
        assert isinstance(ALLOWED_TABLE_NAMES, set)
        assert isinstance(ALLOWED_TIMESTAMP_COLUMNS, set)


class TestRetentionDaysBounds:
    """Invariant: a retention window can never produce a future/all-rows cutoff.

    A negative retention_days would set the cleanup cutoff in the future so that
    ``DELETE ... WHERE created_at < <cutoff>`` matches every row and destroys the
    table. Zero is equally destructive. These tests assert the guard holds at the
    domain layer (the real fix) and that legitimate values still work.
    """

    def test_negative_retention_days_rejected_by_validator(self):
        """A negative day count must be rejected outright."""
        with pytest.raises(ValueError, match="at least"):
            _validate_retention_days(-1)

    def test_zero_retention_days_rejected_by_validator(self):
        """Zero days deletes everything older than 'now' and is not a valid policy."""
        with pytest.raises(ValueError, match="at least"):
            _validate_retention_days(0)

    def test_min_retention_days_accepted(self):
        """The floor value (1 day) is a valid, accepted policy."""
        assert _validate_retention_days(MIN_RETENTION_DAYS) == MIN_RETENTION_DAYS

    def test_absurdly_large_retention_days_rejected(self):
        """A value above the sanity ceiling is rejected as a likely mistake."""
        with pytest.raises(ValueError, match="not exceed"):
            _validate_retention_days(MAX_RETENTION_DAYS + 1)

    def test_boolean_retention_days_rejected(self):
        """A bool must not slip through as 1/0 (bool is an int subclass)."""
        with pytest.raises(ValueError, match="must be an integer"):
            _validate_retention_days(True)

    def test_policy_constructor_rejects_negative_days(self):
        """RetentionPolicy refuses to hold an unsafe negative window."""
        with pytest.raises(ValueError, match="at least"):
            RetentionPolicy(table_name="metrics", retention_days=-30)

    def test_policy_constructor_rejects_zero_days(self):
        """RetentionPolicy refuses to hold a zero-day (delete-all) window."""
        with pytest.raises(ValueError, match="at least"):
            RetentionPolicy(table_name="metrics", retention_days=0)

    def test_legitimate_positive_days_produces_past_cutoff(self):
        """A sensible window (90 days) yields a cutoff safely in the past."""
        policy = RetentionPolicy(table_name="metrics", retention_days=90)
        cutoff = policy._cutoff()
        now = datetime.now()

        assert cutoff < now
        # Roughly 90 days ago (allow a wide slack for test runtime).
        expected = now - timedelta(days=90)
        assert abs((cutoff - expected).total_seconds()) < 60

    def test_cutoff_clamped_to_past_floor_even_if_days_mutated(self):
        """Belt-and-suspenders: a corrupted window clamps to a PAST floor.

        Clamping the cutoff to ``now`` would be useless -- ``created_at < now``
        still matches every already-arrived (real) row. The clamp must instead
        pin the cutoff to ``now - MIN_RETENTION_DAYS`` so the predicate can only
        ever reach rows older than the minimum retention window, even if
        ``retention_days`` is mutated in place past validation.
        """
        policy = RetentionPolicy(table_name="metrics", retention_days=1)
        # Bypass validation to simulate a corrupted in-memory value.
        policy.retention_days = -365

        before = datetime.now()
        cutoff = policy._cutoff()

        expected_floor = before - timedelta(days=MIN_RETENTION_DAYS)
        # The cutoff sits at the floor (within a small slack for test runtime),
        # NOT at now.
        assert abs((cutoff - expected_floor).total_seconds()) < 60
        assert cutoff < before - timedelta(days=MIN_RETENTION_DAYS) + timedelta(seconds=60)

    def test_cleanup_query_cutoff_clamped_to_floor_for_bad_value(self):
        """The DELETE query's cutoff param is clamped to the past floor.

        A corrupted negative window must not yield a cutoff at/after ``now``
        (which would match every row); it must be pinned to
        ``now - MIN_RETENTION_DAYS``.
        """
        policy = RetentionPolicy(table_name="metrics", retention_days=1)
        policy.retention_days = -100  # corrupt past validation

        _query, params = policy.get_cleanup_query()
        cutoff_str = params[0]
        cutoff = datetime.fromisoformat(cutoff_str)

        expected_floor = datetime.now() - timedelta(days=MIN_RETENTION_DAYS)
        assert abs((cutoff - expected_floor).total_seconds()) < 60

    @pytest.mark.asyncio
    async def test_update_policy_rejects_negative_days(self, temp_db):
        """update_policy fails closed on a negative window (no persistence)."""
        manager = RetentionManager()
        manager.storage.db_path = temp_db

        original = manager.policies["metrics"].retention_days
        with pytest.raises(ValueError, match="at least"):
            await manager.update_policy("metrics", retention_days=-5)

        # The existing safe policy must be untouched.
        assert manager.policies["metrics"].retention_days == original

    @pytest.mark.asyncio
    async def test_update_policy_rejects_zero_days(self, temp_db):
        """update_policy fails closed on a zero-day window."""
        manager = RetentionManager()
        manager.storage.db_path = temp_db

        with pytest.raises(ValueError, match="at least"):
            await manager.update_policy("metrics", retention_days=0)

    @pytest.mark.asyncio
    async def test_update_policy_accepts_positive_days(self, temp_db):
        """The legitimate path: a positive window is accepted and stored."""
        manager = RetentionManager()
        manager.storage.db_path = temp_db

        await manager.update_policy("metrics", retention_days=45)
        assert manager.policies["metrics"].retention_days == 45

    @pytest.mark.asyncio
    async def test_corrupted_retention_days_cannot_delete_within_retention_window(self, temp_db):
        """End-to-end: a corrupted negative policy cannot wipe retained data.

        The real invariant: an in-memory-corrupted ``retention_days`` must not
        be able to delete real, recently-written rows. The original bug pushed
        the cutoff into the FUTURE so ``created_at < <future>`` matched EVERY
        row. Merely clamping the cutoff to ``now`` would NOT fix that -- every
        already-arrived past row still satisfies ``created_at < now``. Clamping
        the effective window to ``MIN_RETENTION_DAYS`` is what protects the data:
        rows inside the retention window (recent/now-dated, and rows only a few
        days old) must survive; only rows older than the floor may be reaped.
        """
        manager = RetentionManager()
        manager.storage.db_path = temp_db

        now = datetime.now()
        # Legitimate retained rows spanning the retention window. With a bug that
        # clamps the cutoff to ``now`` (or leaves it in the future), all of these
        # get deleted. With the correct past-floor clamp they must all survive
        # because they are newer than ``now - MIN_RETENTION_DAYS``.
        recent_values = [10.0, 11.0, 12.0, 13.0]
        recent_timestamps = [
            now.isoformat(),  # exactly now
            (now - timedelta(hours=1)).isoformat(),
            (now - timedelta(minutes=5)).isoformat(),
            now.isoformat(),
        ]
        # One genuinely old row (well past the floor) that SHOULD be reaped.
        old_ts = (now - timedelta(days=500)).isoformat()

        async with aiosqlite.connect(temp_db) as db:
            for ts, val in zip(recent_timestamps, recent_values, strict=False):
                await db.execute(
                    "INSERT INTO test_metrics (created_at, value) VALUES (?, ?)",
                    (ts, val),
                )
            await db.execute(
                "INSERT INTO test_metrics (created_at, value) VALUES (?, ?)",
                (old_ts, 99.0),
            )
            await db.commit()

        # Point a policy at test_metrics and corrupt its window past validation.
        manager.policies["test_metrics"] = RetentionPolicy(
            table_name="test_metrics", retention_days=1
        )
        manager.policies["test_metrics"].retention_days = -365

        await manager.cleanup_table("test_metrics", dry_run=False)

        async with aiosqlite.connect(temp_db) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM test_metrics WHERE value IN (10.0, 11.0, 12.0, 13.0)"
            )
            surviving_recent = (await cursor.fetchone())[0]
            cursor = await db.execute("SELECT COUNT(*) FROM test_metrics WHERE value = 99.0")
            surviving_old = (await cursor.fetchone())[0]

        assert surviving_recent == len(recent_values), (
            "a corrupted retention window must NOT delete rows inside the retention window"
        )
        # Sanity: the clamp still permits deleting rows older than the floor, so
        # the test proves survival of retained data rather than a no-op cleanup.
        assert surviving_old == 0, "rows older than the safe floor should still be reaped"

    @pytest.mark.asyncio
    async def test_load_policies_skips_unsafe_db_value(self, temp_db):
        """A persisted unsafe retention_days is ignored, keeping the safe default.

        Simulates an older vulnerable build having written a negative window to
        the retention_policies table. The loader must fail closed and not adopt it.
        """
        manager = RetentionManager()
        manager.storage.db_path = temp_db
        default_metrics_days = manager.policies["metrics"].retention_days

        # Persist a poisoned row directly.
        async with aiosqlite.connect(temp_db) as db:
            await db.execute(
                "INSERT OR REPLACE INTO retention_policies "
                "(table_name, retention_days, is_active, updated_at) "
                "VALUES (?, ?, 1, datetime('now'))",
                ("metrics", -999),
            )
            await db.commit()

        await manager.load_policies_from_database()

        # The poisoned value must NOT have been adopted.
        assert manager.policies["metrics"].retention_days == default_metrics_days
        assert manager.policies["metrics"].retention_days >= MIN_RETENTION_DAYS


class TestRetentionRouteValidation:
    """Invariant: the PUT retention endpoint rejects an unsafe day count (4xx)."""

    @pytest.fixture
    def client(self):
        """Test client with admin auth stubbed.

        These tests target the retention_days RANGE validation, not
        authorization, so the admin dependency is overridden to a static
        principal. The privilege-separation behavior of that dependency is
        covered in test_admin_authorization.py.
        """
        from fastapi.testclient import TestClient
        from app.main import app
        from app.api.auth import verify_admin_api_key

        app.dependency_overrides[verify_admin_api_key] = lambda: "admin"
        try:
            yield TestClient(app)
        finally:
            app.dependency_overrides.pop(verify_admin_api_key, None)

    def test_route_rejects_negative_retention_days(self, client):
        """PUT with negative retention_days returns a 4xx, never mutating state."""
        response = client.put(
            "/admin/retention/policies/metrics",
            params={"retention_days": -1},
        )
        assert response.status_code == 422

    def test_route_rejects_zero_retention_days(self, client):
        """PUT with zero retention_days returns a 4xx."""
        response = client.put(
            "/admin/retention/policies/metrics",
            params={"retention_days": 0},
        )
        assert response.status_code == 422

    def test_route_rejects_absurdly_large_retention_days(self, client):
        """PUT with a window above the sanity ceiling returns a 4xx."""
        response = client.put(
            "/admin/retention/policies/metrics",
            params={"retention_days": MAX_RETENTION_DAYS + 1},
        )
        assert response.status_code == 422
