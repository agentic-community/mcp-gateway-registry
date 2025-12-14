#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Mapping, Optional

import sqlite3

from auth_server.enforceai.audit.retention import (
    compute_cutoff,
    enforce_size_retention,
    enforce_time_retention,
)
from auth_server.enforceai.errors import (
    DependencyUnavailableError,
)
from auth_server.enforceai.stores.sqlite.audit_store import (
    SqliteAuditStore,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

ENV_DB_PATH: str = "ENFORCEAI_DB_PATH"
ENV_RETENTION_DAYS: str = "ENFORCEAI_AUDIT_RETENTION_DAYS"
ENV_MAX_DB_BYTES: str = "ENFORCEAI_AUDIT_MAX_DB_BYTES"

DEFAULT_RETENTION_DAYS: int = 30
DEFAULT_MAX_DB_BYTES: int = 500_000_000
DEFAULT_BATCH_SIZE: int = 500


class CLIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        exit_code: int = 2,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _datetime_to_iso_z(
    value: datetime,
) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc).replace(microsecond=0)
    return value.isoformat().replace("+00:00", "Z")


def _resolve_config_value(
    *,
    cli_value: Optional[str],
    env: Mapping[str, str],
    env_key: str,
) -> Optional[str]:
    if cli_value is not None and cli_value.strip():
        return cli_value.strip()

    raw_env = env.get(env_key)
    if raw_env is None:
        return None

    stripped = raw_env.strip()
    return stripped or None


def _resolve_int(
    *,
    cli_value: Optional[int],
    env: Mapping[str, str],
    env_key: str,
    default: int,
) -> int:
    if cli_value is not None:
        return cli_value

    raw_env = env.get(env_key)
    if raw_env is None:
        return default

    stripped = raw_env.strip()
    if not stripped:
        return default

    try:
        return int(stripped)
    except ValueError as exc:
        raise CLIError(f"{env_key} must be an integer") from exc


def _ensure_audit_table_exists(
    *,
    db_path: Path,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'audit_events'
            """.strip()
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise CLIError(
            "Missing audit_events table in DB. "
            "Ensure EnforceAI DB migrations have been applied."
        )


def _count_events_older_than(
    *,
    db_path: Path,
    cutoff: datetime,
) -> int:
    cutoff_iso = _datetime_to_iso_z(cutoff)
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            """
            SELECT COUNT(*)
            FROM audit_events
            WHERE occurred_at < ?
            """.strip(),
            (cutoff_iso,),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        return 0
    return int(row[0])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EnforceAI audit retention cleanup (out-of-band operator command).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Cleanup using env vars
  export ENFORCEAI_DB_PATH=/path/to/enforceai.db
  export ENFORCEAI_AUDIT_RETENTION_DAYS=30
  export ENFORCEAI_AUDIT_MAX_DB_BYTES=500000000
  python -m cli.enforceai_audit_cleanup

  # Override retention settings
  python -m cli.enforceai_audit_cleanup --db-path /path/to/enforceai.db --retention-days 7 --max-db-bytes 200000000

  # Dry run (no DB writes)
  python -m cli.enforceai_audit_cleanup --db-path /path/to/enforceai.db --dry-run
""".strip(),
    )

    parser.add_argument(
        "--db-path",
        help=f"SQLite DB path (or {ENV_DB_PATH})",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help=f"Days to retain (default: {DEFAULT_RETENTION_DAYS}, or {ENV_RETENTION_DAYS}); use 0 to disable",
    )
    parser.add_argument(
        "--max-db-bytes",
        type=int,
        default=None,
        help=f"Max DB size cap (default: {DEFAULT_MAX_DB_BYTES}, or {ENV_MAX_DB_BYTES}); use 0 to disable",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Delete batch size for size-based retention (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be deleted; no DB writes",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser


def main(
    argv: Optional[list[str]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    args = _build_parser().parse_args(argv)
    env_mapping = dict(env) if env is not None else dict(os.environ)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    db_path_raw = _resolve_config_value(
        cli_value=args.db_path,
        env=env_mapping,
        env_key=ENV_DB_PATH,
    )
    if db_path_raw is None:
        raise CLIError(f"db path is required via --db-path or {ENV_DB_PATH}")

    db_path = Path(db_path_raw)
    if not db_path.exists():
        raise CLIError(f"db path does not exist: {db_path}")

    retention_days = _resolve_int(
        cli_value=args.retention_days,
        env=env_mapping,
        env_key=ENV_RETENTION_DAYS,
        default=DEFAULT_RETENTION_DAYS,
    )
    max_db_bytes = _resolve_int(
        cli_value=args.max_db_bytes,
        env=env_mapping,
        env_key=ENV_MAX_DB_BYTES,
        default=DEFAULT_MAX_DB_BYTES,
    )

    if retention_days < 0:
        raise CLIError("retention-days must be non-negative")
    if max_db_bytes < 0:
        raise CLIError("max-db-bytes must be non-negative")
    if args.batch_size <= 0:
        raise CLIError("batch-size must be positive")

    _ensure_audit_table_exists(db_path=db_path)

    started_at = datetime.now(tz=timezone.utc)
    started_monotonic = time.monotonic()

    logger.info(
        "Starting audit cleanup: db_path=%s retention_days=%s max_db_bytes=%s batch_size=%s dry_run=%s",
        db_path,
        retention_days,
        max_db_bytes,
        args.batch_size,
        args.dry_run,
    )

    cutoff = compute_cutoff(
        now=started_at,
        retention_days=retention_days,
    )

    store = SqliteAuditStore(db_path=db_path)

    if args.dry_run:
        deleted_by_time = 0
        if cutoff is not None:
            deleted_by_time = _count_events_older_than(
                db_path=db_path,
                cutoff=cutoff,
            )
            logger.info(
                "Dry run: would delete %s events older than %s",
                deleted_by_time,
                _datetime_to_iso_z(cutoff),
            )
        else:
            logger.info("Dry run: time-based retention disabled")

        deleted_by_size = 0
        if max_db_bytes != 0:
            logger.info(
                "Dry run: size-based retention not simulated (max_db_bytes=%s current_bytes=%s)",
                max_db_bytes,
                db_path.stat().st_size,
            )

        finished_at = datetime.now(tz=timezone.utc)
        elapsed_seconds = time.monotonic() - started_monotonic

        summary = {
            "deleted_by_time": deleted_by_time,
            "deleted_by_size": deleted_by_size,
            "final_db_bytes": db_path.stat().st_size,
            "started_at": _datetime_to_iso_z(started_at),
            "finished_at": _datetime_to_iso_z(finished_at),
            "elapsed_seconds": round(elapsed_seconds, 6),
        }
        sys.stdout.write(json.dumps(summary, separators=(",", ":"), sort_keys=True))
        sys.stdout.write("\n")
        return 0

    try:
        deleted_by_time = enforce_time_retention(
            audit_store=store,
            cutoff=cutoff,
        )
        deleted_by_size = enforce_size_retention(
            db_path=db_path,
            audit_store=store,
            max_db_bytes=max_db_bytes,
            batch_size=args.batch_size,
        )
    except DependencyUnavailableError:
        logger.exception("Audit cleanup failed due to dependency error")
        return 1
    except Exception:
        logger.exception("Audit cleanup failed unexpectedly")
        return 1

    finished_at = datetime.now(tz=timezone.utc)
    elapsed_seconds = time.monotonic() - started_monotonic

    summary = {
        "deleted_by_time": int(deleted_by_time),
        "deleted_by_size": int(deleted_by_size),
        "final_db_bytes": db_path.stat().st_size,
        "started_at": _datetime_to_iso_z(started_at),
        "finished_at": _datetime_to_iso_z(finished_at),
        "elapsed_seconds": round(elapsed_seconds, 6),
    }
    sys.stdout.write(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CLIError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(exc.exit_code)

