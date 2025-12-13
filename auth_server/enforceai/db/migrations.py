from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

DEFAULT_MIGRATIONS_SQL_DIR: Path = (
    Path(__file__).resolve().parent / "migrations" / "sql"
)

_MIGRATION_FILENAME_RE = re.compile(
    r"^(?P<base>(?P<num>\d{4})_[a-zA-Z0-9][a-zA-Z0-9_-]*)\.(?P<direction>up|down)\.sql$"
)


@dataclass(frozen=True)
class Migration:
    version: str
    version_number: int
    up_path: Path
    down_path: Path


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _ensure_schema_migrations_table(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """.strip()
    )


def _discover_migrations(
    *,
    migrations_sql_dir: Path,
) -> list[Migration]:
    if not migrations_sql_dir.exists():
        raise FileNotFoundError(
            f"Migrations directory does not exist: {migrations_sql_dir}"
        )

    up_files: list[Path] = []
    for path in migrations_sql_dir.iterdir():
        if not path.is_file():
            continue
        match = _MIGRATION_FILENAME_RE.match(path.name)
        if match and match.group("direction") == "up":
            up_files.append(path)

    migrations: list[Migration] = []
    seen_versions: set[str] = set()

    for up_path in sorted(up_files, key=lambda p: p.name):
        match = _MIGRATION_FILENAME_RE.match(up_path.name)
        if match is None:
            continue

        base = match.group("base")
        number = int(match.group("num"))
        version = base
        if version in seen_versions:
            raise ValueError(f"Duplicate migration version: {version}")
        seen_versions.add(version)

        down_path = migrations_sql_dir / f"{base}.down.sql"
        if not down_path.exists():
            raise FileNotFoundError(
                f"Missing down migration for {version}: {down_path}"
            )

        migrations.append(
            Migration(
                version=version,
                version_number=number,
                up_path=up_path,
                down_path=down_path,
            )
        )

    migrations.sort(key=lambda m: (m.version_number, m.version))
    return migrations


def _read_sql(
    path: Path,
) -> str:
    return path.read_text(encoding="utf-8")


def _get_applied_versions(
    connection: sqlite3.Connection,
) -> list[str]:
    _ensure_schema_migrations_table(connection)
    rows = connection.execute(
        "SELECT version FROM schema_migrations ORDER BY version ASC"
    ).fetchall()
    return [row[0] for row in rows]


def upgrade_to_latest(
    connection: sqlite3.Connection,
    *,
    migrations_sql_dir: Path = DEFAULT_MIGRATIONS_SQL_DIR,
) -> None:
    """Apply all unapplied migrations in order."""
    _ensure_schema_migrations_table(connection)
    migrations = _discover_migrations(
        migrations_sql_dir=migrations_sql_dir,
    )
    applied = set(_get_applied_versions(connection))

    for migration in migrations:
        if migration.version in applied:
            continue

        sql = _read_sql(migration.up_path)
        logger.info(f"Applying migration {migration.version}")
        connection.executescript(sql)
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (migration.version, _utc_now_iso()),
        )
        connection.commit()
        applied.add(migration.version)


def downgrade_one(
    connection: sqlite3.Connection,
    *,
    migrations_sql_dir: Path = DEFAULT_MIGRATIONS_SQL_DIR,
) -> str | None:
    """Rollback the most recently applied migration.

    Returns:
        The version rolled back, or None if there are no applied migrations.
    """
    _ensure_schema_migrations_table(connection)
    applied_versions = _get_applied_versions(connection)
    if not applied_versions:
        return None

    migrations_by_version = {
        migration.version: migration
        for migration in _discover_migrations(migrations_sql_dir=migrations_sql_dir)
    }

    target_version = max(applied_versions)
    migration = migrations_by_version.get(target_version)
    if migration is None:
        raise ValueError(
            f"Applied migration not found on disk: {target_version}"
        )

    sql = _read_sql(migration.down_path)
    logger.info(f"Rolling back migration {migration.version}")
    connection.executescript(sql)
    connection.execute(
        "DELETE FROM schema_migrations WHERE version = ?",
        (migration.version,),
    )
    connection.commit()
    return migration.version


def list_migrations(
    *,
    migrations_sql_dir: Path = DEFAULT_MIGRATIONS_SQL_DIR,
) -> Iterable[Migration]:
    """List migrations available on disk, ordered by version."""
    return _discover_migrations(migrations_sql_dir=migrations_sql_dir)

