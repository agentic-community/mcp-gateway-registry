from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _apply_sqlite_pragmas(
    connection: sqlite3.Connection,
) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA busy_timeout = 5000")


@contextmanager
def sqlite_connection(
    db_path: Path,
) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(db_path)
    try:
        _apply_sqlite_pragmas(connection)
        with connection:
            yield connection
    finally:
        connection.close()

