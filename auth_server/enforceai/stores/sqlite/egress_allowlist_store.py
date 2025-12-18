from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Optional

from ...db.connection import (
    sqlite_connection,
)
from ...models.egress_allowlist import (
    EgressAllowlistEntryRecord,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _datetime_to_iso(
    value: Optional[datetime],
) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _datetime_from_iso(
    value: Optional[str],
) -> Optional[datetime]:
    if value is None:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


class SqliteEgressAllowlistStore:
    def __init__(
        self,
        *,
        db_path: Path,
    ) -> None:
        self._db_path = db_path

    def create_entry(
        self,
        *,
        kind: str,
        value: str,
        comment: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> EgressAllowlistEntryRecord:
        now = _utc_now()
        validated = EgressAllowlistEntryRecord(
            entry_id=1,
            kind=kind,
            value=value,
            comment=comment,
            expires_at=expires_at,
            created_at=now,
            updated_at=now,
        )

        with sqlite_connection(self._db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO egress_allowlist_entries(
                    kind,
                    value,
                    comment,
                    expires_at,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """.strip(),
                (
                    validated.kind,
                    validated.value,
                    validated.comment,
                    _datetime_to_iso(validated.expires_at),
                    _datetime_to_iso(validated.created_at),
                    _datetime_to_iso(validated.updated_at),
                ),
            )
            entry_id = int(cursor.lastrowid)

        record = self.get_entry_by_id(entry_id=entry_id)
        if record is None:
            raise RuntimeError(
                "Egress allowlist insert succeeded but record could not be read back"
            )
        return record

    def get_entry_by_id(
        self,
        *,
        entry_id: int,
    ) -> Optional[EgressAllowlistEntryRecord]:
        with sqlite_connection(self._db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    entry_id,
                    kind,
                    value,
                    comment,
                    expires_at,
                    created_at,
                    updated_at
                FROM egress_allowlist_entries
                WHERE entry_id = ?
                """.strip(),
                (entry_id,),
            ).fetchone()

        if row is None:
            return None

        return EgressAllowlistEntryRecord(
            entry_id=int(row[0]),
            kind=row[1],
            value=row[2],
            comment=row[3],
            expires_at=_datetime_from_iso(row[4]),
            created_at=_datetime_from_iso(row[5]),
            updated_at=_datetime_from_iso(row[6]),
        )

    def list_entries(
        self,
        *,
        include_expired: bool = False,
        now: Optional[datetime] = None,
    ) -> list[EgressAllowlistEntryRecord]:
        rows_query = """
            SELECT
                entry_id,
                kind,
                value,
                comment,
                expires_at,
                created_at,
                updated_at
            FROM egress_allowlist_entries
            ORDER BY entry_id ASC
        """.strip()

        with sqlite_connection(self._db_path) as connection:
            rows = connection.execute(rows_query).fetchall()

        records: list[EgressAllowlistEntryRecord] = []
        for row in rows:
            record = EgressAllowlistEntryRecord(
                entry_id=int(row[0]),
                kind=row[1],
                value=row[2],
                comment=row[3],
                expires_at=_datetime_from_iso(row[4]),
                created_at=_datetime_from_iso(row[5]),
                updated_at=_datetime_from_iso(row[6]),
            )
            records.append(record)

        if include_expired:
            return records

        ts = now or _utc_now()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return [
            record
            for record in records
            if record.expires_at is None or record.expires_at > ts
        ]

    def update_entry(
        self,
        *,
        entry_id: int,
        kind: Optional[str] = None,
        value: Optional[str] = None,
        comment: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[EgressAllowlistEntryRecord]:
        existing = self.get_entry_by_id(entry_id=entry_id)
        if existing is None:
            return None

        merged = EgressAllowlistEntryRecord(
            entry_id=existing.entry_id,
            kind=kind if kind is not None else existing.kind,
            value=value if value is not None else existing.value,
            comment=comment if comment is not None else existing.comment,
            expires_at=expires_at if expires_at is not None else existing.expires_at,
            created_at=existing.created_at,
            updated_at=_utc_now(),
        )

        with sqlite_connection(self._db_path) as connection:
            connection.execute(
                """
                UPDATE egress_allowlist_entries
                SET kind = ?, value = ?, comment = ?, expires_at = ?, updated_at = ?
                WHERE entry_id = ?
                """.strip(),
                (
                    merged.kind,
                    merged.value,
                    merged.comment,
                    _datetime_to_iso(merged.expires_at),
                    _datetime_to_iso(merged.updated_at),
                    entry_id,
                ),
            )

        return self.get_entry_by_id(entry_id=entry_id)

    def delete_entry(
        self,
        *,
        entry_id: int,
    ) -> bool:
        with sqlite_connection(self._db_path) as connection:
            cursor = connection.execute(
                "DELETE FROM egress_allowlist_entries WHERE entry_id = ?",
                (entry_id,),
            )
            return int(cursor.rowcount) > 0

