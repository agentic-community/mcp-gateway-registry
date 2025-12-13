from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import sqlite3

from auth_server.enforceai.db.connection import (
    sqlite_connection,
)
from auth_server.enforceai.models.agent import (
    AgentRecord,
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


def _json_dumps(
    value: Any,
) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _json_loads_optional(
    raw: Optional[str],
) -> Any:
    if raw is None:
        return None
    return json.loads(raw)


class SqliteAgentStore:
    def __init__(
        self,
        *,
        db_path: Path,
    ) -> None:
        self._db_path = db_path

    def create_agent(
        self,
        *,
        user_id: str,
        agent_id: str,
        scopes: list[str],
        allowed_tools: Optional[list[str]] = None,
        alias: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> AgentRecord:
        now = _utc_now()
        validated = AgentRecord(
            user_id=user_id,
            agent_id=agent_id,
            scopes=scopes,
            allowed_tools=allowed_tools,
            alias=alias,
            metadata=metadata,
            revoked_at=None,
            tokens_valid_after=None,
            created_at=now,
            updated_at=now,
        )

        with sqlite_connection(self._db_path) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO agents(
                        agent_id,
                        user_id,
                        scopes_json,
                        allowed_tools_json,
                        alias,
                        metadata_json,
                        revoked_at,
                        tokens_valid_after,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """.strip(),
                    (
                        validated.agent_id,
                        validated.user_id,
                        _json_dumps(validated.scopes),
                        _json_dumps(validated.allowed_tools)
                        if validated.allowed_tools is not None
                        else None,
                        validated.alias,
                        _json_dumps(validated.metadata)
                        if validated.metadata is not None
                        else None,
                        None,
                        None,
                        _datetime_to_iso(validated.created_at),
                        _datetime_to_iso(validated.updated_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Agent already exists: {validated.agent_id}") from exc
        record = self.get_agent_by_id(agent_id=agent_id)
        if record is None:
            raise RuntimeError("Agent insert succeeded but record could not be read back")
        return record

    def get_agent_by_id(
        self,
        *,
        agent_id: str,
    ) -> Optional[AgentRecord]:
        with sqlite_connection(self._db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    agent_id,
                    user_id,
                    scopes_json,
                    allowed_tools_json,
                    alias,
                    metadata_json,
                    revoked_at,
                    tokens_valid_after,
                    created_at,
                    updated_at
                FROM agents
                WHERE agent_id = ?
                """.strip(),
                (agent_id,),
            ).fetchone()

        if row is None:
            return None

        return AgentRecord(
            agent_id=row[0],
            user_id=row[1],
            scopes=json.loads(row[2]),
            allowed_tools=_json_loads_optional(row[3]),
            alias=row[4],
            metadata=_json_loads_optional(row[5]),
            revoked_at=_datetime_from_iso(row[6]),
            tokens_valid_after=_datetime_from_iso(row[7]),
            created_at=_datetime_from_iso(row[8]),
            updated_at=_datetime_from_iso(row[9]),
        )

    def list_agents_by_user_id(
        self,
        *,
        user_id: str,
    ) -> list[AgentRecord]:
        with sqlite_connection(self._db_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    agent_id,
                    user_id,
                    scopes_json,
                    allowed_tools_json,
                    alias,
                    metadata_json,
                    revoked_at,
                    tokens_valid_after,
                    created_at,
                    updated_at
                FROM agents
                WHERE user_id = ?
                ORDER BY created_at ASC
                """.strip(),
                (user_id,),
            ).fetchall()

        return [
            AgentRecord(
                agent_id=row[0],
                user_id=row[1],
                scopes=json.loads(row[2]),
                allowed_tools=_json_loads_optional(row[3]),
                alias=row[4],
                metadata=_json_loads_optional(row[5]),
                revoked_at=_datetime_from_iso(row[6]),
                tokens_valid_after=_datetime_from_iso(row[7]),
                created_at=_datetime_from_iso(row[8]),
                updated_at=_datetime_from_iso(row[9]),
            )
            for row in rows
        ]

    def update_agent(
        self,
        *,
        agent_id: str,
        scopes: Optional[list[str]] = None,
        allowed_tools: Optional[list[str]] = None,
        alias: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> Optional[AgentRecord]:
        existing = self.get_agent_by_id(agent_id=agent_id)
        if existing is None:
            return None

        now = _utc_now()
        updated_scopes = scopes if scopes is not None else existing.scopes
        updated_allowed_tools = (
            allowed_tools if allowed_tools is not None else existing.allowed_tools
        )
        updated_alias = alias if alias is not None else existing.alias
        updated_metadata = metadata if metadata is not None else existing.metadata

        AgentRecord(
            user_id=existing.user_id,
            agent_id=existing.agent_id,
            scopes=updated_scopes,
            allowed_tools=updated_allowed_tools,
            alias=updated_alias,
            metadata=updated_metadata,
            revoked_at=existing.revoked_at,
            tokens_valid_after=existing.tokens_valid_after,
            created_at=existing.created_at,
            updated_at=now,
        )

        with sqlite_connection(self._db_path) as connection:
            connection.execute(
                """
                UPDATE agents
                SET
                    scopes_json = ?,
                    allowed_tools_json = ?,
                    alias = ?,
                    metadata_json = ?,
                    updated_at = ?
                WHERE agent_id = ?
                """.strip(),
                (
                    _json_dumps(updated_scopes),
                    _json_dumps(updated_allowed_tools)
                    if updated_allowed_tools is not None
                    else None,
                    updated_alias,
                    _json_dumps(updated_metadata) if updated_metadata is not None else None,
                    _datetime_to_iso(now),
                    agent_id,
                ),
            )

        return self.get_agent_by_id(agent_id=agent_id)

    def revoke_agent(
        self,
        *,
        agent_id: str,
        revoked_at: Optional[datetime] = None,
    ) -> Optional[AgentRecord]:
        existing = self.get_agent_by_id(agent_id=agent_id)
        if existing is None:
            return None
        if existing.revoked_at is not None:
            return existing

        now = revoked_at or _utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        with sqlite_connection(self._db_path) as connection:
            connection.execute(
                """
                UPDATE agents
                SET revoked_at = ?, updated_at = ?
                WHERE agent_id = ?
                """.strip(),
                (
                    _datetime_to_iso(now),
                    _datetime_to_iso(_utc_now()),
                    agent_id,
                ),
            )

        return self.get_agent_by_id(agent_id=agent_id)

    def bump_tokens_valid_after(
        self,
        *,
        agent_id: str,
        tokens_valid_after: datetime,
    ) -> Optional[AgentRecord]:
        existing = self.get_agent_by_id(agent_id=agent_id)
        if existing is None:
            return None

        if tokens_valid_after.tzinfo is None:
            tokens_valid_after = tokens_valid_after.replace(tzinfo=timezone.utc)

        AgentRecord(
            user_id=existing.user_id,
            agent_id=existing.agent_id,
            scopes=existing.scopes,
            allowed_tools=existing.allowed_tools,
            alias=existing.alias,
            metadata=existing.metadata,
            revoked_at=existing.revoked_at,
            tokens_valid_after=tokens_valid_after,
            created_at=existing.created_at,
            updated_at=_utc_now(),
        )

        if (
            existing.tokens_valid_after is not None
            and existing.tokens_valid_after >= tokens_valid_after
        ):
            return existing

        with sqlite_connection(self._db_path) as connection:
            connection.execute(
                """
                UPDATE agents
                SET tokens_valid_after = ?, updated_at = ?
                WHERE agent_id = ?
                """.strip(),
                (
                    _datetime_to_iso(tokens_valid_after),
                    _datetime_to_iso(_utc_now()),
                    agent_id,
                ),
            )

        return self.get_agent_by_id(agent_id=agent_id)
