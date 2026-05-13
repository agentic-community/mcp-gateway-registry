"""Pydantic audit event models for tool-level access enforcement.

These events are emitted by the tool-filter helper (Issue #1026) whenever
a user-visible tool list is pruned by policy. The events are routed through
`registry.audit.sink.emit_audit_event`, which currently logs them as JSON
at INFO on the `registry.audit` logger. A DB-backed sink can be wired in
later without changing the call sites.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ToolFilterAuditEvent(BaseModel):
    """Emitted when a user-visible tool list is pruned by policy.

    Emitted only when `pruned_count > 0`. Wildcard and admin paths do not
    emit audit events because no pruning occurred.
    """

    username: str
    endpoint: Literal[
        "servers",
        "tools_service",
        "tools_all",
        "semantic_search",
        "mcp_tools_list",
    ]
    server_name: str
    pruned_count: int = Field(..., ge=0)
    kept_count: int = Field(..., ge=0)
    pruned_tool_names: list[str]
    user_scopes: list[str]
    ts: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
