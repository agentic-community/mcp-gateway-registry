"""Audit event sink.

This module exposes a single public function, `emit_audit_event`, used by
callers (for example the tool filter in `registry.auth.tool_filter`) to
record structured audit events. The current implementation logs events as
JSON at INFO on the `registry.audit` logger. A DB-backed sink can replace
this implementation without changes at the call sites.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

_audit_logger = logging.getLogger("registry.audit")


def emit_audit_event(
    event: BaseModel,
) -> None:
    """Emit an audit event as a JSON log line.

    Best-effort: never raises. Callers that rely on this (notably the tool
    filter) wrap the call in their own try/except to avoid breaking the
    request path on any unexpected failure.
    """
    try:
        payload = event.model_dump_json()
        _audit_logger.info(payload)
    except Exception:
        _audit_logger.exception("emit_audit_event failed")
