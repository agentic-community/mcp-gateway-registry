"""Audit event sink.

This module exposes a single public function, `emit_audit_event`, used by
callers (for example the tool filter in `registry.auth.tool_filter`) to
record structured audit events. Events are logged as JSON at INFO on the
`registry.audit.records` logger, which is deliberately TERMINAL: it does not
propagate to the root logger, and it carries no identity-claim values. See
`_configure_record_logger` and `_operator_payload` for why.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from pydantic import BaseModel

from ..common.log_redaction import REDACTED
from .models import IDENTITY_CLAIM_FIELDS

# Module logger for this module's OWN failures. Propagates normally: an audit
# emission that breaks is an application fault an operator must see in the
# application log stream.
logger = logging.getLogger(__name__)

# The audit RECORD stream. A dedicated child logger rather than `registry.audit`
# itself, because `registry.audit` is the parent of the audit module loggers
# (`registry.audit.service` emits the CRITICAL "AUDIT RECORD DROPPED" line) and
# making that parent terminal would strand every one of them.
_record_logger = logging.getLogger("registry.audit.records")

# Membership set for the masker's hot path; the tuple is the canonical order.
_CLAIM_FIELDS: frozenset[str] = frozenset(IDENTITY_CLAIM_FIELDS)

# Name on the dedicated handler, so repeated imports / a re-run of
# `setup_logging` cannot install a second copy and double every audit line.
_RECORD_HANDLER_NAME = "registry-audit-records"

# Guard for pathological/cyclic payloads, mirroring `redact_mapping`'s bound.
_MAX_MASK_DEPTH = 10


def _configure_record_logger() -> None:
    """Make the audit record stream terminal, with its own handler.

    An audit record body must never enter the APPLICATION log stream. The root
    logger owns three handlers (`registry.utils.logging_setup.setup_logging`):
    stdout, a `RotatingFileHandler`, and optionally `MongoDBLogHandler`, which
    writes the ``application_logs`` collection. That collection has a DIFFERENT
    retention (``APP_LOG_MONGODB_RETENTION_DAYS``, 1 day by default) and a
    DIFFERENT access boundary (the admin log-viewer API, ``GET
    /api/admin/logs``) than the audit collection the compliance documentation
    reasons about. Propagating an audit record there copies ``principal_name``
    (a UPN/email), ``canonical_id``, ``subject``, ``object_id``, ``tenant_id``
    and ``app_id`` into all three sinks.

    So: ``propagate = False`` plus one dedicated handler. Adding the logger to
    ``APP_LOG_EXCLUDED_LOGGERS`` is NOT the fix -- that setting only filters the
    MongoDB handler, leaving stdout and the rotating file untouched, and an
    operator can unset it.

    The level is pinned on the logger itself so tuning ``APP_LOG_LEVEL`` to
    WARNING cannot silently switch the audit trail off.
    """
    _record_logger.propagate = False
    _record_logger.setLevel(logging.INFO)

    if any(handler.get_name() == _RECORD_HANDLER_NAME for handler in _record_logger.handlers):
        return

    # Operator-facing copy. The tool-filter audit event has no other sink, so
    # this handler must exist; it emits `_operator_payload`, which carries no
    # claim values, because a log agent scrapes stdout into the same store the
    # root handlers write to.
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_RECORD_HANDLER_NAME)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s,audit,%(message)s"))
    _record_logger.addHandler(handler)


def _mask_claim_values(data: Any, _depth: int = 0) -> Any:
    """Replace identity-claim VALUES with the redaction marker, at any depth.

    Presence is preserved (an absent claim stays ``null``) so an operator
    reading the line still knows the record carries a durable identity and can
    go query the audit collection for it -- without the value itself landing in
    a log sink with a shorter retention and a wider audience.

    Recursive because the claims are top level on the ``token_mint`` stream but
    nested under ``identity`` on every other one.
    """
    if _depth > _MAX_MASK_DEPTH:
        return REDACTED

    if isinstance(data, dict):
        return {
            key: (
                REDACTED
                if key in _CLAIM_FIELDS and value is not None
                else _mask_claim_values(value, _depth + 1)
            )
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [_mask_claim_values(item, _depth + 1) for item in data]

    return data


def _operator_payload(event: BaseModel) -> str:
    """Render the JSON body that is safe for the operator-facing audit stream.

    Only the durable IdP identifiers are dropped to a presence marker.
    Everything else (request id, correlation id, token kind and path, outcome,
    pruned tool names, the display ``username``) is emitted verbatim, so this
    stream stays exactly as usable as it already was.

    `registry.common.log_redaction.redact_mapping` is deliberately NOT applied
    on top: its substring key matching is tuned for request bodies and user
    contexts, and here it would redact ``token_kind`` and ``token_path`` -- the
    mint record's primary payload -- because both contain "token". Credential
    masking for these records lives in the models instead (`mask_credential` on
    ``credential_hint``, `SENSITIVE_QUERY_PARAMS` on ``Request.query_params``),
    so no raw credential can reach this function.

    Full-fidelity claim values live in the durable audit store only
    (``AuditLogger.log_event`` -> the audit collection). An operator who opted
    out of a durable sink accepted a trail without them.
    """
    return json.dumps(
        _mask_claim_values(event.model_dump(mode="json")),
        default=str,
        separators=(",", ":"),
    )


def emit_audit_event(
    event: BaseModel,
) -> None:
    """Emit an audit event as a JSON log line.

    Best-effort: never raises. Callers that rely on this (notably the tool
    filter) wrap the call in their own try/except to avoid breaking the
    request path on any unexpected failure.
    """
    try:
        _record_logger.info(_operator_payload(event))
    except Exception:
        logger.exception("emit_audit_event failed for %s", type(event).__name__)


_configure_record_logger()
