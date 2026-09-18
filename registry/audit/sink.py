"""Audit event sink.

This module exposes a single public function, `emit_audit_event`, used by
callers (for example the tool filter in `registry.auth.tool_filter`) to
record structured audit events. Events are logged as JSON at INFO on the
`registry.audit.records` logger, which is deliberately TERMINAL: it does not
propagate to the root logger, it carries no identity-claim values, and an
address-shaped display identity is reduced to its first character and domain.
See `_configure_record_logger` and `_operator_payload` for why.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from pydantic import BaseModel

from ..common.log_redaction import REDACTED
from .events import ToolFilterAuditEvent
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

# The record's readable display identity. Named because it is handled differently
# from the durable claims: partially masked rather than redacted, since it is the
# only thing that lets an operator tell two callers apart on this stream.
_DISPLAY_IDENTITY_FIELD: str = "username"

# Event types this stream is the ONLY sink for. An audit record is persisted in
# full to the audit collection, so reducing its display identity here loses
# nothing recoverable; ToolFilterAuditEvent has no durable counterpart, so the
# same reduction would destroy the only copy of who was affected. Left intact.
_NO_DURABLE_SINK_TYPES: tuple[type, ...] = (ToolFilterAuditEvent,)

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


def _partial_identity(value: Any) -> Any:
    """Mask the local part of an address-shaped display identity.

    ``username`` is the record's readable identity and stays in this stream on
    purpose: an operator scanning it needs to tell one caller from another. But
    on the shape this whole feature exists for -- an Entra v1.0 access token --
    ``_audit_identity_display`` resolves to the ``upn`` and ``principal_name``
    resolves to the same ``upn``. Redacting one while emitting the other verbatim
    left the address in the stream unchanged, so the claim masking bought nothing
    for the readable identifier in the commonest case.

    ``alice@contoso.com`` becomes ``a***@contoso.com``: enough to distinguish
    callers and to recognise a domain, without the mailbox. The full value stays
    in the durable audit store, which is where an investigation reads it.

    A value with no ``@`` is left alone. Those are opaque subjects, service
    identities and the ``anonymous`` literal -- pseudonymous or not personal, and
    the sole handle an operator has on that line.
    """
    if not isinstance(value, str) or "@" not in value:
        return value
    local, _, domain = value.partition("@")
    if not local or not domain:
        return value
    return f"{local[0]}***@{domain}"


def _mask_claim_values(data: Any, mask_display: bool = True, _depth: int = 0) -> Any:
    """Reduce identity VALUES to non-identifying forms, at any depth.

    Durable claims become the redaction marker. Presence is preserved (an absent
    claim stays ``null``) so an operator reading the line still knows the record
    carries a durable identity and can go query the audit collection for it --
    without the value itself landing in a log sink with a shorter retention and a
    wider audience.

    The readable ``username`` is partially masked rather than dropped; see
    ``_partial_identity`` for why it is treated differently from the claims, and
    ``_NO_DURABLE_SINK_TYPES`` for when ``mask_display`` is False.

    Recursive because the claims are top level on the ``token_mint`` stream but
    nested under ``identity`` on every other one.
    """
    if _depth > _MAX_MASK_DEPTH:
        return REDACTED

    if isinstance(data, dict):
        masked: dict[str, Any] = {}
        for key, value in data.items():
            if key in _CLAIM_FIELDS and value is not None:
                masked[key] = REDACTED
            elif key == _DISPLAY_IDENTITY_FIELD and mask_display:
                masked[key] = _partial_identity(value)
            else:
                masked[key] = _mask_claim_values(value, mask_display, _depth + 1)
        return masked

    if isinstance(data, list):
        return [_mask_claim_values(item, mask_display, _depth + 1) for item in data]

    return data


def _operator_payload(event: BaseModel) -> str:
    """Render the JSON body that is safe for the operator-facing audit stream.

    The durable IdP identifiers are dropped to a presence marker, and on an event
    with a durable counterpart an address-shaped display ``username`` keeps only
    its first character and domain. Everything else (request id, correlation id,
    token kind and path, outcome, pruned tool names) is emitted verbatim, so the
    stream stays usable for correlating and for telling two callers apart.

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
        _mask_claim_values(
            event.model_dump(mode="json"),
            mask_display=not isinstance(event, _NO_DURABLE_SINK_TYPES),
        ),
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
