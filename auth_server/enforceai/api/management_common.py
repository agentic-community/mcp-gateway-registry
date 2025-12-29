from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import (
    HTTPException,
    Request,
)

from ..auth.dependency import (
    EnforceAIManagementContext,
)
from ..db.data_layer import (
    EnforceAIStores,
)
from ..errors import (
    EnforceAIError,
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _get_request_id(
    request: Request,
) -> Optional[str]:
    value = request.headers.get("X-Request-Id")
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _require_admin(
    context: EnforceAIManagementContext,
) -> None:
    if context.is_admin:
        return

    raise HTTPException(
        status_code=403,
        detail="Admin required",
    )


def _emit_management_audit_event(
    *,
    stores: EnforceAIStores,
    action: str,
    outcome: str,
    user_id: str,
    agent_id: str,
    request_id: Optional[str],
    details: dict[str, Any],
) -> None:
    payload = {
        "event_type": "enforceai_audit",
        "action": action,
        "outcome": outcome,
        "user_id": user_id,
        "agent_id": agent_id,
        "request_id": request_id,
        "details": details,
    }

    try:
        print(
            json.dumps(
                payload,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
    except Exception:  # noqa: BLE001 - best-effort
        logger.exception("Failed to emit EnforceAI audit event to stdout")

    try:
        stores.audit_store.append_event(
            occurred_at=_utc_now(),
            user_id=user_id,
            agent_id=agent_id,
            action=action,
            outcome=outcome,
            request_id=request_id,
            details=details,
        )
    except Exception:  # noqa: BLE001 - best-effort
        logger.exception("Failed to persist EnforceAI audit event")


def _map_management_error(
    exc: Exception,
) -> HTTPException:
    if isinstance(exc, EnforceAIError):
        return exc.as_http_exception()
    if isinstance(exc, ValueError):
        return HTTPException(
            status_code=400,
            detail=str(exc),
        )
    return HTTPException(
        status_code=503,
        detail="Enforcement dependency unavailable",
    )

