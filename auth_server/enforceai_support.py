from __future__ import annotations

import json
import logging
import os
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from .enforceai_runtime import (
        get_enforceai_stores,
    )
except ImportError:  # pragma: no cover
    from enforceai_runtime import (  # type: ignore[no-redef]
        get_enforceai_stores,
    )

logger = logging.getLogger(__name__)


def resolve_enforceai_scopes_catalog_path() -> Optional[Path]:
    raw = os.environ.get("ENFORCEAI_SCOPES_CATALOG_PATH") or os.environ.get("SCOPES_CATALOG_PATH")
    if raw is None:
        return None

    stripped = raw.strip()
    if not stripped:
        return None

    return Path(stripped)


def emit_enforceai_audit_event(
    *,
    action: str,
    outcome: str,
    user_id: str,
    agent_id: str,
    request_id: Optional[str],
    details: dict[str, Any],
    get_stores: Optional[Callable[[], Any]] = None,
) -> None:
    event = {
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
                event,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
    except Exception:  # noqa: BLE001 - best-effort
        logger.exception("Failed to emit EnforceAI audit event to stdout")

    try:
        stores_getter = get_stores or get_enforceai_stores
        stores = stores_getter()
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc).replace(microsecond=0),
            user_id=user_id,
            agent_id=agent_id,
            action=action,
            outcome=outcome,
            request_id=request_id,
            details=details,
        )
    except Exception:  # noqa: BLE001 - best-effort
        logger.exception("Failed to persist EnforceAI audit event")
