from __future__ import annotations

from fastapi import (
    HTTPException,
)

from ..auth.dependency import (
    EnforceAIManagementContext,
)
from ..db.data_layer import (
    EnforceAIStores,
)
from ..models.upstream_credentials import (
    UpstreamCredentialRecord,
)


def _normalize_server_path(
    raw: str,
) -> str:
    stripped = raw.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="server_path is required")
    if not stripped.startswith("/"):
        stripped = "/" + stripped
    return stripped.rstrip("/") or "/"


def _owned_agent_ids_for_user(
    *,
    stores: EnforceAIStores,
    user_id: str,
) -> set[str]:
    return {
        record.agent_id for record in stores.agent_store.list_agents_by_user_id(user_id=user_id)
    }


def _is_upstream_credential_visible(
    *,
    record: UpstreamCredentialRecord,
    context: EnforceAIManagementContext,
    owned_agent_ids: set[str],
    include_service: bool,
) -> bool:
    if record.credential_binding == "service":
        return include_service and context.is_admin
    if record.credential_binding == "user":
        return record.user_id == context.user_id
    if record.credential_binding == "agent":
        return record.agent_id in owned_agent_ids
    if record.credential_binding == "user+agent":
        return record.user_id == context.user_id and record.agent_id in owned_agent_ids
    return False


def _credential_key(
    record: UpstreamCredentialRecord,
) -> tuple[object, ...]:
    return (
        record.server_path,
        record.credential_type,
        record.credential_binding,
        record.user_id,
        record.agent_id,
        record.provider,
    )
