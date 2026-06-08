"""
Per-item processor for agent batch jobs (issue #956).

Runs one batch item (register / patch / replace / delete) through the same
authorization, registration-gate, validation, persistence, search-index, and
webhook steps the single-card HTTP handlers use. Each item's outcome is
captured into an AgentBatchItemResult so one failing item never aborts a job.

The processor runs inside the batch worker, outside any HTTP request, so it
cannot rebuild a user_context from scopes. Instead it consumes the submitter's
authorization snapshot (is_admin + ui_permissions) captured on the job at
submit time.
"""

import asyncio
import logging
from typing import Any

from fastapi import HTTPException

from ..schemas.agent_models import (
    REGISTRANT_ONLY_FIELDS,
    AgentBatchItem,
    AgentBatchItemResult,
    AgentCard,
    BatchItemOp,
)
from .agent_service import agent_service
from .registration_gate_service import check_registration_gate
from .webhook_service import send_registration_webhook

logger = logging.getLogger(__name__)


def _result(
    index: int,
    op: BatchItemOp,
    path: str | None,
    status: int,
    code: str | None = None,
    message: str | None = None,
) -> AgentBatchItemResult:
    """Build an AgentBatchItemResult, attaching an error block for failures."""
    error = None
    if status >= 400:
        error = {"code": code or "error", "message": message or ""}
    return AgentBatchItemResult(index=index, op=op, path=path, status=status, error=error)


async def _authorize(
    item: AgentBatchItem,
    submitted_by: str,
    is_admin: bool,
    ui_permissions: dict[str, list[str]],
) -> tuple[bool, str]:
    """Re-authorize a batch item using the submitter's persisted permissions.

    Mirrors the single-card handlers: publish_agent to register, modify_service
    + owner-or-admin to patch/replace, and the federated read-only guard plus
    owner-or-admin to delete.
    """
    from ..auth.dependencies import user_has_ui_permission_for_service

    if item.op == BatchItemOp.register:
        if not ui_permissions.get("publish_agent"):
            return False, "publish_agent permission required"
        return True, ""

    existing = await agent_service.get_agent_info(item.path)
    if not existing:
        return False, f"agent not found at '{item.path}'"

    sync_metadata = existing.sync_metadata or {}
    if sync_metadata.get("is_federated") or sync_metadata.get("is_read_only"):
        source_peer = sync_metadata.get("source_peer_id", "unknown peer registry")
        return False, f"agent is synced from {source_peer} and cannot be modified locally"

    if item.op in (BatchItemOp.patch, BatchItemOp.replace):
        if not user_has_ui_permission_for_service("modify_service", existing.name, ui_permissions):
            return False, f"modify_service permission required for {existing.name}"

    if not is_admin and existing.registered_by != submitted_by:
        return False, "you can only modify agents you registered"

    return True, ""


async def _validate(agent_card: AgentCard) -> tuple[bool, Any]:
    """Validate a built AgentCard without probing the endpoint (PUT parity)."""
    from ..utils.agent_validator import agent_validator

    result = await agent_validator.validate_agent_card(agent_card, verify_endpoint=False)
    return result.is_valid, result.errors


async def _run_gate(operation: str, payload: dict, source_api: str) -> tuple[bool, str]:
    """Run the registration gate for a batch item (no request headers)."""
    gate_result = await check_registration_gate(
        asset_type="agent",
        operation=operation,
        source_api=source_api,
        registration_payload=payload,
        raw_headers=[],
    )
    return gate_result.allowed, gate_result.error_message or "registration gate denied"


def _fire_webhook(event_type: str, card: dict, performed_by: str) -> None:
    """Fire a registration webhook fire-and-forget."""
    asyncio.create_task(
        send_registration_webhook(
            event_type=event_type,
            registration_type="agent",
            card_data=card,
            performed_by=performed_by,
        )
    )


def _build_card_from_request(request, path: str, existing: AgentCard | None) -> AgentCard:
    """Build an AgentCard from an AgentRegistrationRequest.

    Mirrors the construction in the register/PUT handlers. When `existing` is
    provided (replace), server-managed fields are preserved from it.
    """
    from ..schemas.agent_models import AgentProvider

    tag_list = [t.strip() for t in request.tags.split(",") if t.strip()]

    external_tag_list: list[str] = []
    if request.external_tags:
        if isinstance(request.external_tags, str):
            external_tag_list = [t.strip() for t in request.external_tags.split(",") if t.strip()]
        elif isinstance(request.external_tags, list):
            external_tag_list = [t.strip() for t in request.external_tags if t.strip()]

    provider_obj = None
    if request.provider:
        provider_obj = AgentProvider(
            organization=request.provider.get("organization", ""),
            url=request.provider.get("url", ""),
        )

    optional_kwargs: dict[str, Any] = {}
    if request.default_input_modes:
        optional_kwargs["default_input_modes"] = request.default_input_modes
    if request.default_output_modes:
        optional_kwargs["default_output_modes"] = request.default_output_modes

    capabilities = dict(request.capabilities) if request.capabilities else {}
    if request.streaming and "streaming" not in capabilities:
        capabilities["streaming"] = request.streaming
    if capabilities:
        optional_kwargs["capabilities"] = capabilities

    if existing is not None:
        # Replace: preserve server-managed fields from the existing card.
        optional_kwargs.update(
            registered_by=existing.registered_by,
            registered_at=existing.registered_at,
            is_enabled=existing.is_enabled,
            num_stars=existing.num_stars,
            rating_details=existing.rating_details,
            ans_metadata=existing.ans_metadata,
            health_status=existing.health_status,
            last_health_check=existing.last_health_check,
            sync_metadata=existing.sync_metadata,
        )
        if request.metadata:
            optional_kwargs["metadata"] = request.metadata
        elif existing.metadata:
            optional_kwargs["metadata"] = existing.metadata

    return AgentCard(
        protocol_version=request.protocol_version,
        name=request.name,
        description=request.description,
        url=request.url,
        path=path,
        version=request.version,
        status=request.status,
        provider=provider_obj,
        security_schemes=request.security_schemes or {},
        skills=request.skills or [],
        tags=tag_list,
        license=request.license,
        visibility=request.visibility,
        allowed_groups=request.allowed_groups,
        trust_level=request.trust_level,
        supported_protocol=request.supported_protocol,
        external_tags=external_tag_list,
        **optional_kwargs,
    )


async def _do_register(
    index: int,
    item: AgentBatchItem,
    submitted_by: str,
) -> AgentBatchItemResult:
    """Register a new agent (POST /api/agents/register parity)."""
    from ..api.agent_routes import _normalize_path

    request = item.card
    path = _normalize_path(request.path, request.name)

    if await agent_service.get_agent_info(path):
        return _result(index, item.op, path, 409, "conflict", f"path '{path}' already exists")

    card = _build_card_from_request(request, path, existing=None)
    card.registered_by = submitted_by

    is_valid, errors = await _validate(card)
    if not is_valid:
        return _result(index, item.op, path, 422, "validation_error", str(errors))

    allowed, reason = await _run_gate(
        "register", request.model_dump(mode="json"), "/api/agents/batch"
    )
    if not allowed:
        return _result(index, item.op, path, 403, "gate_denied", reason)

    await agent_service.register_agent(card)

    from ..search.service import faiss_service

    is_enabled = await agent_service.is_agent_enabled(path)
    await faiss_service.add_or_update_entity(path, card.model_dump(), "a2a_agent", is_enabled)

    _fire_webhook("registration", card.model_dump(mode="json"), submitted_by)
    return _result(index, item.op, path, 201)


async def _do_patch(
    index: int,
    item: AgentBatchItem,
    submitted_by: str,
) -> AgentBatchItemResult:
    """Apply a JSON Merge Patch to an existing agent (PATCH parity)."""
    from ..api.agent_routes import _normalize_path

    path = _normalize_path(item.path)
    existing = await agent_service.get_agent_info(path)
    if not existing:
        return _result(index, item.op, path, 404, "not_found", f"agent not found at '{path}'")

    # Merge only the fields the caller actually supplied. We cannot rely on
    # exclude_unset here: the job (and its patch items) is persisted to and
    # reloaded from MongoDB before the worker runs, and a model rebuilt from a
    # full dict has every field marked "set". Since AgentCardPatch cannot
    # express an explicit null (RFC 7396 null-to-delete is unsupported), a None
    # value is unambiguously "field absent", so dropping None reconstructs the
    # caller's intent whether the item is fresh or reloaded.
    patch_dict = {
        key: value
        for key, value in item.card.model_dump(by_alias=False).items()
        if value is not None
    }
    if not patch_dict:
        return _result(index, item.op, path, 400, "empty_patch", "empty patch body")

    merged_dict = {**existing.model_dump(), **patch_dict}
    try:
        merged = AgentCard(**merged_dict)
    except Exception as e:
        return _result(index, item.op, path, 422, "validation_error", str(e))

    for field in REGISTRANT_ONLY_FIELDS:
        setattr(merged, field, getattr(existing, field))

    is_valid, errors = await _validate(merged)
    if not is_valid:
        return _result(index, item.op, path, 422, "validation_error", str(errors))

    allowed, reason = await _run_gate("update", merged.model_dump(mode="json"), "/api/agents/batch")
    if not allowed:
        return _result(index, item.op, path, 403, "gate_denied", reason)

    await agent_service.update_agent(path, merged.model_dump())

    updated = await agent_service.get_agent_info(path)
    _fire_webhook("update", updated.model_dump(mode="json"), submitted_by)
    return _result(index, item.op, path, 200)


async def _do_replace(
    index: int,
    item: AgentBatchItem,
    submitted_by: str,
) -> AgentBatchItemResult:
    """Fully replace an existing agent card (PUT parity)."""
    from ..api.agent_routes import _normalize_path

    path = _normalize_path(item.path)
    existing = await agent_service.get_agent_info(path)
    if not existing:
        return _result(index, item.op, path, 404, "not_found", f"agent not found at '{path}'")

    try:
        card = _build_card_from_request(item.card, path, existing=existing)
    except Exception as e:
        return _result(index, item.op, path, 422, "validation_error", str(e))

    is_valid, errors = await _validate(card)
    if not is_valid:
        return _result(index, item.op, path, 422, "validation_error", str(errors))

    allowed, reason = await _run_gate(
        "update", item.card.model_dump(mode="json"), "/api/agents/batch"
    )
    if not allowed:
        return _result(index, item.op, path, 403, "gate_denied", reason)

    await agent_service.update_agent(path, card.model_dump())

    updated = await agent_service.get_agent_info(path)
    _fire_webhook("update", updated.model_dump(mode="json"), submitted_by)
    return _result(index, item.op, path, 200)


async def _do_delete(
    index: int,
    item: AgentBatchItem,
    submitted_by: str,
) -> AgentBatchItemResult:
    """Delete an existing agent (DELETE parity)."""
    from ..api.agent_routes import _normalize_path

    path = _normalize_path(item.path)
    existing = await agent_service.get_agent_info(path)
    if not existing:
        return _result(index, item.op, path, 404, "not_found", f"agent not found at '{path}'")

    success = await agent_service.remove_agent(path)
    if not success:
        return _result(index, item.op, path, 500, "internal", "failed to delete agent")

    from ..search.service import faiss_service

    await faiss_service.remove_entity(path)

    _fire_webhook("deletion", existing.model_dump(mode="json"), submitted_by)
    return _result(index, item.op, path, 204)


async def process_item(
    index: int,
    item: AgentBatchItem,
    submitted_by: str,
    is_admin: bool,
    ui_permissions: dict[str, list[str]],
) -> AgentBatchItemResult:
    """Authorize and run a single batch item, capturing any error as a result."""
    try:
        ok, reason = await _authorize(item, submitted_by, is_admin, ui_permissions)
        if not ok:
            path = getattr(item, "path", None)
            return _result(index, item.op, path, 403, "forbidden", reason)

        if item.op == BatchItemOp.register:
            return await _do_register(index, item, submitted_by)
        if item.op == BatchItemOp.patch:
            return await _do_patch(index, item, submitted_by)
        if item.op == BatchItemOp.replace:
            return await _do_replace(index, item, submitted_by)
        if item.op == BatchItemOp.delete:
            return await _do_delete(index, item, submitted_by)

        path = getattr(item, "path", None)
        return _result(index, item.op, path, 400, "unknown_op", f"unsupported op: {item.op}")
    except HTTPException as e:
        path = getattr(item, "path", None)
        return _result(index, item.op, path, e.status_code, "http_exception", str(e.detail))
    except Exception as e:
        logger.exception(f"Batch item {index} failed")
        path = getattr(item, "path", None)
        return _result(index, item.op, path, 500, "internal", str(e))
