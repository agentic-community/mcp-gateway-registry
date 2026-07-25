"""Tests for registry.services.agent_batch_item_processor (issue #956).

Covers per-item authorization and the register/patch/replace/delete handlers,
plus the top-level process_item dispatch and its exception capture. All external
collaborators (agent_service, validator, gate, webhook) are mocked so a
single failing item is exercised in isolation without MongoDB.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import TypeAdapter

from registry.schemas.agent_models import AgentBatchItem, AgentBatchRequest, BatchItemOp
from registry.services import agent_batch_item_processor as proc
from tests.fixtures.factories import AgentCardFactory

_ADAPTER = TypeAdapter(AgentBatchItem)

REGISTER_CARD = {
    "name": "new-agent",
    "url": "https://example.com/new",
    "supported_protocol": "a2a",
    "version": "1.0",
}

# Full agent-management grant set (canonical agent scopes, not the old
# cross-asset modify_service). Named ADMIN_PERMS for historical reasons; it is
# just "holds every agent scope for all agents".
ADMIN_PERMS = {
    "publish_agent": ["all"],
    "modify_agent": ["all"],
    "delete_agent": ["all"],
}


def _item(data):
    return _ADAPTER.validate_python(data)


def _existing(registered_by="alice", sync_metadata=None):
    return AgentCardFactory(
        name="existing-agent",
        path="/agents/existing",
        registered_by=registered_by,
        sync_metadata=sync_metadata,
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestAuthorize:
    async def test_register_requires_publish_permission(self):
        item = _item({"op": "register", "card": REGISTER_CARD})
        ok, reason = await proc._authorize(item, "alice", False, {})
        assert ok is False
        assert "publish_agent" in reason

    async def test_register_allowed_with_publish_permission(self):
        item = _item({"op": "register", "card": REGISTER_CARD})
        ok, reason = await proc._authorize(item, "alice", False, {"publish_agent": ["all"]})
        assert ok is True

    async def test_patch_missing_agent_denied(self):
        item = _item({"op": "patch", "path": "/agents/x", "card": {"description": "d"}})
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=None)):
            ok, reason = await proc._authorize(item, "alice", True, ADMIN_PERMS)
        assert ok is False
        assert "not found" in reason

    async def test_federated_agent_blocked(self):
        item = _item({"op": "delete", "path": "/agents/existing"})
        existing = _existing(sync_metadata={"is_federated": True, "source_peer_id": "peer-7"})
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=existing)):
            ok, reason = await proc._authorize(item, "alice", True, ADMIN_PERMS)
        assert ok is False
        assert "peer-7" in reason

    async def test_patch_requires_modify_agent_permission(self):
        # Owner of the agent, but no modify_agent grant -> denied (scope half).
        item = _item({"op": "patch", "path": "/agents/existing", "card": {"description": "d"}})
        existing = _existing(registered_by="alice")
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=existing)):
            ok, reason = await proc._authorize(item, "alice", False, {})
        assert ok is False
        assert "modify_agent" in reason

    async def test_patch_modify_service_does_not_satisfy_agent(self):
        # The cross-asset bug: a SERVER scope must not authorize an agent patch on
        # the batch path (it previously did via user_has_ui_permission_for_service).
        item = _item({"op": "patch", "path": "/agents/existing", "card": {"description": "d"}})
        existing = _existing(registered_by="alice")
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=existing)):
            ok, reason = await proc._authorize(
                item, "alice", False, {"modify_service": ["all"], "toggle_service": ["all"]}
            )
        assert ok is False
        assert "modify_agent" in reason

    async def test_patch_allowed_with_modify_agent_and_ownership(self):
        item = _item({"op": "patch", "path": "/agents/existing", "card": {"description": "d"}})
        existing = _existing(registered_by="alice")
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=existing)):
            ok, reason = await proc._authorize(item, "alice", False, {"modify_agent": ["all"]})
        assert ok is True

    async def test_delete_requires_delete_agent_permission(self):
        # The bypass kiro found: batch delete must require delete_agent, not just
        # ownership. Owner of the agent but no delete_agent grant -> denied. This
        # is the batch analogue of the single-card owner-without-scope denial.
        item = _item({"op": "delete", "path": "/agents/existing"})
        existing = _existing(registered_by="alice")
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=existing)):
            ok, reason = await proc._authorize(item, "alice", False, {"publish_agent": ["all"]})
        assert ok is False
        assert "delete_agent" in reason

    async def test_delete_allowed_with_delete_agent_and_ownership(self):
        item = _item({"op": "delete", "path": "/agents/existing"})
        existing = _existing(registered_by="alice")
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=existing)):
            ok, reason = await proc._authorize(item, "alice", False, {"delete_agent": ["all"]})
        assert ok is True

    async def test_delete_scope_without_ownership_denied(self):
        # Non-owner holding delete_agent is still denied (AND-owner half).
        item = _item({"op": "delete", "path": "/agents/existing"})
        existing = _existing(registered_by="bob")
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=existing)):
            ok, reason = await proc._authorize(item, "alice", False, {"delete_agent": ["all"]})
        assert ok is False
        assert "only delete agents you registered" in reason

    async def test_non_owner_non_admin_denied(self):
        # Non-owner WITH the delete scope is denied on the ownership half.
        item = _item({"op": "delete", "path": "/agents/existing"})
        existing = _existing(registered_by="bob")
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=existing)):
            ok, reason = await proc._authorize(item, "alice", False, ADMIN_PERMS)
        assert ok is False
        assert "only delete agents you registered" in reason

    async def test_admin_can_delete_any(self):
        item = _item({"op": "delete", "path": "/agents/existing"})
        existing = _existing(registered_by="bob")
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=existing)):
            ok, reason = await proc._authorize(item, "alice", True, ADMIN_PERMS)
        assert ok is True

    async def test_admin_delete_bypasses_scope_and_ownership(self):
        # Admin with NO agent scopes at all still deletes (is_admin bypass).
        item = _item({"op": "delete", "path": "/agents/existing"})
        existing = _existing(registered_by="bob")
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=existing)):
            ok, reason = await proc._authorize(item, "alice", True, {})
        assert ok is True


@pytest.mark.unit
@pytest.mark.asyncio
class TestDoRegister:
    async def test_conflict_when_path_exists(self):
        item = _item({"op": "register", "card": REGISTER_CARD})
        with patch.object(
            proc.agent_service, "get_agent_info", AsyncMock(return_value=_existing())
        ):
            result = await proc._do_register(0, item, "alice")
        assert result.status == 409
        assert result.error["code"] == "conflict"

    async def test_validation_failure_returns_422(self):
        item = _item({"op": "register", "card": REGISTER_CARD})
        with (
            patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=None)),
            patch.object(proc, "_validate", AsyncMock(return_value=(False, ["bad url"]))),
        ):
            result = await proc._do_register(0, item, "alice")
        assert result.status == 422
        assert result.error["code"] == "validation_error"

    async def test_gate_denied_returns_403(self):
        item = _item({"op": "register", "card": REGISTER_CARD})
        with (
            patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=None)),
            patch.object(proc, "_validate", AsyncMock(return_value=(True, []))),
            patch.object(proc, "_run_gate", AsyncMock(return_value=(False, "policy"))),
        ):
            result = await proc._do_register(0, item, "alice")
        assert result.status == 403
        assert result.error["code"] == "gate_denied"

    async def test_success_returns_201_and_fires_webhook(self):
        item = _item({"op": "register", "card": REGISTER_CARD})
        with (
            patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=None)),
            patch.object(proc, "_validate", AsyncMock(return_value=(True, []))),
            patch.object(proc, "_run_gate", AsyncMock(return_value=(True, ""))),
            patch.object(proc.agent_service, "register_agent", AsyncMock()),
            patch.object(proc.agent_service, "is_agent_enabled", AsyncMock(return_value=True)),
            patch.object(proc, "_fire_webhook") as fire,
        ):
            result = await proc._do_register(0, item, "alice")
        assert result.status == 201
        fire.assert_called_once()
        assert fire.call_args.args[0] == "registration"


@pytest.mark.unit
@pytest.mark.asyncio
class TestDoPatch:
    async def test_not_found_returns_404(self):
        item = _item({"op": "patch", "path": "/agents/x", "card": {"description": "d"}})
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=None)):
            result = await proc._do_patch(0, item, "alice")
        assert result.status == 404

    async def test_empty_patch_returns_400(self):
        item = _item({"op": "patch", "path": "/agents/existing", "card": {}})
        with patch.object(
            proc.agent_service, "get_agent_info", AsyncMock(return_value=_existing())
        ):
            result = await proc._do_patch(0, item, "alice")
        assert result.status == 400
        assert result.error["code"] == "empty_patch"

    async def test_success_returns_200(self):
        item = _item(
            {"op": "patch", "path": "/agents/existing", "card": {"description": "updated"}}
        )
        existing = _existing()
        with (
            patch.object(
                proc.agent_service,
                "get_agent_info",
                AsyncMock(side_effect=[existing, existing]),
            ),
            patch.object(proc, "_validate", AsyncMock(return_value=(True, []))),
            patch.object(proc, "_run_gate", AsyncMock(return_value=(True, ""))),
            patch.object(proc.agent_service, "update_agent", AsyncMock()),
            patch.object(proc, "_fire_webhook") as fire,
        ):
            result = await proc._do_patch(0, item, "alice")
        assert result.status == 200
        assert fire.call_args.args[0] == "update"

    async def test_reloaded_item_merges_only_supplied_fields(self):
        """A patch item round-tripped through Mongo must still merge cleanly.

        Regression for the batch-only failure where the job (and its patch
        items) is persisted to and reloaded from MongoDB before the worker
        runs. A model rebuilt from a full dict has every field marked "set",
        so exclude_unset no longer narrows the patch; the unset None defaults
        would clobber required fields on the existing card and AgentCard(**merged)
        would raise 13 validation errors. _do_patch must drop None values instead.
        """
        request = AgentBatchRequest(
            items=[{"op": "patch", "path": "/agents/existing", "card": {"description": "updated"}}]
        )
        reloaded = AgentBatchRequest(**request.model_dump(mode="json"))
        item = reloaded.items[0]
        existing = _existing()
        captured: dict = {}

        async def _capture_update(path, data):
            captured["data"] = data

        with (
            patch.object(
                proc.agent_service,
                "get_agent_info",
                AsyncMock(side_effect=[existing, existing]),
            ),
            patch.object(proc, "_validate", AsyncMock(return_value=(True, []))),
            patch.object(proc, "_run_gate", AsyncMock(return_value=(True, ""))),
            patch.object(
                proc.agent_service, "update_agent", AsyncMock(side_effect=_capture_update)
            ),
            patch.object(proc, "_fire_webhook"),
        ):
            result = await proc._do_patch(0, item, "alice")

        assert result.status == 200
        # The patched field changed; required fields were preserved, not nulled.
        assert captured["data"]["description"] == "updated"
        assert captured["data"]["name"] == existing.name
        assert captured["data"]["url"] == existing.url


@pytest.mark.unit
@pytest.mark.asyncio
class TestDoDelete:
    async def test_not_found_returns_404(self):
        item = _item({"op": "delete", "path": "/agents/x"})
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=None)):
            result = await proc._do_delete(0, item, "alice")
        assert result.status == 404

    async def test_remove_failure_returns_500(self):
        item = _item({"op": "delete", "path": "/agents/existing"})
        with (
            patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=_existing())),
            patch.object(proc.agent_service, "remove_agent", AsyncMock(return_value=False)),
        ):
            result = await proc._do_delete(0, item, "alice")
        assert result.status == 500

    async def test_success_returns_204(self):
        item = _item({"op": "delete", "path": "/agents/existing"})
        with (
            patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=_existing())),
            patch.object(proc.agent_service, "remove_agent", AsyncMock(return_value=True)),
            patch.object(proc, "_fire_webhook") as fire,
        ):
            result = await proc._do_delete(0, item, "alice")
        assert result.status == 204
        assert fire.call_args.args[0] == "deletion"


@pytest.mark.unit
@pytest.mark.asyncio
class TestProcessItemDispatch:
    async def test_unauthorized_returns_403(self):
        item = _item({"op": "register", "card": REGISTER_CARD})
        result = await proc.process_item(0, item, "alice", False, {})
        assert result.status == 403
        assert result.error["code"] == "forbidden"

    async def test_dispatches_to_register(self):
        item = _item({"op": "register", "card": REGISTER_CARD})
        with (
            patch.object(proc, "_authorize", AsyncMock(return_value=(True, ""))),
            patch.object(
                proc, "_do_register", AsyncMock(return_value=MagicMock(status=201))
            ) as do_reg,
        ):
            result = await proc.process_item(0, item, "alice", True, ADMIN_PERMS)
        do_reg.assert_awaited_once()
        assert result.status == 201

    async def test_http_exception_captured(self):
        item = _item({"op": "delete", "path": "/agents/x"})
        with (
            patch.object(proc, "_authorize", AsyncMock(return_value=(True, ""))),
            patch.object(
                proc,
                "_do_delete",
                AsyncMock(side_effect=HTTPException(status_code=409, detail="boom")),
            ),
        ):
            result = await proc.process_item(0, item, "alice", True, ADMIN_PERMS)
        assert result.status == 409
        assert result.error["code"] == "http_exception"

    async def test_unexpected_exception_captured_as_500(self):
        item = _item({"op": "delete", "path": "/agents/x"})
        with (
            patch.object(proc, "_authorize", AsyncMock(return_value=(True, ""))),
            patch.object(proc, "_do_delete", AsyncMock(side_effect=RuntimeError("kaboom"))),
        ):
            result = await proc.process_item(0, item, "alice", True, ADMIN_PERMS)
        assert result.status == 500
        assert result.error["code"] == "internal"
        assert "kaboom" in result.error["message"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestDoReplace:
    async def test_not_found_returns_404(self):
        item = _item({"op": "replace", "path": "/agents/x", "card": REGISTER_CARD})
        with patch.object(proc.agent_service, "get_agent_info", AsyncMock(return_value=None)):
            result = await proc._do_replace(0, item, "alice")
        assert result.status == 404

    async def test_success_returns_200_and_preserves_owner(self):
        item = _item({"op": "replace", "path": "/agents/existing", "card": REGISTER_CARD})
        existing = _existing(registered_by="origowner")
        captured = {}
        with (
            patch.object(
                proc.agent_service,
                "get_agent_info",
                AsyncMock(side_effect=[existing, existing]),
            ),
            patch.object(proc, "_validate", AsyncMock(return_value=(True, []))),
            patch.object(proc, "_run_gate", AsyncMock(return_value=(True, ""))),
            patch.object(
                proc.agent_service,
                "update_agent",
                AsyncMock(side_effect=lambda p, d: captured.update(d)),
            ),
            patch.object(proc, "_fire_webhook"),
        ):
            result = await proc._do_replace(0, item, "alice")
        assert result.status == 200
        # Replace preserves server-managed registered_by from the existing card.
        assert captured["registered_by"] == "origowner"


@pytest.mark.unit
class TestBuildCardFromRequest:
    def test_register_build_uses_submitter_fields(self):
        request = _item({"op": "register", "card": REGISTER_CARD}).card
        card = proc._build_card_from_request(request, "/agents/new", existing=None)
        assert card.name == "new-agent"
        assert card.path == "/agents/new"

    def test_replace_build_preserves_server_fields_from_existing(self):
        request = _item({"op": "replace", "path": "/agents/existing", "card": REGISTER_CARD}).card
        existing = _existing(registered_by="origowner")
        existing.num_stars = 4.5
        card = proc._build_card_from_request(request, "/agents/existing", existing=existing)
        assert card.registered_by == "origowner"
        assert card.num_stars == 4.5

    def test_comma_separated_tags_split(self):
        data = {**REGISTER_CARD, "tags": "a, b, c"}
        request = _item({"op": "register", "card": data}).card
        card = proc._build_card_from_request(request, "/agents/new", existing=None)
        assert card.tags == ["a", "b", "c"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestSmallHelpers:
    async def test_validate_delegates_to_validator(self):
        validation = MagicMock(is_valid=True, errors=[])
        card = _existing()
        with patch("registry.utils.agent_validator.agent_validator") as v:
            v.validate_agent_card = AsyncMock(return_value=validation)
            ok, errors = await proc._validate(card)
        assert ok is True
        # PUT parity: endpoint probing is disabled for batch validation.
        v.validate_agent_card.assert_awaited_once()
        assert v.validate_agent_card.await_args.kwargs["verify_endpoint"] is False

    async def test_run_gate_returns_allowed(self):
        gate = MagicMock(allowed=True, error_message=None)
        with patch.object(proc, "check_registration_gate", AsyncMock(return_value=gate)):
            allowed, reason = await proc._run_gate("register", {}, "/api/agents/batch")
        assert allowed is True

    async def test_run_gate_denied_returns_message(self):
        gate = MagicMock(allowed=False, error_message="blocked by policy")
        with patch.object(proc, "check_registration_gate", AsyncMock(return_value=gate)):
            allowed, reason = await proc._run_gate("register", {}, "/api/agents/batch")
        assert allowed is False
        assert reason == "blocked by policy"

    async def test_fire_webhook_creates_task(self):
        with patch.object(proc, "send_registration_webhook", AsyncMock()):
            proc._fire_webhook("registration", {"name": "x"}, "alice")
            # Give the created task a chance to run so it doesn't warn on GC.
            import asyncio

            await asyncio.sleep(0)


@pytest.mark.unit
def test_result_helper_attaches_error_only_on_failure():
    ok = proc._result(0, BatchItemOp.delete, "/agents/x", 204)
    assert ok.error is None
    bad = proc._result(1, BatchItemOp.delete, "/agents/x", 404, "not_found", "gone")
    assert bad.error == {"code": "not_found", "message": "gone"}
