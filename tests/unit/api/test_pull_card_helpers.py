"""Unit tests for the A2A pull-card helper functions in agent_routes.py.

Covers (per testing.md sections 6.1-6.5):
- _normalize_remote_card_keys: camelCase -> snake_case
- _compute_card_diff: A2A-spec-only field diff
- _build_safe_card_updates: registrant-only protection (S5)
- _fetch_remote_agent_card: HTTP fetch + 502 mapping + 1 MiB cap (S1)
- the A2A_SPEC_FIELDS / REGISTRANT_ONLY_FIELDS disjoint invariant (6.4.1)

The remote fetch is mocked with unittest.mock (the project does not depend on
respx); httpx.AsyncClient is patched directly, mirroring the health-check tests.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from registry.api.agent_routes import (
    A2A_SPEC_FIELDS,
    _build_safe_card_updates,
    _compute_card_diff,
    _fetch_remote_agent_card,
    _normalize_remote_card_keys,
)
from registry.schemas.agent_models import REGISTRANT_ONLY_FIELDS, PullCardFieldChange
from tests.fixtures.factories import AgentCardFactory


def _agent(**overrides):
    """Build a valid A2A AgentCard with sensible defaults for diffing."""
    defaults = {
        "name": "agent",
        "description": "d",
        "version": "1.0.0",
        "url": "http://localhost:9000/a",
        "skills": [],
        "registered_by": "testuser",
        "supported_protocol": "a2a",
    }
    defaults.update(overrides)
    return AgentCardFactory(**defaults)


def _mock_async_client(response=None, exc=None):
    """Build a MagicMock standing in for httpx.AsyncClient(...) as a context manager."""
    client_factory = MagicMock()
    instance = AsyncMock()
    if exc is not None:
        instance.get.side_effect = exc
    else:
        instance.get.return_value = response
    client_factory.return_value.__aenter__ = AsyncMock(return_value=instance)
    client_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return client_factory


def _http_response(status_code=200, content=b"{}"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# 6.1 _normalize_remote_card_keys
# ---------------------------------------------------------------------------
class TestNormalizeRemoteCardKeys:
    def test_611_camel_to_snake(self):
        result = _normalize_remote_card_keys(
            {"protocolVersion": "1.0", "defaultInputModes": ["text/plain"]}
        )
        assert result == {
            "protocol_version": "1.0",
            "default_input_modes": ["text/plain"],
        }

    def test_612_already_snake_unchanged(self):
        card = {"name": "a", "version": "1.0.0"}
        assert _normalize_remote_card_keys(card) == card

    def test_613_unknown_key_passthrough(self):
        assert _normalize_remote_card_keys({"customField": 7}) == {"customField": 7}

    def test_614_empty(self):
        assert _normalize_remote_card_keys({}) == {}


# ---------------------------------------------------------------------------
# 6.2 _compute_card_diff
# ---------------------------------------------------------------------------
class TestComputeCardDiff:
    def test_621_identical_is_empty(self):
        agent = _agent()
        assert _compute_card_diff(agent, {"name": "agent", "version": "1.0.0"}) == []

    def test_622_version_differs(self):
        changes = _compute_card_diff(_agent(), {"version": "2.0.0"})
        assert [c.field for c in changes] == ["version"]
        assert changes[0].current_value == "1.0.0"
        assert changes[0].remote_value == "2.0.0"

    def test_623_three_fields_differ(self):
        changes = _compute_card_diff(
            _agent(), {"name": "new", "description": "e", "version": "9.9"}
        )
        assert sorted(c.field for c in changes) == ["description", "name", "version"]

    def test_624_skills_changed(self):
        changes = _compute_card_diff(_agent(), {"skills": [{"id": "s1", "name": "Search"}]})
        assert [c.field for c in changes] == ["skills"]

    def test_625_provider_added_compared_as_dict(self):
        # current provider is a Pydantic sub-model; remote is a dict. They must
        # be normalized to dicts before comparison.
        changes = _compute_card_diff(
            _agent(), {"provider": {"organization": "acme", "url": "http://acme"}}
        )
        assert [c.field for c in changes] == ["provider"]

    def test_626_security_schemes_changed(self):
        changes = _compute_card_diff(_agent(), {"security_schemes": {"k": {"type": "apiKey"}}})
        assert [c.field for c in changes] == ["security_schemes"]

    def test_627_registry_fields_ignored(self):
        # Remote echoes registry-extension fields; none should appear in the diff.
        changes = _compute_card_diff(
            _agent(),
            {"tags": ["x"], "visibility": "private", "trust_level": "verified"},
        )
        assert changes == []

    def test_628_omitted_field_no_change(self):
        # A field the remote does not send is not compared.
        assert _compute_card_diff(_agent(), {"name": "agent"}) == []


# ---------------------------------------------------------------------------
# 6.3 _build_safe_card_updates (S5)
# ---------------------------------------------------------------------------
class TestBuildSafeCardUpdates:
    def test_631_normal_fields(self):
        changes = [
            PullCardFieldChange(field="version", current_value="1", remote_value="2"),
            PullCardFieldChange(field="name", current_value="a", remote_value="b"),
        ]
        assert _build_safe_card_updates(changes) == {"version": "2", "name": "b"}

    def test_632_registrant_only_raises_400(self):
        changes = [PullCardFieldChange(field="num_stars", current_value=1, remote_value=999)]
        with pytest.raises(HTTPException) as exc_info:
            _build_safe_card_updates(changes)
        assert exc_info.value.status_code == 400
        assert "registrant-only" in str(exc_info.value.detail)

    def test_633_empty_changes(self):
        assert _build_safe_card_updates([]) == {}


# ---------------------------------------------------------------------------
# 6.4 Invariant
# ---------------------------------------------------------------------------
def test_641_a2a_and_registrant_fields_disjoint():
    """The S5 guard is a defensive backstop only when these sets never overlap."""
    assert A2A_SPEC_FIELDS.isdisjoint(REGISTRANT_ONLY_FIELDS)


# ---------------------------------------------------------------------------
# 6.5 _fetch_remote_agent_card (httpx mocked)
# ---------------------------------------------------------------------------
class TestFetchRemoteAgentCard:
    async def test_651_valid_object(self):
        body = json.dumps({"name": "a"}).encode()
        with patch.object(httpx, "AsyncClient", _mock_async_client(_http_response(200, body))):
            card, url = await _fetch_remote_agent_card("http://h:9000/a")
        assert card == {"name": "a"}
        assert url.endswith("/.well-known/agent-card.json")

    async def test_652_json_array_rejected(self):
        with patch.object(httpx, "AsyncClient", _mock_async_client(_http_response(200, b"[]"))):
            with pytest.raises(HTTPException) as e:
                await _fetch_remote_agent_card("http://h:9000/a")
        assert e.value.status_code == 502
        assert "not a JSON object" in str(e.value.detail)

    async def test_653_non_200(self):
        with patch.object(httpx, "AsyncClient", _mock_async_client(_http_response(404, b""))):
            with pytest.raises(HTTPException) as e:
                await _fetch_remote_agent_card("http://h:9000/a")
        assert e.value.status_code == 502
        assert "HTTP 404" in str(e.value.detail)

    async def test_654_timeout(self):
        client = _mock_async_client(exc=httpx.TimeoutException("t"))
        with patch.object(httpx, "AsyncClient", client):
            with pytest.raises(HTTPException) as e:
                await _fetch_remote_agent_card("http://h:9000/a")
        assert e.value.status_code == 502
        assert "Timeout fetching" in str(e.value.detail)

    async def test_655_connect_error(self):
        client = _mock_async_client(exc=httpx.ConnectError("c"))
        with patch.object(httpx, "AsyncClient", client):
            with pytest.raises(HTTPException) as e:
                await _fetch_remote_agent_card("http://h:9000/a")
        assert e.value.status_code == 502
        assert "Failed to fetch" in str(e.value.detail)

    async def test_656_invalid_json(self):
        with patch.object(httpx, "AsyncClient", _mock_async_client(_http_response(200, b"{bad"))):
            with pytest.raises(HTTPException) as e:
                await _fetch_remote_agent_card("http://h:9000/a")
        assert e.value.status_code == 502
        assert "Invalid response" in str(e.value.detail)

    async def test_657_exceeds_size_cap(self):
        # S1: a body over 1 MiB is rejected before json.loads runs.
        oversized = b'{"name":"' + b"a" * 1_048_600 + b'"}'
        with patch.object(httpx, "AsyncClient", _mock_async_client(_http_response(200, oversized))):
            with pytest.raises(HTTPException) as e:
                await _fetch_remote_agent_card("http://h:9000/a")
        assert e.value.status_code == 502
        assert "exceeds 1048576 bytes" in str(e.value.detail)
