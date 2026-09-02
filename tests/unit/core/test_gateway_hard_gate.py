"""HARD GATE: no denied target can become a live nginx route.

The registration-time resolve-and-validate (SSRF layer 2) is the primary gate,
but a stored row can reach the render pipeline WITHOUT passing it — a federation
raw-write, a migration, a manual DB edit, or a row that predates the guard. This
suite drives the ACTUAL render feed (_generate_generic_proxy_blocks, which reads
_fetch_generic_proxied_resources -> resolve_proxy_target -> _safe_generic_block)
with adversarial rows a bypass could have planted, and asserts the composite
render-time defense holds: NO live generic block is emitted for a denied /
federated / disabled / targetless row, and a good row alongside it still renders.

This is the render half of the "don't enable the feature until the gate holds"
invariant; the network egress policy (layer 1) is the socket-level backstop.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from registry.core.nginx_service import NginxConfigService  # noqa: E402

pytestmark = pytest.mark.unit


def _service() -> NginxConfigService:
    return NginxConfigService()


async def _render(rows, allow_private=False):
    """Run the real render feed with `rows` as the raw list_proxied output.

    Patches each repo's list_proxied so the rows flow through the genuine
    resolve_proxy_target + _safe_generic_block path, exactly as at render time.
    """
    agent_repo = AsyncMock()
    skill_repo = AsyncMock()
    custom_repo = AsyncMock()
    # Bucket rows by entity_type so each repo returns its own.
    agent_repo.list_proxied.return_value = [r for r in rows if r.get("_repo") == "agent"]
    skill_repo.list_proxied.return_value = [r for r in rows if r.get("_repo") == "skill"]
    custom_repo.list_proxied.return_value = [r for r in rows if r.get("_repo") == "custom"]

    with patch("registry.core.nginx_service.settings") as s:
        s.gateway_generic_proxy_enabled = True
        s.auth_server_url = "http://auth-server:8888"
        s.gateway_generic_client_max_body_size = "1m"
        s.gateway_proxy_allow_private_targets = allow_private
        with (
            patch("registry.repositories.factory.get_agent_repository", return_value=agent_repo),
            patch("registry.repositories.factory.get_skill_repository", return_value=skill_repo),
            patch(
                "registry.repositories.factory.get_custom_entity_repository",
                return_value=custom_repo,
            ),
        ):
            return await _service()._generate_generic_proxy_blocks(set())


class TestHardGateDeniedTargetsNeverRender:
    async def test_metadata_ip_literal_target_never_renders(self):
        # A bypass-written skill pointing straight at the metadata IP.
        rows = [
            {
                "_repo": "skill",
                "path": "/skills/evil",
                "is_proxied": True,
                "proxy_target_url": "http://169.254.169.254/",
            },
            {
                "_repo": "skill",
                "path": "/skills/ok",
                "is_proxied": True,
                "proxy_target_url": "https://ok.example/",
            },
        ]
        blocks = await _render(rows)
        joined = "\n".join(blocks)
        assert "169.254.169.254" not in joined  # denied target dropped
        assert "/skill/skills/ok" in joined  # good sibling still renders
        assert len(blocks) == 1

    async def test_private_ip_target_dropped_when_flag_false(self):
        rows = [
            {
                "_repo": "custom",
                "entity_type": "workflow",
                "path": "/workflow/x",
                "is_proxied": True,
                "proxy_target_url": "http://10.0.0.5/",
            },
        ]
        assert await _render(rows, allow_private=False) == []

    async def test_non_http_scheme_target_dropped(self):
        rows = [
            {
                "_repo": "skill",
                "path": "/skills/f",
                "is_proxied": True,
                "proxy_target_url": "file:///etc/passwd",
            },
        ]
        assert await _render(rows) == []

    async def test_directive_breakout_target_dropped(self):
        rows = [
            {
                "_repo": "skill",
                "path": "/skills/x",
                "is_proxied": True,
                "proxy_target_url": 'https://x/";}\nlocation /evil {',
            },
        ]
        assert await _render(rows) == []


class TestHardGateFederatedAndDisabledNeverRender:
    async def test_federated_row_never_renders_even_if_locally_flipped(self):
        # A synced row with is_proxied flipped locally + a peer-supplied target.
        rows = [
            {
                "_repo": "agent",
                "path": "/agents/fed",
                "is_proxied": True,
                "url": "https://peer-backend.example/",
                "sync_metadata": {"is_federated": True},
            },
        ]
        assert await _render(rows) == []

    async def test_auto_disabled_row_never_renders(self):
        rows = [
            {
                "_repo": "skill",
                "path": "/skills/dis",
                "is_proxied": True,
                "proxy_target_url": "https://ok.example/",
                "proxy_disabled_reason": "target resolved to a denied IP",
            },
        ]
        assert await _render(rows) == []

    async def test_disabled_is_enabled_false_never_renders(self):
        rows = [
            {
                "_repo": "skill",
                "path": "/skills/off",
                "is_proxied": True,
                "proxy_target_url": "https://ok.example/",
                "is_enabled": False,
            },
        ]
        assert await _render(rows) == []

    async def test_targetless_proxied_row_never_renders(self):
        # is_proxied with no resolvable target (skill w/o proxy_target_url).
        rows = [
            {"_repo": "skill", "path": "/skills/none", "is_proxied": True},
        ]
        assert await _render(rows) == []


class TestHardGateFeatureFlagOff:
    async def test_flag_off_emits_nothing_and_queries_nothing(self):
        agent_repo = AsyncMock()
        with patch("registry.core.nginx_service.settings") as s:
            s.gateway_generic_proxy_enabled = False
            with patch(
                "registry.repositories.factory.get_agent_repository", return_value=agent_repo
            ):
                blocks = await _service()._generate_generic_proxy_blocks(set())
        assert blocks == []
        agent_repo.list_proxied.assert_not_called()
