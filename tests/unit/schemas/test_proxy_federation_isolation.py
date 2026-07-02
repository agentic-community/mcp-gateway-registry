"""Federation isolation for proxy config (owner decision: proxy is local-only).

Proxy fields must never cross the federation boundary in either direction:
- INGEST: a synced entity can never become a local gateway route, and a peer
  cannot plant an SSRF proxy_target_url.
- EXPORT: we never advertise our proxy config (esp. the internal
  proxy_target_url) to peers.

The key-drop (not force-false) semantics are load-bearing: on an update re-sync a
peer must not clobber a local admin's opt-in, so the fields are simply absent
from the synced payload rather than set to False.
"""

import pytest

from registry.schemas.proxy_mixin import (
    PROXY_FIELD_NAMES,
    strip_proxy_fields,
)

pytestmark = pytest.mark.unit


class TestStripProxyFields:
    def test_removes_all_proxy_keys(self):
        doc = {
            "name": "x",
            "path": "/x",
            "is_proxied": True,
            "proxy_target_url": "https://internal.example/",
            "proxy_resolved_ips": ["1.2.3.4"],
            "proxy_target_host": "internal.example",
            "proxy_disabled_reason": "blocked",
        }
        out = strip_proxy_fields(doc)
        assert set(out) == {"name", "path"}
        for k in PROXY_FIELD_NAMES:
            assert k not in out

    def test_preserves_non_proxy_fields(self):
        doc = {"name": "x", "path": "/x", "tags": ["a"], "is_enabled": True}
        assert strip_proxy_fields(doc) == doc

    def test_does_not_force_false_key_dropped_not_set(self):
        # Key-drop, NOT force-false: is_proxied must be ABSENT, not present-and-False,
        # so an update re-sync leaves an existing stored opt-in untouched.
        doc = {"name": "x", "is_proxied": True, "proxy_target_url": "https://x/"}
        out = strip_proxy_fields(doc)
        assert "is_proxied" not in out
        assert "proxy_target_url" not in out

    def test_no_proxy_keys_returns_input_unchanged(self):
        doc = {"name": "x", "path": "/x"}
        assert strip_proxy_fields(doc) is doc  # fast path: same object

    def test_does_not_mutate_input(self):
        doc = {"name": "x", "is_proxied": True}
        strip_proxy_fields(doc)
        assert doc["is_proxied"] is True  # original untouched


class TestExportStripsProxyFields:
    def test_item_to_dict_strips_from_dict(self):
        from registry.api.federation_export_routes import _item_to_dict

        server = {
            "path": "/s",
            "server_name": "s",
            "is_proxied": True,
            "proxy_target_url": "https://internal.example/",
        }
        out = _item_to_dict(server)
        assert "is_proxied" not in out
        assert "proxy_target_url" not in out
        assert out["server_name"] == "s"

    def test_item_to_dict_strips_from_model(self):
        from registry.api.federation_export_routes import _item_to_dict
        from registry.schemas.agent_models import AgentCard

        card = AgentCard(
            name="a",
            description="d",
            url="https://a.example/",
            version="1",
            is_proxied=True,
            proxy_target_url="https://internal.example/",
        )
        out = _item_to_dict(card)
        assert "is_proxied" not in out
        assert "proxy_target_url" not in out
        assert out["name"] == "a"


class TestIngestStripsProxyFields:
    """Source-level guards that every peer-content ingest site strips, and that
    the local re-persist sites (orphan/override) do NOT (they'd clobber a local
    admin's opt-in)."""

    def test_peer_sync_prep_strips(self):
        import inspect

        from registry.services import peer_federation_service as m

        for fn in (
            m.PeerFederationService._store_synced_servers,
            m.PeerFederationService._store_synced_agents,
        ):
            assert "strip_proxy_fields(" in inspect.getsource(fn)

    def test_federation_routes_ingest_strips(self):
        import inspect

        from registry.api import federation_routes

        src = inspect.getsource(federation_routes)
        # 4 peer-content ingest loops (Anthropic servers + AgentCore srv/agent/skill)
        assert src.count("strip_proxy_fields(") >= 4

    def test_ard_skill_ingest_strips(self):
        """ARD catalog-crawl skill ingest is peer content and must strip too."""
        import inspect

        from registry.services import ard_ingestion_service as m

        assert "strip_proxy_fields(" in inspect.getsource(m.ArdIngestionService._store_skills)


class TestFederatedEntityNeverResolvesToRoute:
    """ABSOLUTE isolation: resolve_proxy_target returns None for a federated
    entity even if is_proxied was flipped locally after ingest — so there is no
    path from peer-supplied data (proxy_pass_url/url) to a live gateway route."""

    def test_federated_server_with_local_is_proxied_does_not_resolve(self):
        from registry.schemas.proxy_mixin import resolve_proxy_target

        doc = {
            "is_proxied": True,  # locally flipped on a synced record
            "proxy_pass_url": "https://peer-backend.example/",  # peer-supplied
            "deployment": "remote",
            "sync_metadata": {"is_federated": True, "source_peer_id": "p1"},
        }
        assert resolve_proxy_target("mcp_server", doc) is None

    def test_federated_agent_does_not_resolve(self):
        from registry.schemas.proxy_mixin import resolve_proxy_target

        doc = {
            "is_proxied": True,
            "url": "https://peer-agent.example/",
            "sync_metadata": {"is_federated": True},
        }
        assert resolve_proxy_target("a2a_agent", doc) is None

    def test_non_federated_server_still_resolves(self):
        """A LOCAL (non-federated) proxied server resolves normally."""
        from registry.schemas.proxy_mixin import resolve_proxy_target

        doc = {
            "is_proxied": True,
            "proxy_pass_url": "https://local-backend.example/",
            "deployment": "remote",
        }
        assert resolve_proxy_target("mcp_server", doc) == "https://local-backend.example/"

    def test_non_federated_sync_metadata_shape_is_tolerated(self):
        """A non-dict / absent sync_metadata must not crash the resolve."""
        from registry.schemas.proxy_mixin import resolve_proxy_target

        doc = {
            "is_proxied": True,
            "proxy_target_url": "https://local.example/",
            "sync_metadata": None,
        }
        assert resolve_proxy_target("skill", doc) == "https://local.example/"


class TestDisabledEntityDoesNotResolve:
    """A proxied-but-disabled entity gets no route; is_enabled gates only when
    the caller supplies it (list_proxied projects it), else it's not assumed."""

    def test_disabled_returns_none(self):
        from registry.schemas.proxy_mixin import resolve_proxy_target

        doc = {"is_proxied": True, "proxy_target_url": "https://t/", "is_enabled": False}
        assert resolve_proxy_target("skill", doc) is None

    def test_enabled_resolves(self):
        from registry.schemas.proxy_mixin import resolve_proxy_target

        doc = {"is_proxied": True, "proxy_target_url": "https://t/", "is_enabled": True}
        assert resolve_proxy_target("skill", doc) == "https://t/"

    def test_absent_is_enabled_is_not_assumed_disabled(self):
        """A partial projection without is_enabled must not be treated as disabled."""
        from registry.schemas.proxy_mixin import resolve_proxy_target

        doc = {"is_proxied": True, "proxy_target_url": "https://t/"}
        assert resolve_proxy_target("skill", doc) == "https://t/"

    def test_local_repersist_does_not_strip(self):
        """_mark_orphaned / _set_local_override read a LOCAL record and re-persist
        it; stripping there would wipe a local admin's proxy opt-in. They must NOT
        call strip_proxy_fields."""
        import inspect

        from registry.services import peer_federation_service as m

        for name in ("mark_item_as_orphaned", "set_local_override"):
            fn = getattr(m.PeerFederationService, name, None)
            assert fn is not None, f"expected method {name} to exist"
            assert "strip_proxy_fields(" not in inspect.getsource(fn), name
