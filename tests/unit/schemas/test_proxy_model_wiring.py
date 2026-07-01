"""Tests that the ProxyableMixin is correctly wired into every entity model.

Covers, per entity type:
- the storage AND request/patch models actually ACCEPT is_proxied/proxy_target_url
  (a mixin added only to storage models would let the API 422 or drop them);
- the SSRF egress guard fires at the model edge (a denied target -> ValidationError);
- the "target required when proxied" contract (proxied with no resolvable target
  -> ValidationError), including the per-type fallback (MCP -> proxy_pass_url,
  agent -> url) and the local-MCP exemption;
- virtual servers are alias-only (a proxy_target_url is rejected).
"""

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit

_DENIED = "http://169.254.169.254/"  # metadata endpoint
_OK = "https://backend.example.com/"


class TestServerInfoWiring:
    def test_accepts_and_falls_back_to_proxy_pass_url(self):
        from registry.core.schemas import ServerInfo

        s = ServerInfo(server_name="s", path="/s", proxy_pass_url=_OK, is_proxied=True)
        assert s.is_proxied is True

    def test_storage_model_is_read_safe(self):
        """ServerInfo (storage) does NOT raise on a denied target — reconstructed
        from the DB on every read. The edge is the request model (below)."""
        from registry.core.schemas import ServerInfo

        s = ServerInfo(
            server_name="s",
            path="/s",
            proxy_pass_url=_OK,
            is_proxied=True,
            proxy_target_url=_DENIED,
        )
        assert s.proxy_target_url == _DENIED

    def test_request_model_metadata_target_rejected(self):
        from registry.core.schemas import ServiceRegistrationRequest

        with pytest.raises(ValidationError):
            ServiceRegistrationRequest(
                name="s",
                path="/s",
                proxy_pass_url=_OK,
                is_proxied=True,
                proxy_target_url=_DENIED,
            )

    def test_proxied_without_target_rejected(self):
        from registry.core.schemas import ServerInfo

        with pytest.raises(ValidationError):
            ServerInfo(server_name="s", path="/s", is_proxied=True)  # remote, no url

    def test_local_deployment_proxied_is_noop_not_error(self):
        from registry.core.schemas import ServerInfo

        # local stdio server has no HTTP backend; is_proxied is a documented no-op
        s = ServerInfo(
            server_name="s",
            path="/s",
            deployment="local",
            local_runtime={"type": "uvx", "package": "x"},
            is_proxied=True,
        )
        assert s.is_proxied is True

    def test_request_model_accepts_fields(self):
        from registry.core.schemas import ServiceRegistrationRequest

        r = ServiceRegistrationRequest(
            name="s", path="/s", proxy_pass_url=_OK, is_proxied=True, proxy_target_url=_OK
        )
        assert r.proxy_target_url == _OK


class TestAgentWiring:
    def test_card_falls_back_to_url(self):
        from registry.schemas.agent_models import AgentCard

        a = AgentCard(name="a", description="d", url=_OK, version="1", is_proxied=True)
        assert a.is_proxied is True

    def test_card_storage_model_is_read_safe(self):
        """AgentCard (storage) does NOT raise on a denied target — it is
        reconstructed from the DB on every read. The API edge is the request
        model (below)."""
        from registry.schemas.agent_models import AgentCard

        a = AgentCard(
            name="a",
            description="d",
            url=_OK,
            version="1",
            is_proxied=True,
            proxy_target_url=_DENIED,
        )
        assert a.proxy_target_url == _DENIED

    def test_registration_request_metadata_target_rejected(self):
        """The request model (API edge) rejects a denied target."""
        from registry.schemas.agent_models import AgentRegistrationRequest

        with pytest.raises(ValidationError):
            AgentRegistrationRequest(
                name="a",
                description="d",
                url=_OK,
                version="1",
                supported_protocol="A2A",
                is_proxied=True,
                proxy_target_url=_DENIED,
            )

    def test_registration_request_accepts_fields(self):
        from registry.schemas.agent_models import AgentRegistrationRequest

        r = AgentRegistrationRequest(
            name="a",
            description="d",
            url=_OK,
            version="1",
            supported_protocol="A2A",
            is_proxied=True,
            proxy_target_url=_OK,
        )
        assert r.proxy_target_url == _OK

    def test_patch_accepts_and_guards(self):
        from registry.schemas.agent_models import AgentCardPatch

        # forbid-extra patch model must accept the fields
        p = AgentCardPatch(is_proxied=True, proxy_target_url=_OK)
        assert p.is_proxied is True
        # and still run the egress guard
        with pytest.raises(ValidationError):
            AgentCardPatch(proxy_target_url=_DENIED)


class TestSkillWiring:
    def test_card_missing_target_is_read_safe(self):
        """SkillCard (storage) loads cleanly with is_proxied+no target — a bypass
        write must not vanish on read; the route simply won't render."""
        from registry.schemas.skill_models import SkillCard

        s = SkillCard(
            path="/skills/x", name="x", description="d", skill_md_url="https://s/", is_proxied=True
        )
        assert s.is_proxied is True

    def test_registration_request_requires_explicit_target(self):
        """The request model (API edge) rejects is_proxied without a target."""
        from registry.schemas.skill_models import SkillRegistrationRequest

        with pytest.raises(ValidationError):
            SkillRegistrationRequest(
                name="x", description="d", skill_md_url="https://s/", is_proxied=True
            )

    def test_card_with_target_ok(self):
        from registry.schemas.skill_models import SkillCard

        s = SkillCard(
            path="/skills/x",
            name="x",
            description="d",
            skill_md_url="https://s/",
            is_proxied=True,
            proxy_target_url=_OK,
        )
        assert s.proxy_target_url == _OK

    def test_registration_request_accepts_fields(self):
        from registry.schemas.skill_models import SkillRegistrationRequest

        r = SkillRegistrationRequest(
            name="x",
            description="d",
            skill_md_url="https://s/",
            is_proxied=True,
            proxy_target_url=_OK,
        )
        assert r.is_proxied is True


class TestCustomEntityWiring:
    def test_record_missing_target_is_read_safe(self):
        """CustomEntityRecord (storage) loads cleanly with is_proxied+no target."""
        from registry.schemas.custom_entity_models import CustomEntityRecord

        r = CustomEntityRecord(entity_type="workflow", name="n", is_proxied=True)
        assert r.is_proxied is True

    def test_create_request_requires_explicit_target(self):
        """CustomEntityCreate (API edge) rejects is_proxied without a target."""
        from registry.schemas.custom_entity_models import CustomEntityCreate

        with pytest.raises(ValidationError):
            CustomEntityCreate(name="n", is_proxied=True)

    def test_record_with_target_ok(self):
        from registry.schemas.custom_entity_models import CustomEntityRecord

        r = CustomEntityRecord(
            entity_type="workflow", name="n", is_proxied=True, proxy_target_url=_OK
        )
        assert r.proxy_target_url == _OK

    def test_create_and_update_accept_and_guard(self):
        from registry.schemas.custom_entity_models import CustomEntityCreate, CustomEntityUpdate

        assert CustomEntityCreate(name="n", proxy_target_url=_OK).proxy_target_url == _OK
        assert CustomEntityUpdate(is_proxied=True).is_proxied is True
        with pytest.raises(ValidationError):
            CustomEntityCreate(name="n", proxy_target_url=_DENIED)
        with pytest.raises(ValidationError):
            CustomEntityUpdate(proxy_target_url=_DENIED)


class TestVirtualServerAliasOnly:
    def test_alias_only_accepts_is_proxied(self):
        from registry.schemas.virtual_server_models import VirtualServerConfig

        v = VirtualServerConfig(path="/virtual/x", server_name="x", is_proxied=True)
        assert v.is_proxied is True

    def test_rejects_proxy_target_url(self):
        from registry.schemas.virtual_server_models import VirtualServerConfig

        with pytest.raises(ValidationError):
            VirtualServerConfig(
                path="/virtual/x", server_name="x", is_proxied=True, proxy_target_url=_OK
            )


class TestAutoDisabledStateLoadsCleanly:
    """A model reconstructed from a stored doc in the auto-disabled state
    (is_proxied=True + proxy_disabled_reason set) must NOT raise — otherwise the
    entity throws on every read and silently vanishes from listings."""

    def test_agent_card_auto_disabled_loads(self):
        from registry.schemas.agent_models import AgentCard

        a = AgentCard(
            name="a",
            description="d",
            url=_OK,
            version="1",
            is_proxied=True,
            proxy_target_url=None,
            proxy_disabled_reason="target now resolves to a denied IP",
        )
        assert a.proxy_disabled_reason is not None

    def test_skill_card_auto_disabled_loads(self):
        from registry.schemas.skill_models import SkillCard

        s = SkillCard(
            path="/skills/x",
            name="x",
            description="d",
            skill_md_url="https://s/",
            is_proxied=True,
            proxy_target_url=None,
            proxy_disabled_reason="target now resolves to a denied IP",
        )
        assert s.proxy_disabled_reason is not None

    def test_custom_record_auto_disabled_loads(self):
        from registry.schemas.custom_entity_models import CustomEntityRecord

        r = CustomEntityRecord(
            entity_type="workflow",
            name="n",
            is_proxied=True,
            proxy_target_url=None,
            proxy_disabled_reason="target now resolves to a denied IP",
        )
        assert r.proxy_disabled_reason is not None


class TestServiceLayerRoundTrip:
    """The opt-in must survive request -> storage-model construction (the layer
    that previously dropped the fields)."""

    def test_skill_builder_carries_proxy_fields(self):
        from registry.schemas.skill_models import SkillRegistrationRequest
        from registry.services.skill_service import _build_skill_card

        req = SkillRegistrationRequest(
            name="x",
            description="d",
            skill_md_url="https://s/",
            is_proxied=True,
            proxy_target_url=_OK,
        )
        card = _build_skill_card(
            request=req,
            path="/skills/x",
            owner="bob",
            content_version=None,
            content_updated_at=None,
        )
        assert card.is_proxied is True
        assert card.proxy_target_url == _OK

    def test_custom_create_source_copies_proxy_fields(self):
        """create_record must pass the request's proxy fields into the record.

        Guards the specific two-line carry-through that was missing
        (a source-level assertion avoids brittle mocking of the service's cache/
        repo properties).
        """
        import inspect

        from registry.services import custom_entity_service

        src = inspect.getsource(custom_entity_service.CustomEntityService.create_record)
        assert "is_proxied=request.is_proxied" in src
        assert "proxy_target_url=request.proxy_target_url" in src

    def test_custom_update_source_copies_proxy_fields(self):
        """update_record must include the proxy fields in its updates dict."""
        import inspect

        from registry.services import custom_entity_service

        src = inspect.getsource(custom_entity_service.CustomEntityService.update_record)
        assert 'updates["is_proxied"]' in src
        assert 'updates["proxy_target_url"]' in src

    def test_agent_register_route_copies_proxy_fields(self):
        """The agent register route must pass the request's proxy fields to AgentCard."""
        import inspect

        from registry.api import agent_routes

        src = inspect.getsource(agent_routes)
        assert "is_proxied=request.is_proxied" in src
        assert "proxy_target_url=request.proxy_target_url" in src
