"""Unit tests for EntraIdProvider Graph group-overage handling (#929)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from auth_server.providers.entra import EntraIdProvider


class TestHasGroupOverage:
    """has_group_overage classifies the two indicator formats Entra uses."""

    def test_hasgroups_true_is_overage(self):
        assert EntraIdProvider.has_group_overage({"hasgroups": True}) is True

    def test_hasgroups_false_is_not_overage(self):
        assert EntraIdProvider.has_group_overage({"hasgroups": False}) is False

    def test_claim_names_groups_is_overage(self):
        claims = {"_claim_names": {"groups": "https://graph.microsoft.com/..."}}
        assert EntraIdProvider.has_group_overage(claims) is True

    def test_claim_names_without_groups_is_not_overage(self):
        claims = {"_claim_names": {"src1": "https://example.com"}}
        assert EntraIdProvider.has_group_overage(claims) is False

    def test_no_indicators_is_not_overage(self):
        assert EntraIdProvider.has_group_overage({"groups": ["g1"]}) is False

    def test_empty_claims_is_not_overage(self):
        assert EntraIdProvider.has_group_overage({}) is False


def _mock_response(payload: dict, status_code: int = 200):
    """Build a httpx.Response-shaped mock that supports raise_for_status + json."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code}", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


class TestFetchGroupsViaGraph:
    """fetch_groups_via_graph paginates, filters, dedupes, and degrades safely."""

    @pytest.mark.asyncio
    async def test_returns_only_group_object_ids(self):
        page = {
            "value": [
                {"@odata.type": "#microsoft.graph.group", "id": "g-1"},
                {"@odata.type": "#microsoft.graph.directoryRole", "id": "role-1"},
                {"@odata.type": "#microsoft.graph.group", "id": "g-2"},
            ]
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response(page))
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("auth_server.providers.entra.httpx.AsyncClient", return_value=mock_client):
            ids = await EntraIdProvider.fetch_groups_via_graph("token")

        assert ids == ["g-1", "g-2"]

    @pytest.mark.asyncio
    async def test_pagination_combines_pages(self):
        page1 = {
            "value": [{"@odata.type": "#microsoft.graph.group", "id": "g-1"}],
            "@odata.nextLink": "https://graph.microsoft.com/v1.0/me/memberOf?$skiptoken=foo",
        }
        page2 = {"value": [{"@odata.type": "#microsoft.graph.group", "id": "g-2"}]}

        responses = [_mock_response(page1), _mock_response(page2)]
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=responses)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("auth_server.providers.entra.httpx.AsyncClient", return_value=mock_client):
            ids = await EntraIdProvider.fetch_groups_via_graph("token")

        assert ids == ["g-1", "g-2"]
        assert mock_client.get.await_count == 2

    @pytest.mark.asyncio
    async def test_dedupes_repeated_ids_across_pages(self):
        page1 = {
            "value": [{"@odata.type": "#microsoft.graph.group", "id": "g-1"}],
            "@odata.nextLink": "https://graph.microsoft.com/v1.0/me/memberOf?$skiptoken=foo",
        }
        page2 = {
            "value": [
                {"@odata.type": "#microsoft.graph.group", "id": "g-1"},
                {"@odata.type": "#microsoft.graph.group", "id": "g-2"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[_mock_response(page1), _mock_response(page2)])
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("auth_server.providers.entra.httpx.AsyncClient", return_value=mock_client):
            ids = await EntraIdProvider.fetch_groups_via_graph("token")

        assert ids == ["g-1", "g-2"]

    @pytest.mark.asyncio
    async def test_403_returns_empty_list(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response({"error": "forbidden"}, 403))
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("auth_server.providers.entra.httpx.AsyncClient", return_value=mock_client):
            ids = await EntraIdProvider.fetch_groups_via_graph("token")

        assert ids == []

    @pytest.mark.asyncio
    async def test_network_error_returns_empty_list(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("dns failed"))
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("auth_server.providers.entra.httpx.AsyncClient", return_value=mock_client):
            ids = await EntraIdProvider.fetch_groups_via_graph("token")

        assert ids == []

    @pytest.mark.asyncio
    async def test_hard_cap_truncates(self, monkeypatch):
        monkeypatch.setattr(EntraIdProvider, "GROUP_FETCH_HARD_CAP", 5)
        page = {
            "value": [{"@odata.type": "#microsoft.graph.group", "id": f"g-{i}"} for i in range(10)]
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response(page))
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("auth_server.providers.entra.httpx.AsyncClient", return_value=mock_client):
            ids = await EntraIdProvider.fetch_groups_via_graph("token")

        assert len(ids) == 5
        assert ids == [f"g-{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_url_inferred_from_login_base_url_us_gov(self, monkeypatch):
        """US Gov ENTRA_LOGIN_BASE_URL → graph.microsoft.us, no extra config."""
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.microsoftonline.us")
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response({"value": []}))
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("auth_server.providers.entra.httpx.AsyncClient", return_value=mock_client):
            await EntraIdProvider.fetch_groups_via_graph("token")

        url = mock_client.get.call_args[0][0]
        assert url.startswith("https://graph.microsoft.us/")

    @pytest.mark.asyncio
    async def test_explicit_graph_base_url_overrides_inference(self, monkeypatch):
        """Explicit ENTRA_GRAPH_BASE_URL beats the login-URL inference."""
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.microsoftonline.com")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response({"value": []}))
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("auth_server.providers.entra.httpx.AsyncClient", return_value=mock_client):
            await EntraIdProvider.fetch_groups_via_graph("token")

        url = mock_client.get.call_args[0][0]
        assert url.startswith("https://graph.proxy.example.com/")

    @pytest.mark.asyncio
    async def test_authorization_header_uses_bearer_token(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response({"value": []}))
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("auth_server.providers.entra.httpx.AsyncClient", return_value=mock_client):
            await EntraIdProvider.fetch_groups_via_graph("the-access-token")

        mock_client.get.assert_awaited_once()
        _args, kwargs = mock_client.get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer the-access-token"


class TestEntraAuthorizationServerMetadata:
    """Tests for RFC 8414 metadata exposure via authorization_server_metadata()."""

    def test_emits_v2_metadata(self, monkeypatch):
        """Phase 1 emits Entra v2 metadata only; v1 verbatim handling waits on #990."""
        monkeypatch.delenv("ENTRA_LOGIN_BASE_URL", raising=False)
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)

        from auth_server.providers.entra import EntraIdProvider

        provider = EntraIdProvider(
            tenant_id="tenant-abc",
            client_id="c",
            client_secret="s",
        )

        metadata = provider.authorization_server_metadata()

        assert metadata["issuer"] == "https://login.microsoftonline.com/tenant-abc/v2.0"
        assert (
            metadata["authorization_endpoint"]
            == "https://login.microsoftonline.com/tenant-abc/oauth2/v2.0/authorize"
        )
        assert (
            metadata["token_endpoint"]
            == "https://login.microsoftonline.com/tenant-abc/oauth2/v2.0/token"
        )
        assert "S256" in metadata["code_challenge_methods_supported"]

    def test_authorization_server_issuer_returns_v2(self, monkeypatch):
        monkeypatch.delenv("ENTRA_LOGIN_BASE_URL", raising=False)
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)

        from auth_server.providers.entra import EntraIdProvider

        provider = EntraIdProvider(
            tenant_id="tenant-abc",
            client_id="c",
            client_secret="s",
        )

        assert (
            provider.authorization_server_issuer()
            == "https://login.microsoftonline.com/tenant-abc/v2.0"
        )


def _entra(monkeypatch, **kwargs):
    """Build an EntraIdProvider with a clean env (no stray ENTRA_* overrides)."""
    for var in (
        "ENTRA_LOGIN_BASE_URL",
        "ENTRA_GRAPH_BASE_URL",
        "ENTRA_SCOPE_FORMAT",
        "ENTRA_APPLICATION_ID_URI",
    ):
        monkeypatch.delenv(var, raising=False)
    params = {"tenant_id": "tenant-abc", "client_id": "app-guid", "client_secret": "s"}
    params.update(kwargs)
    return EntraIdProvider(**params)


class TestFormatAdvertisedScopes:
    """format_advertised_scopes rewrites custom scopes for the configured form.

    Regression guard for AADSTS650053: under Entra v1 the client MUST request a
    custom resource scope as ``api://<app-id>/<scope>``; the bare fragment is
    rejected. The PRM ``scopes_supported`` array is what the client copies onto
    ``/authorize``, so it must carry the exact strings Entra expects.
    """

    def test_v2_default_emits_bare_scopes(self, monkeypatch):
        provider = _entra(monkeypatch)  # default scope_format=v2
        assert provider.scope_format == "v2"
        assert provider.format_advertised_scopes(["mcp.read", "mcp.write"]) == [
            "mcp.read",
            "mcp.write",
        ]

    def test_v1_prefixes_custom_scopes_with_app_id(self, monkeypatch):
        provider = _entra(monkeypatch, scope_format="v1")
        assert provider.format_advertised_scopes(["mcp.read"]) == ["api://app-guid/mcp.read"]

    def test_v1_uses_application_id_uri_when_set(self, monkeypatch):
        provider = _entra(monkeypatch, scope_format="v1", application_id_uri="api://custom-uri")
        assert provider.format_advertised_scopes(["mcp.read", "mcp.write"]) == [
            "api://custom-uri/mcp.read",
            "api://custom-uri/mcp.write",
        ]

    def test_v1_application_id_uri_trailing_slash_stripped(self, monkeypatch):
        provider = _entra(monkeypatch, scope_format="v1", application_id_uri="api://custom-uri/")
        assert provider.format_advertised_scopes(["mcp.read"]) == ["api://custom-uri/mcp.read"]

    def test_v1_leaves_standard_oidc_scopes_bare(self, monkeypatch):
        """AADSTS650053: Entra rejects api://<app>/openid even under v1."""
        provider = _entra(monkeypatch, scope_format="v1")
        result = provider.format_advertised_scopes(
            ["openid", "email", "profile", "offline_access", "mcp.read"]
        )
        assert result == [
            "openid",
            "email",
            "profile",
            "offline_access",
            "api://app-guid/mcp.read",
        ]

    def test_v1_leaves_already_uri_qualified_scopes_untouched(self, monkeypatch):
        """The per-server OBO PRM passes fully-resolved https:// resource scopes."""
        provider = _entra(monkeypatch, scope_format="v1")
        scopes = ["https://gw.example.com/github/mcp/user_impersonation"]
        assert provider.format_advertised_scopes(scopes) == scopes

    def test_v1_double_underscore_case_insensitive(self, monkeypatch):
        """scope_format is normalized (e.g. 'V1' -> 'v1')."""
        provider = _entra(monkeypatch, scope_format="V1")
        assert provider.scope_format == "v1"
        assert provider.format_advertised_scopes(["mcp.read"]) == ["api://app-guid/mcp.read"]

    def test_unknown_format_treated_as_v2(self, monkeypatch):
        provider = _entra(monkeypatch, scope_format="v3")
        assert provider.format_advertised_scopes(["mcp.read"]) == ["mcp.read"]

    def test_env_var_supplies_scope_format(self, monkeypatch):
        monkeypatch.setenv("ENTRA_SCOPE_FORMAT", "v1")
        for var in ("ENTRA_LOGIN_BASE_URL", "ENTRA_GRAPH_BASE_URL", "ENTRA_APPLICATION_ID_URI"):
            monkeypatch.delenv(var, raising=False)
        provider = EntraIdProvider(tenant_id="t", client_id="app-guid", client_secret="s")
        assert provider.scope_format == "v1"
        assert provider.format_advertised_scopes(["mcp.read"]) == ["api://app-guid/mcp.read"]

    def test_empty_scopes_dropped(self, monkeypatch):
        provider = _entra(monkeypatch, scope_format="v1")
        assert provider.format_advertised_scopes(["", "mcp.read"]) == ["api://app-guid/mcp.read"]


class TestProtectedResourceMetadata:
    """protected_resource_metadata applies scope formatting for the Entra form."""

    def test_v1_prm_emits_api_prefixed_custom_scopes(self, monkeypatch):
        provider = _entra(monkeypatch, scope_format="v1")
        doc = provider.protected_resource_metadata(
            resource="https://gw.example.com",
            scopes_supported=["openid", "mcp.read"],
        )
        assert doc["resource"] == "https://gw.example.com"
        assert doc["scopes_supported"] == ["openid", "api://app-guid/mcp.read"]
        assert doc["bearer_methods_supported"] == ["header"]

    def test_v2_prm_emits_bare_scopes(self, monkeypatch):
        provider = _entra(monkeypatch)
        doc = provider.protected_resource_metadata(
            resource="https://gw.example.com",
            scopes_supported=["openid", "mcp.read"],
        )
        assert doc["scopes_supported"] == ["openid", "mcp.read"]


class TestAADSTS650053Regression:
    """Regression for the AADSTS650053 failure mode.

    Entra v1 rejects an /authorize request whose scope is the bare custom-scope
    fragment ('The application asked for scope X that doesn't exist on the
    resource Y'). The scope the client sends is copied verbatim from the PRM
    ``scopes_supported`` array. This test mocks the Entra /authorize contract:
    given a registered v1 resource scope, the PRM MUST advertise the exact
    ``api://<app-id>/<scope>`` string the resource exposes, so the client's
    request matches and Entra does not raise AADSTS650053.
    """

    def _entra_authorize_accepts(self, exposed_scopes, requested_scope):
        """Model Entra v1 /authorize: a custom scope must exactly match a scope
        exposed on the resource, else AADSTS650053. OIDC scopes always pass."""
        oidc = {"openid", "profile", "email", "offline_access"}
        if requested_scope in oidc:
            return True
        return requested_scope in exposed_scopes

    def test_v1_prm_scope_matches_resource_exposed_scope(self, monkeypatch):
        # The Entra app registration exposes this scope on its resource.
        exposed = {"api://app-guid/mcp.read"}
        provider = _entra(monkeypatch, scope_format="v1")

        advertised = provider.format_advertised_scopes(["mcp.read"])

        # Every advertised custom scope must be accepted by Entra /authorize.
        for scope in advertised:
            assert self._entra_authorize_accepts(exposed, scope), (
                f"AADSTS650053: Entra would reject advertised scope '{scope}'"
            )
        assert advertised == ["api://app-guid/mcp.read"]

    def test_v1_bare_scope_would_be_rejected_by_entra(self, monkeypatch):
        """Documents the failure this fix prevents: the pre-fix bare form (what
        v2 emits) does NOT match the v1 resource scope -> AADSTS650053."""
        exposed = {"api://app-guid/mcp.read"}
        # Bare form (v2 output) is not exposed on the v1 resource.
        assert self._entra_authorize_accepts(exposed, "mcp.read") is False
        # The v1-formatted form IS accepted.
        assert self._entra_authorize_accepts(exposed, "api://app-guid/mcp.read") is True


class TestAcceptedAudiences:
    """accepted_audiences centralizes dual-audience normalization (v1 aud forms)."""

    def test_accepts_bare_guid_and_api_uri(self, monkeypatch):
        """Entra v1 aud may be the bare GUID or api://<app-id>; both accepted."""
        provider = _entra(monkeypatch)
        auds = provider.accepted_audiences()
        assert "app-guid" in auds
        assert "api://app-guid" in auds

    def test_includes_configured_application_id_uri(self, monkeypatch):
        provider = _entra(monkeypatch, application_id_uri="api://custom-uri")
        auds = provider.accepted_audiences()
        assert "api://custom-uri" in auds

    def test_application_id_uri_trailing_slash_stripped(self, monkeypatch):
        provider = _entra(monkeypatch, application_id_uri="api://custom-uri/")
        assert "api://custom-uri" in provider.accepted_audiences()
        assert "api://custom-uri/" not in provider.accepted_audiences()

    def test_extra_audiences_appended_and_stripped(self, monkeypatch):
        provider = _entra(monkeypatch)
        auds = provider.accepted_audiences(
            extra_audiences=["https://gw.example.com/github/mcp/", None, ""]
        )
        assert "https://gw.example.com/github/mcp" in auds
        # None/empty entries are ignored, not appended.
        assert None not in auds
        assert "" not in auds

    def test_deduplicated_preserving_order(self, monkeypatch):
        # application_id_uri equals the default api:// form -> only one entry.
        provider = _entra(monkeypatch, application_id_uri="api://app-guid")
        auds = provider.accepted_audiences()
        assert auds.count("api://app-guid") == 1
        assert auds == ["app-guid", "api://app-guid"]


class TestRejectNonAccessToken:
    """_reject_non_access_token blocks id_token->access_token confusion.

    An Entra id_token shares the tenant JWKS + issuer and has aud == client_id
    (which accepted_audiences() accepts), so a client could replay the id_token
    it gets from the auth-code exchange as the bearer. The discriminator rejects
    on id_token-only claims (nonce/at_hash/c_hash), WITHOUT requiring scp/roles
    (which a valid roleless M2M access token legitimately lacks).
    """

    def _reject(self, claims):
        from auth_server.providers.entra import EntraIdProvider

        EntraIdProvider._reject_non_access_token(claims)

    def test_rejects_id_token_with_nonce(self):
        with pytest.raises(ValueError, match="id_token"):
            self._reject({"aud": "app-guid", "nonce": "abc", "groups": ["g"]})

    def test_rejects_id_token_with_at_hash(self):
        with pytest.raises(ValueError, match="id_token"):
            self._reject({"aud": "app-guid", "at_hash": "xyz"})

    def test_rejects_id_token_with_c_hash(self):
        with pytest.raises(ValueError, match="id_token"):
            self._reject({"aud": "app-guid", "c_hash": "xyz"})

    def test_accepts_delegated_access_token_with_scp(self):
        # scp present, no id_token-only claim -> valid delegated access token.
        self._reject({"aud": "app-guid", "scp": "user_impersonation"})

    def test_accepts_app_access_token_with_roles(self):
        self._reject({"aud": "app-guid", "roles": ["Registry.Admin"]})

    def test_accepts_roleless_m2m_access_token(self):
        # A client-credentials token for an app with no assigned app roles has
        # neither scp nor roles -- and MUST still be accepted (no false-reject).
        self._reject({"aud": "api://app-guid", "azp": "svc-client"})
