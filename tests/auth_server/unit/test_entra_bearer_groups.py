"""Entra delegated bearer group recovery on /validate and its Graph OBO boundary."""

import asyncio
import time
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from auth_server.providers.entra import EntraIdProvider, GraphGroupLookupError

pytestmark = [pytest.mark.unit, pytest.mark.auth]


@pytest.fixture(autouse=True)
def _isolated_memberships(monkeypatch):
    from providers.entra import EntraIdProvider as RuntimeEntraIdProvider

    for provider_type in (EntraIdProvider, RuntimeEntraIdProvider):
        monkeypatch.setattr(provider_type, "_group_cache", OrderedDict())
        monkeypatch.setattr(provider_type, "_group_inflight", {})
        monkeypatch.setattr(provider_type, "_group_denials", OrderedDict())
        monkeypatch.setattr(provider_type, "_group_verified", OrderedDict())


@pytest.fixture
def entra_provider(monkeypatch):
    monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
    monkeypatch.delenv("ENTRA_LOGIN_BASE_URL", raising=False)
    return EntraIdProvider(tenant_id="tenant-one", client_id="gateway-app", client_secret="secret")


@pytest.fixture
def runtime_entra_provider(monkeypatch):
    from providers.entra import EntraIdProvider as RuntimeEntraIdProvider

    monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
    monkeypatch.delenv("ENTRA_LOGIN_BASE_URL", raising=False)
    return RuntimeEntraIdProvider(
        tenant_id="tenant-one", client_id="gateway-app", client_secret="secret"
    )


def _verified_result(claims, *, method="entra"):
    return {
        "valid": True,
        "method": method,
        "username": "alice@example.com",
        "client_id": "gateway-app",
        "groups": claims.get("groups") or claims.get("roles") or [],
        "scopes": [],
        "data": dict(claims),
    }


def _delegated(**claims):
    return {"sub": "alice", "tid": "tenant-one", "oid": "object-alice", "scp": "mcp.read", **claims}


def _validate(client, token="incoming-entra-token"):
    return client.get(
        "/validate",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Original-URL": "https://example.com/test-server/mcp",
        },
    )


@pytest.fixture
def bearer_client(
    runtime_entra_provider, auth_env_vars, mock_scope_repository_with_data, mock_scopes_config
):
    import group_filter

    group_filter._clear_mapped_group_names_cache()

    from auth_server import server

    repo = mock_scope_repository_with_data
    # /test-server/mcp resolves to method "mcp" on the nginx subrequest;
    # the shared read:servers fixture grants initialize/tools/list only.
    mock_scopes_config["read:servers"][0]["methods"].append("mcp")
    repo.get_all_mapped_group_names.return_value = {"nested-parent", "inline-allowed", "m2m-role"}
    repo.get_group_mappings_bulk.side_effect = lambda groups: [
        scope
        for group, scope in (
            ("nested-parent", "read:servers"),
            ("inline-allowed", "read:servers"),
            ("m2m-role", "read:servers"),
        )
        if group in groups
    ]
    with (
        patch.object(server, "get_auth_provider", return_value=runtime_entra_provider),
        patch.object(server, "get_scope_repository", return_value=repo),
        patch("registry.repositories.factory.get_scope_repository", return_value=repo),
        patch.object(group_filter, "ALLOWED_IDP_GROUPS", []),
        patch.object(server, "_get_blocked_tools", new=AsyncMock(return_value=set())),
        patch("mongodb_groups_enrichment.should_enrich_groups", return_value=False),
        patch("mongodb_groups_enrichment.should_enrich_user_groups", return_value=False),
    ):
        yield TestClient(server.app), runtime_entra_provider, repo
        group_filter._clear_mapped_group_names_cache()


def test_signed_entra_scp_authorizes_direct_request_without_graph(bearer_client, monkeypatch):
    import json

    from cryptography.hazmat.primitives.asymmetric import rsa
    from jwt.algorithms import RSAAlgorithm

    client, provider, _repo = bearer_client
    monkeypatch.setenv("ENTRA_GROUP_RECOVERY_ENABLED", "false")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk["kid"] = "test-key"
    now = int(time.time())
    token = jwt.encode(
        {
            **_delegated(),
            "iss": provider.issuer_v2,
            "aud": provider.client_id,
            "scp": "read:servers",
            "iat": now,
            "exp": now + 3600,
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )
    with (
        patch.object(provider, "get_jwks", return_value={"keys": [jwk]}),
        patch("providers.entra.obo_exchange", new_callable=AsyncMock) as exchange,
    ):
        response = _validate(client, token)

    assert response.status_code == 200, response.text
    assert response.json()["scopes"] == ["read:servers"]
    assert response.json()["groups"] == []
    exchange.assert_not_awaited()


def test_signed_scope_only_user_never_inherits_m2m_client_groups(bearer_client, monkeypatch):
    import json

    from cryptography.hazmat.primitives.asymmetric import rsa
    from jwt.algorithms import RSAAlgorithm

    from auth_server.mongodb_groups_enrichment import should_enrich_groups

    client, provider, _repo = bearer_client
    monkeypatch.setenv("ENTRA_GROUP_RECOVERY_ENABLED", "false")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk["kid"] = "test-key"
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "alice",
            "tid": "tenant-one",
            "oid": "object-alice",
            "iss": provider.issuer_v2,
            "aud": provider.client_id,
            "scope": "read:servers",
            "iat": now,
            "exp": now + 3600,
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )
    with (
        patch.object(provider, "get_jwks", return_value={"keys": [jwk]}),
        patch("providers.entra.obo_exchange", new_callable=AsyncMock) as exchange,
        patch(
            "mongodb_groups_enrichment.should_enrich_groups",
            wraps=should_enrich_groups,
        ),
        patch(
            "mongodb_groups_enrichment.enrich_groups_from_mongodb",
            new_callable=AsyncMock,
            return_value=["m2m-role"],
        ) as m2m,
    ):
        response = _validate(client, token)

    assert response.status_code == 200, response.text
    assert response.json()["groups"] == []
    assert response.json()["scopes"] == ["read:servers"]
    m2m.assert_not_awaited()
    exchange.assert_not_awaited()


@pytest.mark.parametrize("recovered", [False, True], ids=["inline", "graph"])
def test_signed_scp_does_not_add_scopes_when_groups_map(bearer_client, recovered):
    """Registry scopes come only from group mappings when the user has groups."""
    import json

    from cryptography.hazmat.primitives.asymmetric import rsa
    from jwt.algorithms import RSAAlgorithm

    client, provider, _repo = bearer_client
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk["kid"] = "test-key"
    now = int(time.time())
    token = jwt.encode(
        {
            **_delegated(**({"hasgroups": True} if recovered else {"groups": ["nested-parent"]})),
            "iss": provider.issuer_v2,
            "aud": provider.client_id,
            "scp": "direct:scope read:servers",
            "iat": now,
            "exp": now + 3600,
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )
    with (
        patch.object(provider, "get_jwks", return_value={"keys": [jwk]}),
        patch(
            "providers.entra.obo_exchange", new_callable=AsyncMock, return_value="graph-token"
        ) as exchange,
        patch.object(
            type(provider),
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent"],
        ) as graph,
    ):
        response = _validate(client, token)

    assert response.status_code == 200, response.text
    # nested-parent maps to read:servers; the token's scp ("direct:scope
    # read:servers") must not add direct:scope on top of that mapping.
    assert response.json()["scopes"] == ["read:servers"]
    assert response.json()["groups"] == ["nested-parent"]
    if recovered:
        exchange.assert_awaited_once()
        graph.assert_awaited_once()
    else:
        exchange.assert_not_awaited()
        graph.assert_not_awaited()


@pytest.mark.parametrize(
    "claims",
    [
        _delegated(),  # No overage marker: missing groups alone is enough for a user token.
        _delegated(hasgroups=True),
        _delegated(groups=["stale-inline"], _claim_names={"groups": "src1"}),
    ],
    ids=["missing-without-marker", "hasgroups", "claim-names-overage"],
)
def test_validate_recovers_verified_delegated_groups_before_scope_mapping(bearer_client, claims):
    client, provider, repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(claims))
    with (
        patch(
            "providers.entra.obo_exchange", new_callable=AsyncMock, return_value="graph-token"
        ) as obo,
        patch.object(
            type(provider),
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent", "unmapped-group"],
        ) as graph,
    ):
        response = _validate(client)

    assert response.status_code == 200, response.text
    assert response.json()["scopes"] == ["read:servers"]
    assert response.json()["groups"] == ["nested-parent"]
    assert response.headers["X-Scopes"] == "read:servers"
    obo.assert_awaited_once_with(
        provider,
        subject_token="incoming-entra-token",
        target_audience=provider.graph_base_url,
        scopes=[f"{provider.graph_base_url}/User.Read"],
    )
    graph.assert_awaited_once_with("graph-token", tenant_id="tenant-one", object_id="object-alice")
    assert repo.get_group_mappings_bulk.await_args.args[0] == ["nested-parent"]


def test_validate_maps_transitive_graph_membership_to_scope(bearer_client):
    client, provider, _repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(_delegated()))
    graph_client = AsyncMock()
    graph_client.get.return_value = httpx.Response(
        200,
        json={
            "value": [
                {"@odata.type": "#microsoft.graph.directoryRole", "id": "not-a-group"},
                {"@odata.type": "#microsoft.graph.group", "id": "nested-parent"},
            ]
        },
        request=httpx.Request("GET", "https://graph.microsoft.com/v1.0/me/transitiveMemberOf"),
    )
    graph_client.__aenter__.return_value = graph_client
    with (
        patch("providers.entra.obo_exchange", new_callable=AsyncMock, return_value="graph-token"),
        patch("providers.entra.guarded_async_client", return_value=graph_client),
    ):
        response = _validate(client)

    assert response.status_code == 200, response.text
    assert response.json()["groups"] == ["nested-parent"]
    assert response.json()["scopes"] == ["read:servers"]
    graph_client.get.assert_awaited_once()
    assert (
        graph_client.get.await_args.args[0]
        == "https://graph.microsoft.com/v1.0/me/transitiveMemberOf"
    )
    assert graph_client.get.await_args.kwargs["headers"]["Authorization"] == "Bearer graph-token"


def test_validate_filters_hundreds_of_graph_groups_before_mapping_or_token(bearer_client):
    client, provider, repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(_delegated(hasgroups=True)))
    recovered = [f"unmapped-{i}" for i in range(350)] + ["nested-parent"]
    with (
        patch("providers.entra.obo_exchange", new_callable=AsyncMock, return_value="graph-token"),
        patch.object(
            type(provider), "fetch_groups_via_graph", new_callable=AsyncMock, return_value=recovered
        ),
    ):
        response = _validate(client)

    assert response.status_code == 200, response.text
    assert response.json()["groups"] == ["nested-parent"]
    assert response.json()["scopes"] == ["read:servers"]
    assert repo.get_group_mappings_bulk.await_args.args[0] == ["nested-parent"]


def test_validate_denies_oversized_recovered_group_set(bearer_client):
    from providers.entra import GRAPH_GROUP_MAX_BYTES

    client, provider, repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(_delegated(hasgroups=True)))
    oversized = ["nested-parent", "x" * (GRAPH_GROUP_MAX_BYTES + 1)]
    with (
        patch("providers.entra.obo_exchange", new_callable=AsyncMock, return_value="graph-token"),
        patch.object(
            type(provider), "fetch_groups_via_graph", new_callable=AsyncMock, return_value=oversized
        ),
        patch("group_filter.filter_session_groups", new_callable=AsyncMock, return_value=oversized),
    ):
        response = _validate(client)

    assert response.status_code == 503
    assert "X-Scopes" not in response.headers
    repo.get_group_mappings_bulk.assert_not_awaited()


def test_empty_graph_membership_never_uses_m2m_enrichment(bearer_client):
    client, provider, repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(_delegated(hasgroups=True)))
    with (
        patch("providers.entra.obo_exchange", new_callable=AsyncMock, return_value="graph-token"),
        patch.object(
            type(provider), "fetch_groups_via_graph", new_callable=AsyncMock, return_value=[]
        ) as graph,
        patch("mongodb_groups_enrichment.should_enrich_groups", return_value=True),
        patch(
            "mongodb_groups_enrichment.enrich_groups_from_mongodb",
            new_callable=AsyncMock,
            return_value=["m2m-role"],
        ) as m2m,
    ):
        response = _validate(client)

    assert response.status_code == 403, response.text
    graph.assert_awaited_once()
    m2m.assert_not_awaited()
    repo.get_group_mappings_bulk.assert_not_awaited()


@pytest.mark.parametrize("graph_result", [[], ["nested-parent"]])
def test_graph_memberships_override_configured_stale_user_fallback(bearer_client, graph_result):
    from auth_server import server
    from auth_server.mongodb_groups_enrichment import should_enrich_user_groups

    client, provider, repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(_delegated(hasgroups=True)))
    with (
        patch.object(server, "IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS", ["entra"]),
        patch(
            "mongodb_groups_enrichment.should_enrich_user_groups", wraps=should_enrich_user_groups
        ),
        patch(
            "mongodb_groups_enrichment.enrich_user_groups_from_mongodb",
            new_callable=AsyncMock,
            return_value=["m2m-role"],
        ) as local_fallback,
        patch("providers.entra.obo_exchange", new_callable=AsyncMock, return_value="graph-token"),
        patch.object(
            type(provider),
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=graph_result,
        ) as graph,
    ):
        response = _validate(client)

    graph.assert_awaited_once()
    local_fallback.assert_not_awaited()
    if graph_result:
        assert response.status_code == 200, response.text
        assert response.json()["groups"] == graph_result
        assert response.json()["scopes"] == ["read:servers"]
    else:
        assert response.status_code == 403, response.text
        assert "X-Scopes" not in response.headers
        repo.get_group_mappings_bulk.assert_not_awaited()


def test_disabled_graph_recovery_preserves_configured_user_fallback(bearer_client, monkeypatch):
    from auth_server import server
    from auth_server.mongodb_groups_enrichment import should_enrich_user_groups

    monkeypatch.setenv("ENTRA_GROUP_RECOVERY_ENABLED", "false")
    client, provider, _repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(_delegated(hasgroups=True)))
    with (
        patch.object(server, "IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS", ["entra"]),
        patch(
            "mongodb_groups_enrichment.should_enrich_user_groups", wraps=should_enrich_user_groups
        ),
        patch(
            "mongodb_groups_enrichment.enrich_user_groups_from_mongodb",
            new_callable=AsyncMock,
            return_value=["m2m-role"],
        ) as local_fallback,
        patch("providers.entra.obo_exchange", new_callable=AsyncMock) as obo,
    ):
        response = _validate(client)

    assert response.status_code == 200, response.text
    assert response.json()["groups"] == ["m2m-role"]
    assert response.json()["scopes"] == ["read:servers"]
    local_fallback.assert_awaited_once_with("alice@example.com", [], "entra")
    obo.assert_not_awaited()


@pytest.mark.parametrize("missing", [False, True])
def test_validate_denies_recovered_groups_when_scope_mappings_unavailable(bearer_client, missing):
    client, provider, repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(_delegated()))
    if missing:
        repo.get_all_mapped_group_names.return_value = set()
    else:
        repo.get_all_mapped_group_names.side_effect = RuntimeError("database unavailable")
    with (
        patch("providers.entra.obo_exchange", new_callable=AsyncMock, return_value="graph-token"),
        patch.object(
            type(provider),
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent"],
        ),
    ):
        response = _validate(client)
    assert response.status_code == 503
    assert response.headers["X-Entra-Group-Lookup-Unavailable"] == "1"
    assert "X-Scopes" not in response.headers
    repo.get_group_mappings_bulk.assert_not_awaited()


@pytest.mark.parametrize(
    "claims,expected",
    [
        (_delegated(hasgroups=True, groups=["stale-admin"], roles=["m2m-role"]), ["m2m-role"]),
        (_delegated(), []),
    ],
)
def test_disabled_recovery_never_uses_stale_group_claims(
    bearer_client, monkeypatch, claims, expected
):
    monkeypatch.setenv("ENTRA_GROUP_RECOVERY_ENABLED", "false")
    client, provider, _repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(claims))
    with (
        patch("providers.entra.obo_exchange", new_callable=AsyncMock) as obo,
        patch.object(type(provider), "fetch_groups_via_graph", new_callable=AsyncMock) as graph,
    ):
        response = _validate(client)
    assert response.status_code == (200 if expected else 403)
    if expected:
        assert response.json()["groups"] == expected
    else:
        assert "X-Scopes" not in response.headers
    obo.assert_not_awaited()
    graph.assert_not_awaited()


@pytest.mark.parametrize(
    "claims,method",
    [
        (_delegated(groups=["inline-allowed"]), "entra"),
        (_delegated(roles=["m2m-role"], scp=""), "entra"),
        (_delegated(hasgroups=True, roles=["m2m-role"], scp=""), "entra"),
    ],
    ids=["inline-user-groups", "m2m-roles", "m2m-roles-with-overage-marker"],
)
def test_validate_does_not_look_up_inline_or_machine_roles(bearer_client, claims, method):
    client, provider, _repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(claims, method=method))
    with (
        patch("providers.entra.obo_exchange", new_callable=AsyncMock) as obo,
        patch.object(type(provider), "fetch_groups_via_graph", new_callable=AsyncMock) as graph,
    ):
        response = _validate(client)

    assert response.status_code == 200, response.text
    assert response.json()["scopes"] == ["read:servers"]
    assert response.json()["groups"] == claims.get("groups", claims.get("roles"))
    obo.assert_not_awaited()
    graph.assert_not_awaited()


def test_validate_carries_large_inline_group_set_once_in_registry_token(bearer_client):
    """Issue #1939: a 200-group inline token must reach the registry unfiltered.

    Group-restricted visibility matches allowed_groups against these groups, so
    they are not trimmed to scope-mapped ones. Among the response headers, which
    nginx reads into a fixed buffer, they travel once, in the signed registry
    token, not again in an X-Groups header that doubled the auth_request
    response past that buffer. (The JSON body, which nginx discards, also lists
    them.)
    """
    from registry.core.config import settings

    client, provider, _repo = bearer_client
    inline = ["inline-allowed"] + [f"00000000-0000-4000-8000-{i:012d}" for i in range(199)]
    provider.validate_token = MagicMock(return_value=_verified_result(_delegated(groups=inline)))

    response = client.get(
        "/validate",
        headers={
            "Authorization": "Bearer incoming-entra-token",
            "X-Original-URL": "https://example.com/test-server/mcp",
            "X-Registry-Api-Auth": "1",
            "X-Validate-Source-Secret": settings.auth_server_nginx_marker_secret,
        },
    )

    assert response.status_code == 200, response.text
    assert "X-Groups" not in response.headers
    token = response.headers["X-Internal-Token-Registry"]
    assert jwt.decode(token, options={"verify_signature": False})["groups"] == inline
    # Entra caps inline groups at 200. Sent once, they need under a quarter of the
    # 64k /validate buffer and would fit even nginx's 16k http-level default; a
    # second copy (the old X-Groups header) pushed this response to about 18 KB.
    header_bytes = sum(len(k) + len(v) for k, v in response.headers.items())
    assert header_bytes < 16 * 1024


def test_validate_self_signed_token_never_enters_graph_path(bearer_client):
    import time

    from auth_server import server

    client, provider, _repo = bearer_client
    # A locally signed access token can still contain copied Entra identity and
    # overage markers; its verified method, not those claims, controls OBO.
    token = jwt.encode(
        {
            **_delegated(hasgroups=True, groups=["inline-allowed"]),
            "iss": server.JWT_ISSUER,
            "aud": "mcp-registry",
            "token_use": "access",
            "token_kind": "user",
            "scope": "read:servers",
            "exp": int(time.time()) + 60,
        },
        server.SECRET_KEY,
        algorithm="HS256",
    )
    provider.validate_token = MagicMock()
    with (
        patch("providers.entra.obo_exchange", new_callable=AsyncMock) as obo,
        patch.object(type(provider), "fetch_groups_via_graph", new_callable=AsyncMock) as graph,
    ):
        response = _validate(client, token)

    assert response.status_code == 200, response.text
    assert response.headers["X-Auth-Method"] == "self_signed"
    assert response.json()["groups"] == ["inline-allowed"]
    obo.assert_not_awaited()
    graph.assert_not_awaited()
    provider.validate_token.assert_not_called()


def test_consent_failure_retries_only_after_short_cache_or_new_token(bearer_client, monkeypatch):
    from egress_obo import OboConsentRequired

    client, provider, _repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(_delegated(hasgroups=True)))
    clock = [1000.0]
    exchange = AsyncMock(side_effect=OboConsentRequired("consent missing"))
    with (
        patch("providers.entra.time.monotonic", side_effect=lambda: clock[0]),
        patch("providers.entra.obo_exchange", exchange),
    ):
        first = _validate(client, "token-one")
        second = _validate(client, "token-one")
        new_token = _validate(client, "token-two")
        clock[0] += 30
        after_ttl = _validate(client, "token-one")
    assert [response.status_code for response in (first, second, new_token, after_ttl)] == [
        503,
        503,
        503,
        503,
    ]
    assert exchange.await_count == 3
    assert all(
        "X-Scopes" not in response.headers for response in (first, second, new_token, after_ttl)
    )


@pytest.mark.parametrize("failure", ["obo", "graph"])
@pytest.mark.parametrize(
    "claims",
    [_delegated(hasgroups=True), _delegated(hasgroups=True, groups=["stale-inline"])],
    ids=["missing", "overage-with-stale-groups"],
)
def test_validate_denies_failed_recovery_and_retries_next_request(bearer_client, failure, claims):
    from egress_obo import OboExchangeError

    client, provider, _repo = bearer_client
    provider.validate_token = MagicMock(side_effect=lambda *_a, **_kw: _verified_result(claims))
    obo = AsyncMock(return_value="graph-token")
    graph = AsyncMock(return_value=["nested-parent"])
    if failure == "obo":
        obo.side_effect = [OboExchangeError("IdP unavailable"), "graph-token"]
    else:
        from providers.entra import GraphGroupLookupError as RuntimeGraphGroupLookupError

        graph.side_effect = [RuntimeGraphGroupLookupError("Graph unavailable"), ["nested-parent"]]
    with (
        patch("providers.entra.obo_exchange", obo),
        patch.object(type(provider), "fetch_groups_via_graph", graph),
    ):
        denied = _validate(client)
        accepted = _validate(client)

    assert denied.status_code == 503, denied.text
    assert denied.headers["X-Entra-Group-Lookup-Unavailable"] == "1"
    assert "X-Scopes" not in denied.headers
    assert accepted.status_code == 200, accepted.text
    assert "X-Entra-Group-Lookup-Unavailable" not in accepted.headers
    assert accepted.json()["scopes"] == ["read:servers"]
    assert accepted.json()["groups"] == ["nested-parent"]
    assert obo.await_count == 2
    assert graph.await_count == (1 if failure == "obo" else 2)


@pytest.mark.parametrize("malformed", [ValueError("non-JSON"), AttributeError("not-object")])
def test_malformed_obo_response_returns_unavailable_without_grants(bearer_client, malformed):
    client, provider, _repo = bearer_client
    provider.validate_token = MagicMock(return_value=_verified_result(_delegated(hasgroups=True)))
    with (
        patch("providers.entra.obo_exchange", new_callable=AsyncMock, side_effect=malformed),
        patch.object(type(provider), "fetch_groups_via_graph", new_callable=AsyncMock) as graph,
    ):
        response = _validate(client)

    assert response.status_code == 503
    assert "X-Scopes" not in response.headers
    graph.assert_not_awaited()


def test_invalid_bearer_group_claim_returns_401_without_graph(bearer_client):
    client, provider, _repo = bearer_client
    provider.validate_token = MagicMock(
        return_value=_verified_result(_delegated(groups={"admin": True}))
    )
    with patch("providers.entra.obo_exchange", new_callable=AsyncMock) as exchange:
        response = _validate(client)

    assert response.status_code == 401
    assert "X-Scopes" not in response.headers
    exchange.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        {**_verified_result(_delegated()), "method": "self_signed"},
        _verified_result({"tid": "tenant-one", "oid": "object-alice", "roles": ["m2m-role"]}),
        _verified_result(_delegated(groups=["inline-allowed"])),
    ],
    ids=["self-signed", "m2m-role", "inline-groups"],
)
async def test_recovery_skips_non_delegated_claims(entra_provider, result):
    with (
        patch("auth_server.providers.entra.obo_exchange", new_callable=AsyncMock) as obo,
        patch.object(EntraIdProvider, "fetch_groups_via_graph", new_callable=AsyncMock) as graph,
    ):
        assert await entra_provider.recover_bearer_groups(result, "subject-token") is None

    obo.assert_not_awaited()
    graph.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        {**_verified_result(_delegated()), "valid": False},
        {
            **_verified_result(_delegated()),
            "data": {"scp": "mcp.read"},
            "oid": "outer-oid",
            "tid": "outer-tid",
        },
        _verified_result(_delegated(tid="", hasgroups=True)),
        _verified_result(_delegated(oid="", hasgroups=True)),
    ],
    ids=["unverified-token", "outer-identity-untrusted", "missing-tenant", "missing-object"],
)
async def test_recovery_refuses_to_exchange_without_verified_identity(entra_provider, result):
    result.update(oid="outer-oid", tid="outer-tid")
    with (
        patch("auth_server.providers.entra.obo_exchange", new_callable=AsyncMock) as obo,
        patch.object(EntraIdProvider, "fetch_groups_via_graph", new_callable=AsyncMock) as graph,
    ):
        with pytest.raises(GraphGroupLookupError):
            await entra_provider.recover_bearer_groups(result, "subject-token")

    obo.assert_not_awaited()
    graph.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_same_token_misses_share_one_obo_and_graph_lookup(entra_provider):
    ready = asyncio.Event()
    release = asyncio.Event()

    async def wait_for_exchange(*_args, **_kwargs):
        ready.set()
        await release.wait()
        return "graph-token"

    with (
        patch(
            "auth_server.providers.entra.obo_exchange",
            new_callable=AsyncMock,
            side_effect=wait_for_exchange,
        ) as obo,
        patch.object(
            EntraIdProvider,
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent"],
        ) as graph,
    ):
        calls = [
            asyncio.create_task(
                entra_provider.recover_bearer_groups(
                    _verified_result(_delegated(hasgroups=True)), "same-token"
                )
            )
            for _ in range(12)
        ]
        await ready.wait()
        # Let the other calls register their wait on the same in-flight lookup.
        await asyncio.sleep(0)
        release.set()
        result = await asyncio.gather(*calls)

    assert result == [["nested-parent"] for _ in calls]
    obo.assert_awaited_once()
    graph.assert_awaited_once_with("graph-token", tenant_id="tenant-one", object_id="object-alice")


@pytest.mark.asyncio
async def test_concurrent_distinct_tokens_do_not_share_obo_failure(entra_provider):
    from egress_obo import OboConsentRequired

    started = asyncio.Event()
    release = asyncio.Event()
    result = _verified_result(_delegated(hasgroups=True))

    async def exchange(*_args, subject_token, **_kwargs):
        started.set()
        await release.wait()
        if subject_token == "old-token":
            raise OboConsentRequired("consent required")
        return "graph-token"

    with (
        patch(
            "auth_server.providers.entra.obo_exchange", new_callable=AsyncMock, side_effect=exchange
        ) as obo,
        patch.object(
            EntraIdProvider,
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent"],
        ),
    ):
        old = asyncio.create_task(entra_provider.recover_bearer_groups(result, "old-token"))
        await started.wait()
        new = asyncio.create_task(entra_provider.recover_bearer_groups(result, "new-token"))
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(GraphGroupLookupError, match="token exchange failed"):
            await old
        assert await new == ["nested-parent"]
    assert obo.await_count == 2


@pytest.mark.asyncio
async def test_stalled_graph_exchange_times_out_without_caching_failure(
    entra_provider, monkeypatch
):
    from auth_server.providers import entra as entra_module

    monkeypatch.setattr(entra_module, "GRAPH_GROUP_LOOKUP_TIMEOUT_SECONDS", 0.01)

    async def stalled_exchange(*_args, **_kwargs):
        await asyncio.Event().wait()

    # The first exchange hangs, the next attempt is allowed to try again.
    call_count = 0

    async def maybe_stall(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await stalled_exchange(*args, **kwargs)
        return "graph-token"

    result = _verified_result(_delegated(hasgroups=True))
    with (
        patch(
            "auth_server.providers.entra.obo_exchange",
            new_callable=AsyncMock,
            side_effect=maybe_stall,
        ) as obo,
        patch.object(
            EntraIdProvider,
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent"],
        ),
    ):
        with pytest.raises(GraphGroupLookupError, match="timed out"):
            await entra_provider.recover_bearer_groups(result, "token")
        assert await entra_provider.recover_bearer_groups(result, "token") == ["nested-parent"]
        assert obo.await_count == 2


@pytest.mark.asyncio
async def test_cached_graph_memberships_keep_each_token_app_roles(entra_provider):
    first = _verified_result(_delegated(hasgroups=True, roles=["first-role"]))
    second = _verified_result(_delegated(hasgroups=True, roles=["second-role"]))
    with (
        patch(
            "auth_server.providers.entra.obo_exchange",
            new_callable=AsyncMock,
            return_value="graph-token",
        ) as exchange,
        patch.object(
            EntraIdProvider,
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent"],
        ) as graph,
    ):
        initial = await entra_provider.recover_bearer_groups(first, "first-token")
        # The network stub doesn't populate the provider's normal Graph cache.
        EntraIdProvider._group_cache[
            (
                EntraIdProvider._graph_memberof_url(),
                "tenant-one",
                "object-alice",
                "https://graph.microsoft.com",
            )
        ] = (time.monotonic() + 300, ("nested-parent",))
        cached = await entra_provider.recover_bearer_groups(second, "second-token")

    assert initial == ["first-role", "nested-parent"]
    assert cached == ["second-role", "nested-parent"]
    assert exchange.await_count == 2
    assert graph.await_count == 2


@pytest.mark.asyncio
async def test_denied_token_cannot_borrow_other_tokens_cached_memberships(entra_provider):
    from egress_obo import OboConsentRequired

    result = _verified_result(_delegated(hasgroups=True))

    async def exchange(*_args, subject_token, **_kwargs):
        if subject_token == "denied-token":
            raise OboConsentRequired("consent required")
        return "graph-token"

    with (
        patch(
            "auth_server.providers.entra.obo_exchange", new_callable=AsyncMock, side_effect=exchange
        ) as obo,
        patch.object(
            EntraIdProvider,
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent"],
        ),
    ):
        with pytest.raises(GraphGroupLookupError) as denied:
            await entra_provider.recover_bearer_groups(result, "denied-token")
        assert denied.value.reason == "consent"
        await entra_provider.recover_bearer_groups(result, "allowed-token")
        with pytest.raises(GraphGroupLookupError) as still_denied:
            await entra_provider.recover_bearer_groups(result, "denied-token")

    assert still_denied.value.reason == "consent"
    assert obo.await_count == 2


@pytest.mark.asyncio
async def test_browser_cached_membership_requires_each_bearer_token_obo(entra_provider):
    key = (
        EntraIdProvider._graph_memberof_url(),
        "tenant-one",
        "object-alice",
        "https://graph.microsoft.com",
    )
    EntraIdProvider._group_cache[key] = (time.monotonic() + 300, ("nested-parent",))
    result = _verified_result(_delegated(hasgroups=True))
    with (
        patch(
            "auth_server.providers.entra.obo_exchange",
            new_callable=AsyncMock,
            return_value="graph-token",
        ) as obo,
        patch.object(
            EntraIdProvider,
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent"],
        ) as graph,
    ):
        assert await entra_provider.recover_bearer_groups(result, "first-token") == [
            "nested-parent"
        ]
        assert await entra_provider.recover_bearer_groups(result, "first-token") == [
            "nested-parent"
        ]
        assert await entra_provider.recover_bearer_groups(result, "second-token") == [
            "nested-parent"
        ]

    assert obo.await_count == 2
    assert graph.await_count == 2


@pytest.mark.asyncio
async def test_graph_proxy_does_not_change_obo_resource(monkeypatch):
    monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
    provider = EntraIdProvider(
        tenant_id="tenant-one", client_id="gateway-app", client_secret="secret"
    )
    with (
        patch(
            "auth_server.providers.entra.obo_exchange",
            new_callable=AsyncMock,
            return_value="graph-token",
        ) as exchange,
        patch.object(
            EntraIdProvider, "fetch_groups_via_graph", new_callable=AsyncMock, return_value=[]
        ),
    ):
        await provider.recover_bearer_groups(_verified_result(_delegated()), "ingress-token")

    exchange.assert_awaited_once_with(
        provider,
        subject_token="ingress-token",
        target_audience="https://graph.microsoft.com",
        scopes=["https://graph.microsoft.com/User.Read"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "login_host,graph_resource",
    [
        ("login.microsoftonline.us", "https://graph.microsoft.us"),
        ("login.chinacloudapi.cn", "https://microsoftgraph.chinacloudapi.cn"),
    ],
)
async def test_graph_proxy_obo_uses_sovereign_login_resource(
    monkeypatch, login_host, graph_resource
):
    monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", f"https://{login_host}")
    monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
    provider = EntraIdProvider(
        tenant_id="tenant-one", client_id="gateway-app", client_secret="secret"
    )
    with (
        patch(
            "auth_server.providers.entra.obo_exchange",
            new_callable=AsyncMock,
            return_value="graph-token",
        ) as exchange,
        patch.object(
            EntraIdProvider, "fetch_groups_via_graph", new_callable=AsyncMock, return_value=[]
        ),
    ):
        await provider.recover_bearer_groups(_verified_result(_delegated()), "ingress-token")

    exchange.assert_awaited_once_with(
        provider,
        subject_token="ingress-token",
        target_audience=graph_resource,
        scopes=[f"{graph_resource}/User.Read"],
    )


@pytest.mark.asyncio
async def test_membership_cache_isolated_by_verified_tenant_and_object(entra_provider):
    def page(gid):
        return httpx.Response(
            200,
            json={"value": [{"@odata.type": "#microsoft.graph.group", "id": gid}]},
            request=httpx.Request("GET", "https://graph.microsoft.com/v1.0/me/transitiveMemberOf"),
        )

    graph_client = AsyncMock()
    graph_client.get.side_effect = [page("alice-t1"), page("alice-t2"), page("bob-t1")]
    graph_client.__aenter__.return_value = graph_client
    with (
        patch(
            "auth_server.providers.entra.obo_exchange",
            new_callable=AsyncMock,
            return_value="graph-token",
        ) as obo,
        patch("auth_server.providers.entra.guarded_async_client", return_value=graph_client),
    ):
        recover = entra_provider.recover_bearer_groups
        for tid, oid, expected in (
            ("tenant-one", "object-alice", ["alice-t1"]),
            ("tenant-two", "object-alice", ["alice-t2"]),
            ("tenant-one", "object-bob", ["bob-t1"]),
            ("tenant-one", "object-alice", ["alice-t1"]),
        ):
            result = _verified_result(_delegated(tid=tid, oid=oid))
            assert await recover(result, "new-subject-token") == expected

    assert obo.await_count == 3
    assert graph_client.get.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["obo", "graph"])
async def test_transient_failed_recovery_is_not_cached(entra_provider, failure):
    from egress_obo import OboExchangeError

    obo = AsyncMock(return_value="graph-token")
    if failure == "obo":
        obo.side_effect = [OboExchangeError("IdP unavailable"), "graph-token"]
    graph_client = AsyncMock()
    graph_client.get.side_effect = (
        [
            httpx.Response(
                500,
                request=httpx.Request(
                    "GET", "https://graph.microsoft.com/v1.0/me/transitiveMemberOf"
                ),
            )
        ]
        if failure == "graph"
        else []
    ) + [
        httpx.Response(
            200,
            json={"value": [{"@odata.type": "#microsoft.graph.group", "id": "nested-parent"}]},
            request=httpx.Request("GET", "https://graph.microsoft.com/v1.0/me/transitiveMemberOf"),
        )
    ]
    graph_client.__aenter__.return_value = graph_client
    recover = entra_provider.recover_bearer_groups
    result = _verified_result(_delegated())
    with (
        patch("auth_server.providers.entra.obo_exchange", obo),
        patch("auth_server.providers.entra.guarded_async_client", return_value=graph_client),
    ):
        with pytest.raises(GraphGroupLookupError):
            await recover(result, "subject-token")
        assert await recover(result, "subject-token") == ["nested-parent"]

    assert obo.await_count == 2
    assert graph_client.get.await_count == (1 if failure == "obo" else 2)
