"""Unit tests for Entra group resolution through Microsoft Graph."""

import asyncio
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from auth_server.providers.entra import EntraIdProvider, GraphGroupLookupError
from registry.utils.url_guard import ENTRA_GRAPH_PROFILE


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


def _mock_response(payload: object, status_code: int = 200):
    """Build an HTTP response stub for Graph requests."""
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


def _mock_client(*pages):
    client = AsyncMock()
    client.get = AsyncMock(side_effect=[_mock_response(page) for page in pages])
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    return client


class TestFetchGroupsViaGraph:
    """Graph returns transitive group IDs or raises on lookup failure."""

    @pytest.mark.asyncio
    async def test_returns_transitive_group_ids_and_filters_other_objects(self, monkeypatch):
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
        monkeypatch.delenv("ENTRA_LOGIN_BASE_URL", raising=False)
        client = _mock_client(
            {
                "value": [
                    {"@odata.type": "#microsoft.graph.group", "id": "direct-group"},
                    {"@odata.type": "#microsoft.graph.directoryRole", "id": "role-1"},
                    {"@odata.type": "#microsoft.graph.group", "id": "nested-parent-group"},
                ]
            }
        )
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            ids = await EntraIdProvider.fetch_groups_via_graph("token")

        assert ids == ["direct-group", "nested-parent-group"]
        assert "/v1.0/me/transitiveMemberOf" in client.get.call_args.args[0]

    @pytest.mark.asyncio
    async def test_pagination_combines_pages_and_dedupes(self, monkeypatch):
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
        monkeypatch.delenv("ENTRA_LOGIN_BASE_URL", raising=False)
        next_link = "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$skiptoken=foo"
        client = _mock_client(
            {
                "value": [{"@odata.type": "#microsoft.graph.group", "id": "g-1"}],
                "@odata.nextLink": next_link,
            },
            {
                "value": [
                    {"@odata.type": "#microsoft.graph.group", "id": "g-1"},
                    {"@odata.type": "#microsoft.graph.group", "id": "g-2"},
                ]
            },
        )
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            ids = await EntraIdProvider.fetch_groups_via_graph("token")

        assert ids == ["g-1", "g-2"]
        assert [call.args[0] for call in client.get.await_args_list][1] == next_link
        assert client.get.await_args_list[0].kwargs["params"] == {"$top": 999}
        assert client.get.await_args_list[1].kwargs["params"] is None
        assert all(
            call.kwargs["headers"]["Authorization"] == "Bearer token"
            for call in client.get.await_args_list
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", [403, httpx.ConnectError("dns failed")])
    async def test_failed_lookup_raises_instead_of_becoming_empty_membership(self, failure):
        client = AsyncMock()
        if failure == 403:
            client.get = AsyncMock(return_value=_mock_response({"error": "forbidden"}, 403))
        else:
            client.get = AsyncMock(side_effect=failure)
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError) as error:
                await EntraIdProvider.fetch_groups_via_graph("token")
        assert error.value.reason == ("authorization" if failure == 403 else "transport")

    @pytest.mark.asyncio
    async def test_hard_cap_fails_closed_instead_of_returning_partial_groups(self, monkeypatch):
        monkeypatch.setattr(EntraIdProvider, "GROUP_FETCH_HARD_CAP", 5)
        page = {
            "value": [{"@odata.type": "#microsoft.graph.group", "id": f"g-{i}"} for i in range(10)]
        }
        client = _mock_client(page)
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError) as error:
                await EntraIdProvider.fetch_groups_via_graph("token")
        assert error.value.reason == "limit"

    @pytest.mark.asyncio
    async def test_empty_graph_membership_is_a_successful_lookup(self):
        client = _mock_client({"value": []})
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            assert await EntraIdProvider.fetch_groups_via_graph("token") == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            [],
            {"value": None},
            {"value": {"id": "group"}},
            {"value": [None]},
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": ["bad"]}]},
            {"value": [], "@odata.nextLink": 42},
        ],
    )
    async def test_malformed_graph_pages_raise(self, payload):
        client = _mock_client(payload)
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError):
                await EntraIdProvider.fetch_groups_via_graph("token")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "next_link",
        [
            "https://attacker.example/v1.0/me/transitiveMemberOf?$skiptoken=leak",
            "http://graph.microsoft.com/v1.0/me/transitiveMemberOf?$skiptoken=leak",
            "https://graph.microsoft.com.attacker.example/v1.0/me/transitiveMemberOf",
            "https://graph.microsoft.com@attacker.example/v1.0/me/transitiveMemberOf",
            "https://graph.microsoft.com/v1.0/users/other/transitiveMemberOf?$skiptoken=leak",
            "//attacker.example/v1.0/me/transitiveMemberOf?$skiptoken=leak",
        ],
    )
    async def test_unsafe_pagination_never_sends_bearer_token(self, monkeypatch, next_link):
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
        monkeypatch.delenv("ENTRA_LOGIN_BASE_URL", raising=False)
        client = _mock_client({"value": [], "@odata.nextLink": next_link})
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError):
                await EntraIdProvider.fetch_groups_via_graph("token")
        client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pagination_cycle_raises_without_repeating_request(self, monkeypatch):
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
        monkeypatch.delenv("ENTRA_LOGIN_BASE_URL", raising=False)
        initial_url = EntraIdProvider._graph_memberof_url()
        client = _mock_client({"value": [], "@odata.nextLink": initial_url})
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError):
                await EntraIdProvider.fetch_groups_via_graph("token")
        client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_page_limit_fails_without_accepting_incomplete_membership(self, monkeypatch):
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
        monkeypatch.delenv("ENTRA_LOGIN_BASE_URL", raising=False)
        monkeypatch.setattr(EntraIdProvider, "GROUP_FETCH_PAGE_HARD_CAP", 2)
        base = "https://graph.microsoft.com/v1.0/me/transitiveMemberOf"
        client = _mock_client(
            {
                "value": [{"@odata.type": "#microsoft.graph.group", "id": "first"}],
                "@odata.nextLink": f"{base}?skiptoken=1",
            },
            {
                "value": [{"@odata.type": "#microsoft.graph.group", "id": "second"}],
                "@odata.nextLink": f"{base}?skiptoken=2",
            },
        )
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError):
                await EntraIdProvider.fetch_groups_via_graph("token")
        assert client.get.await_count == 2

    @pytest.mark.asyncio
    async def test_failure_on_later_page_does_not_return_partial_groups(self, monkeypatch):
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
        monkeypatch.delenv("ENTRA_LOGIN_BASE_URL", raising=False)
        next_link = "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$skiptoken=next"
        client = _mock_client(
            {
                "value": [{"@odata.type": "#microsoft.graph.group", "id": "first"}],
                "@odata.nextLink": next_link,
            },
            {"error": "Graph rejected continuation"},
        )
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError):
                await EntraIdProvider.fetch_groups_via_graph("token")
        assert client.get.await_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "login_host,expected_graph_host",
        [
            ("login.microsoftonline.com", "graph.microsoft.com"),
            ("login.microsoftonline.us", "graph.microsoft.us"),
            ("login.chinacloudapi.cn", "microsoftgraph.chinacloudapi.cn"),
        ],
    )
    async def test_login_cloud_selects_matching_graph_endpoint(
        self, monkeypatch, login_host, expected_graph_host
    ):
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", f"https://{login_host}")
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
        client = _mock_client({"value": []})
        with patch(
            "auth_server.providers.entra.guarded_async_client", return_value=client
        ) as factory:
            assert await EntraIdProvider.fetch_groups_via_graph("token") == []
        assert client.get.await_args.args[0] == (
            f"https://{expected_graph_host}/v1.0/me/transitiveMemberOf"
        )
        assert factory.call_args.kwargs["follow_redirects"] is False
        assert factory.call_args.kwargs["profile"] is ENTRA_GRAPH_PROFILE

    @pytest.mark.asyncio
    async def test_unknown_login_with_explicit_resource_rewrites_native_graph_continuation(
        self, monkeypatch
    ):
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.private.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com/")
        monkeypatch.setenv("ENTRA_GRAPH_RESOURCE_URL", "https://graph.microsoft.com")
        next_link = "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$skiptoken=next"
        client = _mock_client(
            {
                "value": [{"@odata.type": "#microsoft.graph.group", "id": "first"}],
                "@odata.nextLink": next_link,
            },
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "second"}]},
        )
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            ids = await EntraIdProvider.fetch_groups_via_graph("delegated-token")

        assert ids == ["first", "second"]
        assert [call.args[0] for call in client.get.await_args_list] == [
            "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf",
            "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf?$skiptoken=next",
        ]
        assert all(
            call.kwargs["headers"]["Authorization"] == "Bearer delegated-token"
            for call in client.get.await_args_list
        )

    @pytest.mark.asyncio
    async def test_unknown_login_without_resource_rejects_native_graph_continuation(
        self, monkeypatch
    ):
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.private.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        monkeypatch.delenv("ENTRA_GRAPH_RESOURCE_URL", raising=False)
        client = _mock_client(
            {
                "value": [],
                "@odata.nextLink": "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$skiptoken=next",
            }
        )
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError, match="pagination") as error:
                await EntraIdProvider.fetch_groups_via_graph("delegated-token")
        assert error.value.reason == "pagination"
        client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_login_rejects_other_native_graph_continuation(self, monkeypatch):
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.private.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_RESOURCE_URL", "https://dod-graph.microsoft.us")
        client = _mock_client(
            {
                "value": [],
                "@odata.nextLink": "https://graph.microsoft.us/v1.0/me/transitiveMemberOf?$skiptoken=x",
            }
        )
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError) as error:
                await EntraIdProvider.fetch_groups_via_graph("token")
        assert error.value.reason == "pagination"
        client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_login_without_proxy_fails_before_graph_request(self, monkeypatch):
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.private.example.com")
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
        client = _mock_client({"value": []})
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError, match="Unknown Entra cloud"):
                await EntraIdProvider.fetch_groups_via_graph("delegated-token")
        client.get.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "login_host,graph_host",
        [
            ("login.microsoftonline.com", "graph.microsoft.us"),
            ("login.microsoftonline.us", "graph.microsoft.com"),
            ("login.private.example.com", "graph.microsoft.com"),
        ],
    )
    async def test_known_graph_endpoint_must_match_login_cloud(
        self, monkeypatch, login_host, graph_host
    ):
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", f"https://{login_host}")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", f"https://{graph_host}")
        client = _mock_client({"value": []})
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError, match="Graph endpoint does not match"):
                await EntraIdProvider.fetch_groups_via_graph("delegated-token")
        client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_graph_proxy_pagination_remains_on_configured_path(self, monkeypatch):
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        next_link = "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf?$skiptoken=next"
        client = _mock_client({"value": [], "@odata.nextLink": next_link}, {"value": []})
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            assert await EntraIdProvider.fetch_groups_via_graph("token") == []
        assert [call.args[0] for call in client.get.await_args_list] == [
            "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf",
            next_link,
        ]

    @pytest.mark.asyncio
    async def test_graph_proxy_rewrites_graph_origin_nextlink_without_leaking_token(
        self, monkeypatch
    ):
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        next_link = "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$skiptoken=next"
        client = _mock_client(
            {
                "value": [{"@odata.type": "#microsoft.graph.group", "id": "first"}],
                "@odata.nextLink": next_link,
            },
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "second"}]},
        )
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            ids = await EntraIdProvider.fetch_groups_via_graph("graph-bearer")

        assert ids == ["first", "second"]
        assert [call.args[0] for call in client.get.await_args_list] == [
            "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf",
            "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf?$skiptoken=next",
        ]
        assert all(
            call.kwargs["headers"]["Authorization"] == "Bearer graph-bearer"
            for call in client.get.await_args_list
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "next_link",
        [
            "https://evil.proxy.example.com/v1.0/me/transitiveMemberOf?$skiptoken=next",
            "http://graph.proxy.example.com/v1.0/me/transitiveMemberOf?$skiptoken=next",
            "https://graph.proxy.example.com/v1.0/users/other/transitiveMemberOf?$skiptoken=next",
            "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf/extra?$skiptoken=next",
        ],
    )
    async def test_proxy_pagination_cannot_send_bearer_off_origin_or_path(
        self, monkeypatch, next_link
    ):
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        client = _mock_client({"value": [], "@odata.nextLink": next_link})
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError):
                await EntraIdProvider.fetch_groups_via_graph("token")
        client.get.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "configured_base",
        [
            "http://graph.proxy.example.com",
            "https://graph.proxy.example.com/extra",
            "https://graph.proxy.example.com?next=1",
            "https://user@graph.proxy.example.com",
            "https://graph.proxy.example.com:8443",
        ],
    )
    async def test_invalid_graph_base_rejected_without_sending_bearer(
        self, monkeypatch, configured_base
    ):
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", configured_base)
        client = _mock_client({"value": []})
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError):
                await EntraIdProvider.fetch_groups_via_graph("token")
        client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_graph_redirect_response_never_forwards_bearer(self, monkeypatch):
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        client = AsyncMock()
        client.get.return_value = httpx.Response(
            302,
            headers={"Location": "https://attacker.example/collect"},
            request=httpx.Request(
                "GET", "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf"
            ),
        )
        client.__aenter__.return_value = client
        with patch(
            "auth_server.providers.entra.guarded_async_client", return_value=client
        ) as factory:
            with pytest.raises(GraphGroupLookupError):
                await EntraIdProvider.fetch_groups_via_graph("token")
        assert factory.call_args.kwargs["follow_redirects"] is False
        client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_authorization_header_uses_bearer_token(self):
        client = _mock_client({"value": []})
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            await EntraIdProvider.fetch_groups_via_graph("the-access-token")
        client.get.assert_awaited_once()
        assert client.get.call_args.kwargs["headers"]["Authorization"] == "Bearer the-access-token"


class TestGraphProxyBearerRecovery:
    """A Graph reverse proxy is not an OAuth audience for bearer OBO."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "login_host,graph_resource",
        [
            ("login.microsoftonline.com", "https://graph.microsoft.com"),
            ("login.microsoftonline.us", "https://graph.microsoft.us"),
        ],
    )
    async def test_known_cloud_obo_requests_native_graph_resource(
        self, monkeypatch, login_host, graph_resource
    ):
        monkeypatch.setattr(EntraIdProvider, "_group_cache", OrderedDict())
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", f"https://{login_host}")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        provider = EntraIdProvider("tenant", "app", "secret")
        client = _mock_client({"value": [{"@odata.type": "#microsoft.graph.group", "id": "group"}]})
        with (
            patch(
                "auth_server.providers.entra.obo_exchange",
                new_callable=AsyncMock,
                return_value="graph-token",
            ) as exchange,
            patch("auth_server.providers.entra.guarded_async_client", return_value=client),
        ):
            groups = await EntraIdProvider._lookup_bearer_groups(
                (EntraIdProvider._graph_memberof_url(), "tenant", "object"),
                "ingress-token",
                provider,
            )

        assert groups == ["group"]
        exchange.assert_awaited_once_with(
            provider,
            subject_token="ingress-token",
            target_audience=graph_resource,
            scopes=[f"{graph_resource}/User.Read"],
        )
        assert client.get.await_args.args[0] == (
            "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf"
        )
        assert client.get.await_args.kwargs["headers"] == {"Authorization": "Bearer graph-token"}

    @pytest.mark.asyncio
    async def test_unknown_cloud_proxy_cannot_be_implicit_obo_resource(self, monkeypatch):
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.private.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        monkeypatch.delenv("ENTRA_GRAPH_RESOURCE_URL", raising=False)
        provider = EntraIdProvider("tenant", "app", "secret")
        with patch("auth_server.providers.entra.obo_exchange", new_callable=AsyncMock) as exchange:
            with pytest.raises(
                GraphGroupLookupError, match="cannot determine Graph token resource"
            ) as error:
                await provider.recover_bearer_groups(
                    {
                        "valid": True,
                        "method": "entra",
                        "data": {"scp": "mcp.read", "tid": "tenant", "oid": "object"},
                    },
                    "ingress-token",
                )
        assert error.value.reason == "configuration"
        exchange.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_cloud_explicit_resource_obo_uses_native_audience_and_proxy_fetch(
        self, monkeypatch
    ):
        monkeypatch.setattr(EntraIdProvider, "_group_cache", OrderedDict())
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.private.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_RESOURCE_URL", "https://dod-graph.microsoft.us")
        provider = EntraIdProvider("tenant", "app", "secret")
        client = _mock_client({"value": [{"@odata.type": "#microsoft.graph.group", "id": "group"}]})
        with (
            patch(
                "auth_server.providers.entra.obo_exchange",
                new_callable=AsyncMock,
                return_value="graph-token",
            ) as exchange,
            patch("auth_server.providers.entra.guarded_async_client", return_value=client),
        ):
            groups = await provider.recover_bearer_groups(
                {
                    "valid": True,
                    "method": "entra",
                    "data": {"scp": "mcp.read", "tid": "tenant", "oid": "object"},
                },
                "ingress-token",
            )
        assert groups == ["group"]
        exchange.assert_awaited_once_with(
            provider,
            subject_token="ingress-token",
            target_audience="https://dod-graph.microsoft.us",
            scopes=["https://dod-graph.microsoft.us/User.Read"],
        )
        assert client.get.await_args.args[0] == (
            "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf"
        )
        assert client.get.await_args.kwargs["headers"] == {"Authorization": "Bearer graph-token"}

    @pytest.mark.asyncio
    async def test_dod_explicit_resource_uses_dod_for_obo_and_native_pagination(self, monkeypatch):
        monkeypatch.setattr(EntraIdProvider, "_group_cache", OrderedDict())
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.microsoftonline.us")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_RESOURCE_URL", "https://dod-graph.microsoft.us")
        provider = EntraIdProvider("tenant", "app", "secret")
        client = _mock_client(
            {
                "value": [],
                "@odata.nextLink": "https://dod-graph.microsoft.us/v1.0/me/transitiveMemberOf?$skiptoken=next",
            },
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "dod-group"}]},
        )
        with (
            patch(
                "auth_server.providers.entra.obo_exchange",
                new_callable=AsyncMock,
                return_value="dod-token",
            ) as exchange,
            patch("auth_server.providers.entra.guarded_async_client", return_value=client),
        ):
            result = await provider.recover_bearer_groups(
                {
                    "valid": True,
                    "method": "entra",
                    "data": {"scp": "mcp.read", "tid": "tenant", "oid": "dod-user"},
                },
                "ingress-token",
            )
        assert result == ["dod-group"]
        assert exchange.await_args.kwargs["target_audience"] == "https://dod-graph.microsoft.us"
        assert [call.args[0] for call in client.get.await_args_list] == [
            "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf",
            "https://graph.proxy.example.com/v1.0/me/transitiveMemberOf?$skiptoken=next",
        ]

    @pytest.mark.parametrize(
        "login_host,resource",
        [
            ("login.microsoftonline.com", "https://graph.microsoft.us"),
            ("login.microsoftonline.com", "https://dod-graph.microsoft.us"),
            ("login.microsoftonline.us", "https://graph.microsoft.com"),
            ("login.chinacloudapi.cn", "https://graph.microsoft.com"),
        ],
    )
    def test_wrong_cloud_resource_rejected_before_provider_setup(
        self, monkeypatch, login_host, resource
    ):
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", f"https://{login_host}")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_RESOURCE_URL", resource)
        with pytest.raises(GraphGroupLookupError, match="resource does not match") as error:
            EntraIdProvider("tenant", "app", "secret")
        assert error.value.reason == "configuration"

    @pytest.mark.parametrize(
        "resource",
        [
            "http://graph.microsoft.com",
            "https://graph.proxy.example.com",
            "https://graph.microsoft.com.attacker.example",
            "https://graph.microsoft.com@attacker.example",
            "https://user@graph.microsoft.com",
            "https://graph.microsoft.com:8443",
            "https://graph.microsoft.com/v1.0",
            "https://graph.microsoft.com?next=1",
            "https://graph.microsoft.com#fragment",
            "https://graph.microsoft.com ",
        ],
    )
    def test_invalid_resource_rejected_before_token_exchange(self, monkeypatch, resource):
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.private.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_RESOURCE_URL", resource)
        with pytest.raises(GraphGroupLookupError) as error:
            EntraIdProvider("tenant", "app", "secret")
        assert error.value.reason == "configuration"

    def test_empty_resource_setting_keeps_known_default_but_not_unknown_proxy(self, monkeypatch):
        monkeypatch.setenv("ENTRA_GRAPH_RESOURCE_URL", "")
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.microsoftonline.us")
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
        assert EntraIdProvider._graph_memberof_url().startswith("https://graph.microsoft.us/")
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.private.example.com")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        from auth_server.providers.entra import _graph_resource_url

        with pytest.raises(GraphGroupLookupError, match="cannot determine Graph token resource"):
            _graph_resource_url()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error_type,reason",
        [
            ("OboConsentRequired", "consent"),
            ("OboConfigError", "configuration"),
            ("OboExchangeError", "token_exchange"),
        ],
    )
    async def test_obo_failures_report_safe_reason(self, monkeypatch, error_type, reason):
        from auth_server.providers import entra as entra_module

        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.microsoftonline.com")
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
        monkeypatch.delenv("ENTRA_GRAPH_RESOURCE_URL", raising=False)
        provider = EntraIdProvider("tenant", "app", "secret")
        error = getattr(entra_module, error_type)("secret assertion never belongs in reason")
        with patch("auth_server.providers.entra.obo_exchange", new_callable=AsyncMock) as exchange:
            exchange.side_effect = error
            with pytest.raises(GraphGroupLookupError) as failure:
                await EntraIdProvider._lookup_bearer_groups(
                    (EntraIdProvider._graph_memberof_url(), "tenant", "object"),
                    "ingress-token",
                    provider,
                )
        assert failure.value.reason == reason
        assert "secret" not in failure.value.reason

    def test_reason_cannot_be_overridden_with_sensitive_text(self):
        assert GraphGroupLookupError("failure", reason="secret assertion").reason == "unknown"

    @pytest.mark.parametrize(
        "result,expected",
        [
            ({"method": "entra", "valid": True, "data": {"scp": "read"}}, True),
            ({"method": "entra", "valid": True, "data": {"scope": "read"}}, True),
            ({"method": "entra", "valid": True, "data": {"scope": "   "}}, False),
            ({"method": "entra", "valid": True, "data": {"scp": "  "}}, False),
            ({"method": "entra", "valid": True, "data": {"roles": ["app"]}}, False),
            ({"method": "entra", "valid": False, "data": {"scp": "read"}}, False),
            ({"method": "other", "valid": True, "data": {"scp": "read"}}, False),
        ],
    )
    def test_only_verified_delegated_entra_users_qualify(self, result, expected):
        assert EntraIdProvider.is_delegated_user(result) is expected


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
        "ENTRA_GRAPH_RESOURCE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    params = {"tenant_id": "tenant-abc", "client_id": "app-guid", "client_secret": "s"}
    params.update(kwargs)
    return EntraIdProvider(**params)


class TestGuardedUserInfo:
    """Graph userinfo keeps the bearer on the selected HTTPS Graph origin."""

    @pytest.fixture(autouse=True)
    def direct_guarded_egress(self, monkeypatch):
        from registry.utils import url_guard

        monkeypatch.setattr(
            url_guard, "_forward_proxy_config", lambda: url_guard._ForwardProxyConfig()
        )

    @staticmethod
    def dns_answer(address):
        return [(2, 1, 6, "", (address, 443))]

    def test_native_graph_userinfo_resolves_and_pins_bearer(self, monkeypatch):
        provider = _entra(monkeypatch)
        payload = {"preferred_username": "alice@example.com", "email": "alice@example.com"}

        def respond(_transport, request):
            return httpx.Response(200, json=payload, request=request)

        with (
            patch(
                "registry.utils.url_guard.socket.getaddrinfo",
                return_value=self.dns_answer("8.8.8.8"),
            ) as dns,
            patch(
                "httpx.HTTPTransport.handle_request", autospec=True, side_effect=respond
            ) as delegate,
            patch("auth_server.providers.entra.requests.get") as unguarded,
        ):
            assert provider.get_user_info("private-bearer") == payload

        dns.assert_called_once_with("graph.microsoft.com", 443, proto=6)
        outbound = delegate.call_args.args[1]
        assert outbound.url == httpx.URL("https://8.8.8.8/oidc/userinfo")
        assert outbound.headers["Host"] == "graph.microsoft.com"
        assert outbound.extensions["sni_hostname"] == "graph.microsoft.com"
        assert outbound.headers["Authorization"] == "Bearer private-bearer"
        unguarded.assert_not_called()

    @pytest.mark.parametrize(
        "url",
        [
            "http://graph.microsoft.com/oidc/userinfo",
            "https://169.254.169.254/oidc/userinfo",
            "https://127.0.0.1/oidc/userinfo",
            "https://graph.microsoft.com.attacker.example/oidc/userinfo",
        ],
    )
    def test_invalid_userinfo_destination_never_receives_bearer(self, monkeypatch, url, caplog):
        provider = _entra(monkeypatch)
        provider.userinfo_url = url
        with (
            patch("registry.utils.url_guard.socket.getaddrinfo") as dns,
            patch("httpx.HTTPTransport.handle_request", autospec=True) as delegate,
            pytest.raises(ValueError, match="User info retrieval failed") as error,
        ):
            provider.get_user_info("private-bearer")
        dns.assert_not_called()
        delegate.assert_not_called()
        assert "private-bearer" not in str(error.value)
        assert "private-bearer" not in caplog.text

    @pytest.mark.parametrize(
        "invalid_origin",
        [
            "http://graph.proxy.example.com",
            "https://169.254.169.254",
            "https://graph.microsoft.com.attacker.example/extra",
        ],
    )
    def test_changed_invalid_graph_origin_never_receives_bearer(self, monkeypatch, invalid_origin):
        provider = _entra(monkeypatch)
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", invalid_origin)
        with (
            patch("registry.utils.url_guard.socket.getaddrinfo") as dns,
            patch("httpx.HTTPTransport.handle_request", autospec=True) as delegate,
            pytest.raises(ValueError, match="User info retrieval failed"),
        ):
            provider.get_user_info("private-bearer")
        dns.assert_not_called()
        delegate.assert_not_called()

    def test_exact_graph_proxy_can_pin_private_host_but_metadata_is_forbidden(self, monkeypatch):
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        provider = EntraIdProvider("tenant", "app", "secret")
        addresses = [self.dns_answer("10.12.34.5"), self.dns_answer("169.254.169.254")]

        def respond(_transport, request):
            return httpx.Response(200, json={"sub": "alice"}, request=request)

        with (
            patch("registry.utils.url_guard.socket.getaddrinfo", side_effect=addresses) as dns,
            patch(
                "httpx.HTTPTransport.handle_request", autospec=True, side_effect=respond
            ) as delegate,
        ):
            assert provider.get_user_info("private-bearer") == {"sub": "alice"}
            with pytest.raises(ValueError, match="User info retrieval failed"):
                provider.get_user_info("private-bearer")
        assert dns.call_count == 2
        delegate.assert_called_once()
        outbound = delegate.call_args.args[1]
        assert outbound.url == httpx.URL("https://10.12.34.5/oidc/userinfo")
        assert outbound.headers["Host"] == "graph.proxy.example.com"
        assert outbound.extensions["sni_hostname"] == "graph.proxy.example.com"

    def test_graph_redirect_does_not_follow_with_bearer(self, monkeypatch, caplog):
        provider = _entra(monkeypatch)

        def redirect(_transport, request):
            return httpx.Response(
                302,
                headers={"Location": "https://attacker.example/collect"},
                request=request,
            )

        with (
            patch(
                "registry.utils.url_guard.socket.getaddrinfo",
                return_value=self.dns_answer("8.8.8.8"),
            ) as dns,
            patch(
                "httpx.HTTPTransport.handle_request", autospec=True, side_effect=redirect
            ) as delegate,
            pytest.raises(ValueError, match="User info retrieval failed") as error,
        ):
            provider.get_user_info("private-bearer")
        dns.assert_called_once()
        delegate.assert_called_once()
        assert "private-bearer" not in str(error.value)
        assert "private-bearer" not in caplog.text

    def test_http_status_does_not_expose_bearer_in_error_or_log(self, monkeypatch, caplog):
        provider = _entra(monkeypatch)

        def forbidden(_transport, request):
            return httpx.Response(403, json={"error": "forbidden"}, request=request)

        with (
            patch(
                "registry.utils.url_guard.socket.getaddrinfo",
                return_value=self.dns_answer("8.8.8.8"),
            ),
            patch("httpx.HTTPTransport.handle_request", autospec=True, side_effect=forbidden),
            pytest.raises(ValueError, match="User info retrieval failed") as error,
        ):
            provider.get_user_info("private-bearer")
        assert "private-bearer" not in str(error.value)
        assert "private-bearer" not in caplog.text

    def test_native_graph_rebinding_to_private_address_is_denied(self, monkeypatch):
        provider = _entra(monkeypatch)
        with (
            patch(
                "registry.utils.url_guard.socket.getaddrinfo",
                return_value=self.dns_answer("10.12.34.5"),
            ) as dns,
            patch("httpx.HTTPTransport.handle_request", autospec=True) as delegate,
            pytest.raises(ValueError, match="User info retrieval failed"),
        ):
            provider.get_user_info("private-bearer")
        dns.assert_called_once()
        delegate.assert_not_called()


class TestResolveGroups:
    """Inline claims keep their behavior; missing claims consult Graph."""

    @pytest.fixture(autouse=True)
    def _isolate_cache(self, monkeypatch):
        monkeypatch.setattr(EntraIdProvider, "_group_cache", OrderedDict())

    @pytest.mark.asyncio
    async def test_inline_groups_do_not_call_graph(self, monkeypatch):
        provider = _entra(monkeypatch)
        with patch.object(
            EntraIdProvider, "fetch_groups_via_graph", new_callable=AsyncMock
        ) as graph:
            result = await provider.resolve_groups(
                {"groups": ["inline-direct", "inline-parent"], "tid": "tenant", "oid": "alice"},
                "access-token",
            )
        assert result == ["inline-direct", "inline-parent"]
        graph.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_nonempty_groups_take_precedence_over_roles(self, monkeypatch):
        provider = _entra(monkeypatch)
        with patch.object(
            EntraIdProvider, "fetch_groups_via_graph", new_callable=AsyncMock
        ) as graph:
            result = await provider.resolve_groups(
                {"groups": ["group-id"], "roles": ["role-id"]}, "access-token"
            )
        assert result == ["group-id"]
        graph.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_roles_without_groups_do_not_call_graph(self, monkeypatch):
        provider = _entra(monkeypatch)
        with patch.object(
            EntraIdProvider, "fetch_groups_via_graph", new_callable=AsyncMock
        ) as graph:
            result = await provider.resolve_groups(
                {"roles": ["reader", "writer"], "tid": "tenant", "oid": "alice"},
                "access-token",
            )
        assert result == ["reader", "writer"]
        graph.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_overage_without_graph_consent_drops_stale_inline_groups(self, monkeypatch):
        provider = _entra(monkeypatch)
        claims = {
            "groups": ["stale-inline"],
            "roles": ["verified-role"],
            "hasgroups": True,
        }
        with patch.object(
            EntraIdProvider, "fetch_groups_via_graph", new_callable=AsyncMock
        ) as graph:
            result = await provider.resolve_groups(claims, "access-token")

        assert result == ["verified-role"]
        graph.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "indicator",
        [
            {"hasgroups": True},
            {"_claim_names": {"groups": "src1"}},
        ],
    )
    async def test_overage_uses_graph_even_when_inline_groups_present(self, monkeypatch, indicator):
        provider = _entra(monkeypatch)
        claims = {"groups": ["stale-inline"], "tid": "tenant", "oid": "alice", **indicator}
        with patch.object(
            EntraIdProvider,
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["direct", "nested-parent"],
        ) as graph:
            result = await provider.resolve_groups(claims, "access-token", graph_requested=True)
        assert result == ["direct", "nested-parent"]
        graph.assert_awaited_once()
        assert graph.call_args.args[0] == "access-token"

    @pytest.mark.asyncio
    async def test_overage_uses_roles_when_groups_missing(self, monkeypatch):
        provider = _entra(monkeypatch)
        with patch.object(
            EntraIdProvider,
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent"],
        ) as graph:
            result = await provider.resolve_groups(
                {"roles": ["inline-role"], "hasgroups": True, "tid": "tenant", "oid": "alice"},
                "access-token",
                graph_requested=True,
            )
        assert result == ["inline-role", "nested-parent"]
        graph.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graph_overage_deduplicates_app_role_and_ignores_stale_inline_group(
        self, monkeypatch
    ):
        provider = _entra(monkeypatch)
        with patch.object(
            EntraIdProvider,
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent", "role-id"],
        ):
            result = await provider.resolve_groups(
                {"groups": ["stale-inline"], "roles": ["role-id"], "hasgroups": True},
                "access-token",
                graph_requested=True,
            )

        assert result == ["role-id", "nested-parent"]

    @pytest.mark.asyncio
    async def test_empty_graph_overage_preserves_roles_not_stale_groups(self, monkeypatch):
        provider = _entra(monkeypatch)
        with patch.object(
            EntraIdProvider, "fetch_groups_via_graph", new_callable=AsyncMock, return_value=[]
        ):
            result = await provider.resolve_groups(
                {"groups": ["stale-inline"], "roles": ["role-id"], "hasgroups": True},
                "access-token",
                graph_requested=True,
            )

        assert result == ["role-id"]

    @pytest.mark.asyncio
    async def test_missing_groups_and_roles_uses_graph(self, monkeypatch):
        provider = _entra(monkeypatch)
        with patch.object(
            EntraIdProvider,
            "fetch_groups_via_graph",
            new_callable=AsyncMock,
            return_value=["nested-parent"],
        ) as graph:
            result = await provider.resolve_groups(
                {"tid": "tenant", "oid": "alice"}, "access-token", graph_requested=True
            )
        assert result == ["nested-parent"]
        graph.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graph_failure_propagates_for_overage_and_missing_claims(self, monkeypatch):
        provider = _entra(monkeypatch)
        for claims in (
            {"hasgroups": True, "groups": ["stale"], "tid": "tenant", "oid": "alice"},
            {"tid": "tenant", "oid": "alice"},
        ):
            with patch.object(
                EntraIdProvider,
                "fetch_groups_via_graph",
                new_callable=AsyncMock,
                side_effect=GraphGroupLookupError("Graph failed"),
            ):
                with pytest.raises(GraphGroupLookupError):
                    await provider.resolve_groups(claims, "access-token", graph_requested=True)

    @pytest.mark.asyncio
    async def test_browser_graph_lookup_timeout_refuses_incomplete_groups(self, monkeypatch):
        from auth_server.providers import entra as entra_module

        monkeypatch.setattr(entra_module, "GRAPH_GROUP_LOOKUP_TIMEOUT_SECONDS", 0.01)

        async def never_returns(*_args, **_kwargs):
            await asyncio.Event().wait()

        provider = _entra(monkeypatch)
        with patch.object(EntraIdProvider, "fetch_groups_via_graph", side_effect=never_returns):
            with pytest.raises(GraphGroupLookupError, match="timed out"):
                await provider.resolve_groups(
                    {"hasgroups": True, "groups": ["stale-inline"]},
                    "access-token",
                    graph_requested=True,
                )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "claims",
        [
            {"groups": {"admin": True}},
            {"groups": ["valid", 42]},
            {"roles": {"admin": True}},
        ],
    )
    async def test_invalid_inline_claim_does_not_trigger_graph(self, monkeypatch, claims):
        provider = _entra(monkeypatch)
        with patch.object(
            EntraIdProvider, "fetch_groups_via_graph", new_callable=AsyncMock
        ) as graph:
            with pytest.raises(ValueError):
                await provider.resolve_groups(claims, "access-token")
        graph.assert_not_awaited()


class TestGraphGroupCache:
    """Successful Graph results belong to one tenant and object ID."""

    @pytest.fixture(autouse=True)
    def _isolate_cache(self, monkeypatch):
        monkeypatch.setattr(EntraIdProvider, "_group_cache", OrderedDict())
        monkeypatch.setattr(EntraIdProvider, "_group_denials", OrderedDict())
        monkeypatch.delenv("ENTRA_GROUP_RECOVERY_ENABLED", raising=False)
        monkeypatch.delenv("ENTRA_GROUP_CACHE_TTL_SECONDS", raising=False)
        monkeypatch.delenv("ENTRA_GRAPH_BASE_URL", raising=False)
        monkeypatch.delenv("ENTRA_LOGIN_BASE_URL", raising=False)
        monkeypatch.delenv("ENTRA_GRAPH_RESOURCE_URL", raising=False)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ttl", ["0", "29", "3601", "not-a-number"])
    async def test_invalid_ttl_cannot_serve_cached_memberships(self, monkeypatch, ttl):
        client = _mock_client({"value": [{"@odata.type": "#microsoft.graph.group", "id": "id"}]})
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            assert await EntraIdProvider.fetch_groups_via_graph(
                "token", tenant_id="tenant", object_id="user"
            ) == ["id"]
            monkeypatch.setenv("ENTRA_GROUP_CACHE_TTL_SECONDS", ttl)
            with pytest.raises(ValueError, match="ENTRA_GROUP_CACHE_TTL_SECONDS"):
                await EntraIdProvider.fetch_groups_via_graph(
                    "token", tenant_id="tenant", object_id="user"
                )
        client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_audience_change_cannot_reuse_graph_membership_cache(self, monkeypatch):
        monkeypatch.setenv("ENTRA_LOGIN_BASE_URL", "https://login.microsoftonline.us")
        monkeypatch.setenv("ENTRA_GRAPH_BASE_URL", "https://graph.proxy.example.com")
        client = _mock_client(
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "gov"}]},
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "dod"}]},
        )
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            assert await EntraIdProvider.fetch_groups_via_graph(
                "gov-token", tenant_id="tenant", object_id="user"
            ) == ["gov"]
            monkeypatch.setenv("ENTRA_GRAPH_RESOURCE_URL", "https://dod-graph.microsoft.us")
            assert await EntraIdProvider.fetch_groups_via_graph(
                "dod-token", tenant_id="tenant", object_id="user"
            ) == ["dod"]
        assert client.get.await_count == 2

    @pytest.mark.asyncio
    async def test_cache_keys_on_tenant_and_object_id_and_expires_at_configured_ttl(
        self, monkeypatch
    ):
        monkeypatch.setenv("ENTRA_GROUP_CACHE_TTL_SECONDS", "600")
        clock = [1000.0]
        client = _mock_client(
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "alice-t1"}]},
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "alice-t2"}]},
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "bob-t1"}]},
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "alice-t1-refreshed"}]},
        )
        with (
            patch("auth_server.providers.entra.time.monotonic", side_effect=lambda: clock[0]),
            patch("auth_server.providers.entra.guarded_async_client", return_value=client),
        ):
            fetch = EntraIdProvider.fetch_groups_via_graph
            assert await fetch("token", tenant_id="t1", object_id="alice") == ["alice-t1"]
            clock[0] = 1599.0
            assert await fetch("another-token", tenant_id="t1", object_id="alice") == ["alice-t1"]
            assert client.get.await_count == 1
            assert await fetch("token", tenant_id="t2", object_id="alice") == ["alice-t2"]
            assert await fetch("token", tenant_id="t1", object_id="bob") == ["bob-t1"]
            clock[0] = 1600.0
            assert await fetch("token", tenant_id="t1", object_id="alice") == ["alice-t1-refreshed"]
            assert client.get.await_count == 4

    @pytest.mark.asyncio
    async def test_cached_list_cannot_be_mutated_by_a_caller(self):
        client = _mock_client(
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "original"}]}
        )
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            first = await EntraIdProvider.fetch_groups_via_graph(
                "token", tenant_id="tenant", object_id="alice"
            )
            first.append("injected")
            second = await EntraIdProvider.fetch_groups_via_graph(
                "token", tenant_id="tenant", object_id="alice"
            )
        assert second == ["original"]
        client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graph_forbidden_reports_authorization_without_result_cache(self):
        client = _mock_client({"value": []})
        client.get.side_effect = [_mock_response({"error": "forbidden"}, 403)]
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError) as denied:
                await EntraIdProvider.fetch_groups_via_graph(
                    "token", tenant_id="tenant", object_id="alice"
                )
        assert denied.value.reason == "authorization"
        assert not EntraIdProvider._group_cache

    @pytest.mark.asyncio
    async def test_empty_result_is_cached_but_error_is_retried(self):
        client = _mock_client({"value": []})
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            fetch = EntraIdProvider.fetch_groups_via_graph
            assert await fetch("token", tenant_id="t", object_id="empty") == []
            assert await fetch("token", tenant_id="t", object_id="empty") == []
            assert client.get.await_count == 1

        client = AsyncMock()
        client.get = AsyncMock(
            side_effect=[
                _mock_response({"error": "forbidden"}, 403),
                _mock_response(
                    {"value": [{"@odata.type": "#microsoft.graph.group", "id": "recovered"}]}
                ),
            ]
        )
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            with pytest.raises(GraphGroupLookupError):
                await fetch("token", tenant_id="t", object_id="recover")
            assert await fetch("token", tenant_id="t", object_id="recover") == ["recovered"]
            assert client.get.await_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "identity",
        [
            {"tenant_id": "t"},
            {"object_id": "alice"},
            {"tenant_id": "", "object_id": "alice"},
            {"tenant_id": "t", "object_id": ""},
            {"tenant_id": 42, "object_id": "alice"},
            {"tenant_id": "t", "object_id": ["alice"]},
        ],
    )
    async def test_without_valid_identity_result_is_not_cached(self, identity):
        client = _mock_client(
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "first"}]},
            {"value": [{"@odata.type": "#microsoft.graph.group", "id": "second"}]},
        )
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            fetch = EntraIdProvider.fetch_groups_via_graph
            assert await fetch("token", **identity) == ["first"]
            assert await fetch("token", **identity) == ["second"]
        assert client.get.await_count == 2

    @pytest.mark.asyncio
    async def test_cache_evicts_oldest_entry_at_512(self):
        client = AsyncMock()
        client.get = AsyncMock(
            side_effect=[
                _mock_response(
                    {"value": [{"@odata.type": "#microsoft.graph.group", "id": f"g-{i}"}]}
                )
                for i in range(513)
            ]
            + [
                _mock_response(
                    {"value": [{"@odata.type": "#microsoft.graph.group", "id": "refetched"}]}
                )
            ]
        )
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None
        with patch("auth_server.providers.entra.guarded_async_client", return_value=client):
            fetch = EntraIdProvider.fetch_groups_via_graph
            for i in range(513):
                assert await fetch("token", tenant_id="t", object_id=f"user-{i}") == [f"g-{i}"]
            assert await fetch("token", tenant_id="t", object_id="user-512") == ["g-512"]
            assert client.get.await_count == 513
            assert await fetch("token", tenant_id="t", object_id="user-0") == ["refetched"]
            assert client.get.await_count == 514


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


class TestCredentialedTokenTransport:
    """Provider token methods cannot send secrets to an unsafe destination."""

    def test_m2m_token_secret_uses_pinned_credentialed_client(self, monkeypatch):
        from registry.utils import url_guard

        monkeypatch.setattr(
            url_guard, "_forward_proxy_config", lambda: url_guard._ForwardProxyConfig()
        )
        provider = _entra(monkeypatch)

        def respond(_transport, request):
            assert request.url == httpx.URL("https://8.8.8.8/tenant-abc/oauth2/v2.0/token")
            assert request.headers["Host"] == "login.microsoftonline.com"
            assert request.extensions["sni_hostname"] == "login.microsoftonline.com"
            assert b"client_secret=s" in request.content
            return httpx.Response(200, json={"access_token": "issued"}, request=request)

        with (
            patch(
                "registry.utils.url_guard.socket.getaddrinfo",
                return_value=[(2, 1, 6, "", ("8.8.8.8", 443))],
            ),
            patch("httpx.HTTPTransport.handle_request", autospec=True, side_effect=respond) as send,
            patch("auth_server.providers.entra.requests.post") as unguarded,
        ):
            result = provider.get_m2m_token()

        assert result == {"access_token": "issued"}
        send.assert_called_once()
        unguarded.assert_not_called()

    def test_private_m2m_destination_never_receives_client_secret(self, monkeypatch):
        provider = _entra(monkeypatch)
        provider.token_url = "https://169.254.169.254/token"
        with patch("httpx.HTTPTransport.handle_request", autospec=True) as send:
            with pytest.raises(ValueError, match="M2M token generation failed"):
                provider.get_m2m_token()
        send.assert_not_called()


class TestValidatedDelegatedScopes:
    """Only signed Entra access-token scopes reach authorization."""

    @pytest.mark.parametrize(
        "claim,expected",
        [
            ({"scp": "read:servers write:servers"}, ["read:servers", "write:servers"]),
            ({"scope": "read:servers"}, ["read:servers"]),
            ({"scp": "read:servers", "scope": "write:servers"}, ["read:servers"]),
        ],
    )
    def test_signed_delegated_scope_precedes_legacy_scope(self, monkeypatch, claim, expected):
        import json
        import time

        import jwt
        from cryptography.hazmat.primitives.asymmetric import rsa
        from jwt.algorithms import RSAAlgorithm

        provider = _entra(monkeypatch)
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
        jwk["kid"] = "test-key"
        now = int(time.time())
        token = jwt.encode(
            {
                "iss": provider.issuer_v2,
                "aud": provider.client_id,
                "sub": "alice",
                "tid": "tenant-abc",
                "oid": "object-alice",
                "iat": now,
                "exp": now + 3600,
                **claim,
            },
            private_key,
            algorithm="RS256",
            headers={"kid": "test-key"},
        )
        with patch.object(provider, "get_jwks", return_value={"keys": [jwk]}):
            result = provider.validate_token(token)
        assert result["scopes"] == expected

    def test_signed_non_string_scp_denied(self, monkeypatch):
        import json
        import time

        import jwt
        from cryptography.hazmat.primitives.asymmetric import rsa
        from jwt.algorithms import RSAAlgorithm

        provider = _entra(monkeypatch)
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
        jwk["kid"] = "test-key"
        now = int(time.time())
        token = jwt.encode(
            {
                "iss": provider.issuer_v2,
                "aud": provider.client_id,
                "sub": "alice",
                "scp": ["read:servers"],
                "iat": now,
                "exp": now + 3600,
            },
            private_key,
            algorithm="RS256",
            headers={"kid": "test-key"},
        )
        with patch.object(provider, "get_jwks", return_value={"keys": [jwk]}):
            with pytest.raises(ValueError, match="scp claim must be a string"):
                provider.validate_token(token)


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
