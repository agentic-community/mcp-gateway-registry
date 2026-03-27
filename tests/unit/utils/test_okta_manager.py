"""Unit tests for OktaIAMManager (okta_manager.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


def _make_response(json_data, status_code=200, links=None, headers=None):
    """Create a mock httpx.Response with synchronous json()."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    resp.links = links or {}
    resp.headers = headers or {}
    return resp


def _make_async_client(**overrides):
    """Create a mock async httpx client."""
    client = AsyncMock()
    for method, value in overrides.items():
        setattr(client, method, value)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# =============================================================================
# USER MANAGEMENT TESTS
# =============================================================================


class TestOktaUserManagement:
    """Tests for Okta user management functions."""

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "test-api-token")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_list_users_ssws_auth(self):
        """Verifies SSWS authorization header is sent."""
        from registry.utils.okta_manager import list_okta_users

        resp = _make_response([])
        mock_client = _make_async_client()
        mock_client.get = AsyncMock(return_value=resp)

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            await list_okta_users()

            call_args = mock_client.get.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "SSWS test-api-token"

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "tok")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_list_users_pagination(self):
        """Pagination across multiple pages (200 per page)."""
        from registry.utils.okta_manager import list_okta_users

        page1_users = [
            {
                "id": f"u{i}",
                "profile": {
                    "login": f"u{i}@t.com",
                    "email": f"u{i}@t.com",
                    "firstName": "F",
                    "lastName": "L",
                },
                "status": "ACTIVE",
                "created": "2026-01-01",
            }
            for i in range(3)
        ]
        page2_users = [
            {
                "id": "u99",
                "profile": {
                    "login": "u99@t.com",
                    "email": "u99@t.com",
                    "firstName": "F",
                    "lastName": "L",
                },
                "status": "ACTIVE",
                "created": "2026-01-01",
            }
        ]

        resp1 = _make_response(
            page1_users, links={"next": {"url": "https://dev-123.okta.com/api/v1/users?after=abc"}}
        )
        resp2 = _make_response(page2_users)
        groups_resp = _make_response([{"profile": {"name": "users"}}])

        mock_client = _make_async_client()
        mock_client.get = AsyncMock(
            side_effect=[resp1, resp2, groups_resp, groups_resp, groups_resp, groups_resp]
        )

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            result = await list_okta_users(include_groups=True)
            assert len(result) == 4

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "tok")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_create_human_user_with_group_assignment(self):
        """User creation and group assignment flow."""
        from registry.utils.okta_manager import create_okta_human_user

        mock_client = _make_async_client()
        mock_client.post = AsyncMock(
            return_value=_make_response({"id": "u1", "profile": {"login": "new@t.com"}})
        )
        mock_client.get = AsyncMock(
            return_value=_make_response([{"id": "g1", "profile": {"name": "devs"}}])
        )
        mock_client.put = AsyncMock()

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            result = await create_okta_human_user("new@t.com", "new@t.com", "New", "User", ["devs"])

            assert result["username"] == "new@t.com"
            assert result["groups"] == ["devs"]
            mock_client.put.assert_called_once()

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "tok")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_delete_user_deactivates_then_deletes(self):
        """Two-step deactivate + delete flow."""
        from registry.utils.okta_manager import delete_okta_user

        mock_client = _make_async_client()
        mock_client.get = AsyncMock(return_value=_make_response({"id": "u1"}, status_code=200))
        mock_client.post = AsyncMock()
        mock_client.delete = AsyncMock(return_value=_make_response(None))

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            result = await delete_okta_user("user@test.com")
            assert result is True
            mock_client.post.assert_called_once()  # deactivate
            mock_client.delete.assert_called_once()  # delete

    @pytest.mark.asyncio
    async def test_rate_limit_429_raises_with_retry_after(self):
        """HTTP 429 raises ValueError with Retry-After."""
        from registry.utils.okta_manager import _check_rate_limit

        resp = _make_response(
            None, status_code=429, headers={"Retry-After": "30", "X-Rate-Limit-Remaining": "0"}
        )

        with pytest.raises(ValueError, match="Retry after 30 seconds"):
            _check_rate_limit(resp)


# =============================================================================
# GROUP MANAGEMENT TESTS
# =============================================================================


class TestOktaGroupManagement:
    """Tests for Okta group management functions."""

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "tok")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_list_groups_returns_all_fields(self):
        """Returns id, name, description, type for each group."""
        from registry.utils.okta_manager import list_okta_groups

        api_groups = [
            {
                "id": "g1",
                "profile": {"name": "admins", "description": "Admin group"},
                "type": "OKTA_GROUP",
            },
            {
                "id": "g2",
                "profile": {"name": "users", "description": "User group"},
                "type": "OKTA_GROUP",
            },
        ]

        mock_client = _make_async_client()
        mock_client.get = AsyncMock(return_value=_make_response(api_groups))

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            result = await list_okta_groups()

            assert len(result) == 2
            assert result[0]["id"] == "g1"
            assert result[0]["name"] == "admins"
            assert result[0]["description"] == "Admin group"
            assert result[0]["type"] == "OKTA_GROUP"

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "tok")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_create_group(self):
        """Group creation via Admin API."""
        from registry.utils.okta_manager import create_okta_group

        mock_client = _make_async_client()
        mock_client.post = AsyncMock(
            return_value=_make_response({"id": "g-new", "profile": {"name": "new-group"}})
        )

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            result = await create_okta_group("new-group", "A new group")
            assert result["name"] == "new-group"
            assert result["id"] == "g-new"

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "tok")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_delete_group_resolves_name_to_id(self):
        """Name-to-ID resolution before deletion."""
        from registry.utils.okta_manager import delete_okta_group

        mock_client = _make_async_client()
        mock_client.get = AsyncMock(
            return_value=_make_response([{"id": "g1", "profile": {"name": "target"}}])
        )
        mock_client.delete = AsyncMock(return_value=_make_response(None))

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            result = await delete_okta_group("target")
            assert result is True
            delete_url = mock_client.delete.call_args[0][0]
            assert "g1" in delete_url

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "tok")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_delete_group_not_found_raises(self):
        """ValueError when group name doesn't match."""
        from registry.utils.okta_manager import delete_okta_group

        mock_client = _make_async_client()
        mock_client.get = AsyncMock(return_value=_make_response([]))

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError, match="Group not found"):
                await delete_okta_group("nonexistent")


# =============================================================================
# SERVICE ACCOUNT TESTS
# =============================================================================


class TestOktaServiceAccount:
    """Tests for Okta service account management."""

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "tok")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_create_service_account(self):
        """OIDC service app with client_credentials grant type and group assignment."""
        from registry.utils.okta_manager import create_okta_service_account

        created_app = {
            "id": "app1",
            "credentials": {"oauthClient": {"client_id": "gen-cid", "client_secret": "gen-cs"}},
        }
        group_search = [{"id": "g1", "profile": {"name": "agents"}}]

        mock_client = _make_async_client()
        mock_client.post = AsyncMock(return_value=_make_response(created_app))
        mock_client.get = AsyncMock(return_value=_make_response(group_search))
        mock_client.put = AsyncMock()

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            result = await create_okta_service_account("my-agent", ["agents"])

            assert result["client_id"] == "gen-cid"
            assert result["client_secret"] == "gen-cs"
            assert result["groups"] == ["agents"]

            app_data = mock_client.post.call_args[1]["json"]
            assert "client_credentials" in app_data["settings"]["oauthClient"]["grant_types"]
            assert app_data["settings"]["oauthClient"]["application_type"] == "service"


# =============================================================================
# UPDATE OPERATIONS TESTS
# =============================================================================


class TestOktaUpdateOperations:
    """Tests for Okta update operations."""

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "tok")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_update_user_groups(self):
        """Update user groups calculates minimal diff."""
        from registry.utils.okta_manager import update_okta_user_groups

        user_resp = _make_response({"id": "u1"}, status_code=200)
        current_groups = [
            {"id": "g1", "profile": {"name": "old-group"}, "type": "OKTA_GROUP"},
            {"id": "g2", "profile": {"name": "keep-group"}, "type": "OKTA_GROUP"},
        ]
        current_groups_resp = _make_response(current_groups)
        all_groups = [
            {"id": "g1", "profile": {"name": "old-group"}},
            {"id": "g2", "profile": {"name": "keep-group"}},
            {"id": "g3", "profile": {"name": "new-group"}},
        ]
        all_groups_resp = _make_response(all_groups)

        mock_client = _make_async_client()
        mock_client.get = AsyncMock(side_effect=[user_resp, current_groups_resp, all_groups_resp])
        mock_client.delete = AsyncMock()
        mock_client.put = AsyncMock()

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            result = await update_okta_user_groups("user@test.com", ["keep-group", "new-group"])

            assert result["groups"] == ["keep-group", "new-group"]
            mock_client.delete.assert_called_once()  # remove old-group
            mock_client.put.assert_called_once()  # add new-group

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "tok")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_update_group(self):
        """Update group description resolves name to ID."""
        from registry.utils.okta_manager import update_okta_group

        search_resp = _make_response([{"id": "g1", "profile": {"name": "my-group"}}])
        put_resp = _make_response(None)

        mock_client = _make_async_client()
        mock_client.get = AsyncMock(return_value=search_resp)
        mock_client.put = AsyncMock(return_value=put_resp)

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            result = await update_okta_group("my-group", "Updated description")

            assert result["name"] == "my-group"
            assert result["description"] == "Updated description"
            put_url = mock_client.put.call_args[0][0]
            assert "g1" in put_url

    @pytest.mark.asyncio
    @patch("registry.utils.okta_manager.OKTA_API_TOKEN", "tok")
    @patch("registry.utils.okta_manager.OKTA_DOMAIN", "dev-123.okta.com")
    async def test_update_group_not_found_raises(self):
        """Update group raises ValueError when not found."""
        from registry.utils.okta_manager import update_okta_group

        mock_client = _make_async_client()
        mock_client.get = AsyncMock(return_value=_make_response([]))

        with patch("registry.utils.okta_manager.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError, match="Group not found"):
                await update_okta_group("nonexistent", "desc")
