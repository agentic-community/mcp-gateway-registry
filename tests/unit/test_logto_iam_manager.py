"""Unit tests for the Logto IAM manager (registry/utils/iam_manager.py).

The manager is exercised against a stub Management API client, verifying the
role→group mapping, payload shapes, and diff logic without network access.
"""

import importlib
from typing import Any

import pytest

import registry.utils.iam_manager as iam_module
from registry.utils.logto_admin import LogtoAdminError


class StubClient:
    """Records calls and serves canned responses for the manager."""

    def __init__(self):
        self.calls: list[tuple[str, str, Any]] = []
        self.paged: dict[str, list[Any]] = {}
        self.responses: dict[tuple[str, str], Any] = {}

    async def _record(self, method: str, path: str, payload=None):
        self.calls.append((method, path, payload))
        return self.responses.get((method, path))

    async def get(self, path: str, params=None):
        return await self._record("GET", path, params)

    async def post(self, path: str, json=None):
        return await self._record("POST", path, json)

    async def patch(self, path: str, json=None):
        return await self._record("PATCH", path, json)

    async def delete(self, path: str):
        return await self._record("DELETE", path)

    async def get_paged(self, path: str, params=None, page_size=100, max_items=1000):
        await self._record("LIST", path, params)
        return self.paged.get(path, [])

    def method(self, method: str, path: str) -> list[Any]:
        return [c[2] for c in self.calls if c[0] == method and c[1] == path]


@pytest.fixture
def stub():
    return StubClient()


@pytest.fixture
def manager(stub):
    return iam_module.LogtoIAMManager(client=stub)


def test_factory_returns_logto_manager(monkeypatch):
    monkeypatch.setenv("AUTH_PROVIDER", "logto")
    importlib.reload(iam_module)
    assert isinstance(iam_module.get_iam_manager(), iam_module.LogtoIAMManager)


@pytest.mark.asyncio
async def test_list_groups_maps_roles_to_keycloak_shape(manager, stub):
    stub.paged["/api/roles"] = [
        {"id": "r1", "name": "legal", "description": "Legal team"},
        {"id": "r2", "name": "finance", "description": None},
    ]
    groups = await manager.list_groups()
    assert groups == [
        {
            "id": "r1",
            "name": "legal",
            "path": "/legal",
            "attributes": {"description": ["Legal team"]},
        },
        {
            "id": "r2",
            "name": "finance",
            "path": "/finance",
            "attributes": {"description": [""]},
        },
    ]


@pytest.mark.asyncio
async def test_list_groups_honours_prefix_filter(manager, stub, monkeypatch):
    monkeypatch.setattr(iam_module, "IDP_GROUP_FILTER_PREFIXES", ["mcp-"])
    stub.paged["/api/roles"] = [
        {"id": "r1", "name": "mcp-registry-admin", "description": ""},
        {"id": "r2", "name": "legal", "description": ""},
    ]
    names = [g["name"] for g in await manager.list_groups()]
    assert names == ["mcp-registry-admin"]


@pytest.mark.asyncio
async def test_create_group_posts_role_payload(manager, stub):
    stub.responses[("POST", "/api/roles")] = {
        "id": "r9",
        "name": "legal",
        "description": "Legal team",
    }
    group = await manager.create_group("legal", "Legal team")
    assert stub.method("POST", "/api/roles") == [{"name": "legal", "description": "Legal team"}]
    assert group["id"] == "r9" and group["path"] == "/legal"


@pytest.mark.asyncio
async def test_delete_group_missing_role_raises_404_message(manager, stub):
    stub.paged["/api/roles"] = []
    with pytest.raises(LogtoAdminError, match="HTTP 404"):
        await manager.delete_group("nope")


@pytest.mark.asyncio
async def test_delete_group_deletes_by_role_id(manager, stub):
    stub.paged["/api/roles"] = [{"id": "r1", "name": "legal", "description": ""}]
    assert await manager.delete_group("legal") is True
    assert stub.method("DELETE", "/api/roles/r1") == [None]


@pytest.mark.asyncio
async def test_update_user_groups_adds_and_removes(manager, stub):
    stub.paged["/api/users"] = [{"id": "u1", "username": "alice", "primaryEmail": "a@x.co"}]
    stub.paged["/api/users/u1/roles"] = [
        {"id": "r-old", "name": "legal"},
        {"id": "r-keep", "name": "finance"},
    ]
    stub.paged["/api/roles"] = [
        {"id": "r-new", "name": "audit", "description": ""},
        {"id": "r-keep", "name": "finance", "description": ""},
    ]
    result = await manager.update_user_groups("alice", ["finance", "audit"])
    assert result["groups"] == ["audit", "finance"]
    # Added audit (by role id), removed legal (by role id), kept finance.
    assert stub.method("POST", "/api/users/u1/roles") == [{"roleIds": ["r-new"]}]
    assert stub.method("DELETE", "/api/users/u1/roles/r-old") == [None]
    assert not stub.method("DELETE", "/api/users/u1/roles/r-keep")


@pytest.mark.asyncio
async def test_list_users_maps_logto_user_and_roles(manager, stub):
    stub.paged["/api/users"] = [
        {
            "id": "u1",
            "username": "alice",
            "primaryEmail": "alice@x.co",
            "name": "Alice Liddell",
            "isSuspended": False,
        }
    ]
    stub.paged["/api/users/u1/roles"] = [{"id": "r1", "name": "legal"}]
    users = await manager.list_users()
    assert users == [
        {
            "id": "u1",
            "username": "alice",
            "email": "alice@x.co",
            "firstName": "Alice",
            "lastName": "Liddell",
            "enabled": True,
            "groups": ["legal"],
        }
    ]


@pytest.mark.asyncio
async def test_create_service_account_creates_m2m_app_and_assigns_roles(manager, stub):
    stub.responses[("POST", "/api/applications")] = {
        "id": "app1",
        "secret": "s3cr3t",
        "type": "machine-to-machine",
    }
    stub.paged["/api/roles"] = [{"id": "r1", "name": "legal", "description": ""}]
    result = await manager.create_service_account("ci-bot", ["legal"], "CI pipeline")
    assert stub.method("POST", "/api/applications") == [
        {"name": "ci-bot", "description": "CI pipeline", "type": "machine-to-machine"}
    ]
    assert stub.method("POST", "/api/applications/app1/roles") == [{"roleIds": ["r1"]}]
    assert result["client_id"] == "app1" and result["secret"] == "s3cr3t"
