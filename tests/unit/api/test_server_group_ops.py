from __future__ import annotations

from unittest.mock import (
    AsyncMock,
    patch,
)

import pytest

from fastapi import (
    HTTPException,
)

from registry.api.server_group_ops import (
    _delete_group,
)


def _admin_user_context(
    user_context: dict | None,
) -> dict:
    if user_context is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not user_context.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin required")
    return user_context


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_group_denies_system_group_without_force() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _delete_group(
            group_name="mcp-registry-admin",
            delete_from_keycloak=True,
            force=False,
            user_context={"username": "admin", "is_admin": True},
            require_user_context=_admin_user_context,
        )

    assert exc_info.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_group_allows_system_group_with_force() -> None:
    with patch(
        "registry.utils.scopes_manager.delete_group_from_scopes",
        new=AsyncMock(return_value=True),
    ), patch(
        "registry.utils.keycloak_manager.group_exists_in_keycloak",
        new=AsyncMock(return_value=False),
    ), patch(
        "registry.utils.keycloak_manager.delete_keycloak_group",
        new=AsyncMock(),
    ):
        response = await _delete_group(
            group_name="mcp-registry-admin",
            delete_from_keycloak=True,
            force=True,
            user_context={"username": "admin", "is_admin": True},
            require_user_context=_admin_user_context,
        )

    assert response.status_code == 200
    payload = response.body.decode("utf-8")
    assert "Group deletion completed" in payload

