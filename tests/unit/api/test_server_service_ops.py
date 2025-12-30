from __future__ import annotations

from unittest.mock import (
    AsyncMock,
    Mock,
    patch,
)

import pytest
from fastapi import (
    HTTPException,
)

from registry.api.server_service_ops import (
    _register_service_external,
    _remove_service_external,
    _toggle_service_external,
)


def _can_modify_servers(
    user_context: dict | None,
) -> dict:
    if user_context is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not user_context.get("can_modify_servers", False):
        raise HTTPException(
            status_code=403,
            detail="Insufficient privileges to modify servers",
        )
    return user_context


def _close_coroutine(
    coro,
):
    coro.close()
    return None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_external_conflict_when_overwrite_false() -> None:
    mock_server_service = Mock()
    mock_server_service.get_server_info.return_value = {"server_name": "svc"}

    with patch(
        "registry.api.server_service_ops._enforce_proxy_pass_url_allowlist",
    ), patch(
        "registry.api.server_service_ops._build_server_entry_from_form",
        return_value=(
            "/svc",
            {"server_name": "svc", "proxy_pass_url": "https://example.com/mcp"},
        ),
    ):
        response = await _register_service_external(
            name="svc",
            description="svc",
            path="/svc",
            proxy_pass_url="https://example.com/mcp",
            tags="",
            num_tools=0,
            num_stars=0,
            is_python=False,
            license_str="N/A",
            overwrite=False,
            auth_provider=None,
            auth_type=None,
            upstream_auth=None,
            supported_transports=None,
            headers=None,
            tool_list_json=None,
            user_context={"username": "u", "can_modify_servers": True},
            require_user_context=_can_modify_servers,
            server_service_obj=mock_server_service,
            create_task=_close_coroutine,
            logger=Mock(),
        )

    assert response.status_code == 409


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_external_success_runs_side_effects() -> None:
    mock_server_service = Mock()
    mock_server_service.get_server_info.return_value = None
    mock_server_service.register_server.return_value = True
    mock_server_service.toggle_service.return_value = True
    mock_server_service.is_service_enabled.return_value = True
    mock_server_service.get_enabled_services.return_value = ["/svc"]

    with patch(
        "registry.api.server_service_ops._enforce_proxy_pass_url_allowlist",
    ), patch(
        "registry.api.server_service_ops._build_server_entry_from_form",
        return_value=(
            "/svc",
            {"server_name": "svc", "proxy_pass_url": "https://example.com/mcp"},
        ),
    ), patch(
        "registry.search.service.faiss_service.add_or_update_service",
        new=AsyncMock(),
    ) as mock_add_or_update, patch(
        "registry.search.service.faiss_service.save_data",
        new=AsyncMock(),
    ), patch(
        "registry.core.nginx_service.nginx_service.generate_config_async",
        new=AsyncMock(),
    ) as mock_generate_config, patch(
        "registry.health.service.health_service.broadcast_health_update",
        new=AsyncMock(),
    ) as mock_broadcast_health, patch(
        "registry.health.service.health_service.perform_immediate_health_check",
        new=AsyncMock(),
    ), patch(
        "registry.utils.scopes_manager.update_server_scopes",
        new=AsyncMock(),
    ):
        response = await _register_service_external(
            name="svc",
            description="svc",
            path="svc",
            proxy_pass_url="https://example.com/mcp",
            tags="",
            num_tools=0,
            num_stars=0,
            is_python=False,
            license_str="N/A",
            overwrite=True,
            auth_provider=None,
            auth_type=None,
            upstream_auth=None,
            supported_transports=None,
            headers=None,
            tool_list_json=None,
            user_context={"username": "u", "can_modify_servers": True},
            require_user_context=_can_modify_servers,
            server_service_obj=mock_server_service,
            create_task=_close_coroutine,
            logger=Mock(),
        )

    assert response.status_code == 201
    mock_server_service.toggle_service.assert_called_once_with("/svc", True)
    mock_add_or_update.assert_awaited()
    mock_generate_config.assert_awaited()
    mock_broadcast_health.assert_awaited_once_with("/svc")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_toggle_external_requires_path() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _toggle_service_external(
            path=None,
            service_path=None,
            new_state=None,
            user_context={"username": "u", "can_modify_servers": True},
            require_user_context=_can_modify_servers,
            server_service_obj=Mock(),
            logger=Mock(),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_external_not_found_returns_404() -> None:
    mock_server_service = Mock()
    mock_server_service.get_server_info.return_value = None

    with patch(
        "registry.api.server_service_ops._apply_remove_side_effects",
        new=AsyncMock(),
    ):
        response = await _remove_service_external(
            path="/missing",
            user_context={"username": "u", "can_modify_servers": True},
            require_user_context=_can_modify_servers,
            server_service_obj=mock_server_service,
            logger=Mock(),
        )

    assert response.status_code == 404

