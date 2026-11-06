"""The session cookie the auth server sets is the one the registry reads.

The auth server sets the session cookie at login and reads it at ``/validate``;
the registry reads it when it resolves a browser session (the egress consent
routes, CSRF, logout). Both take its name from ``SESSION_COOKIE_NAME``. These
tests drive the real login callback and ``/validate`` and the registry's own
session check, with the default name and a configured one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import (
    AsyncMock,
    patch,
)

import pytest
from starlette.requests import Request

import registry.auth.dependencies as registry_dependencies
from registry.core.config import Settings
from tests.auth_server.unit.test_egress_vault_identity_properties import (
    _COLLIDING_PROFILE,
    _client,
    _FakeSessionCollection,
    _login,
    _session_store,
    _validate_hop_tokens,
    server_module,
)

pytestmark = [pytest.mark.unit, pytest.mark.auth]

_DEFAULT_NAME: str = "mcp_gateway_session"
_CUSTOM_NAME: str = "custom_session"
_ACCOUNT_ID: str = "account-1"


def _configure_session_cookie_name(
    monkeypatch: pytest.MonkeyPatch,
    configured: str | None,
) -> str:
    """Set the session cookie variables and load the name both services read.

    Args:
        monkeypatch: Sets the environment and the shared settings.
        configured: The ``SESSION_COOKIE_NAME`` value, or None to leave it unset.

    Returns:
        The session cookie name.
    """
    if configured is None:
        monkeypatch.delenv("SESSION_COOKIE_NAME", raising=False)
    else:
        monkeypatch.setenv("SESSION_COOKIE_NAME", configured)
    loaded = Settings()
    for field in (
        "session_cookie_name",
        "session_cookie_secure",
        "session_cookie_domain",
    ):
        monkeypatch.setattr(server_module.settings, field, getattr(loaded, field))
    return loaded.session_cookie_name


def _registry_sees_browser_session(
    cookie_name: str,
    cookie: str,
    context: dict,
    store: _FakeSessionCollection,
) -> bool:
    """Resolve the browser request with the registry's ``browser_session_context``.

    Args:
        cookie_name: Name the browser's session cookie is stored under.
        cookie: The session cookie value.
        context: The registry user context ``/validate`` produced for the request.
        store: The session collection the login wrote.

    Returns:
        True when the registry resolves the cookie to the session ``context`` names.
    """
    request = Request(
        {"type": "http", "headers": [(b"cookie", f"{cookie_name}={cookie}".encode())]}
    )
    with (
        patch.object(registry_dependencies, "signer", server_module.signer),
        patch("registry.auth.session_store._get_collection", AsyncMock(return_value=store)),
        patch(
            "registry.auth.dependencies.map_cognito_groups_to_scopes",
            AsyncMock(return_value=[]),
        ),
        patch(
            "registry.auth.dependencies.resolve_scope_access",
            AsyncMock(return_value=SimpleNamespace(servers=[], tools={})),
        ),
        patch(
            "registry.auth.dependencies.get_ui_permissions_for_user",
            AsyncMock(return_value={}),
        ),
    ):
        browser = asyncio.run(registry_dependencies.browser_session_context(request))
    return browser is not None and browser.get("session_id") == context.get("session_id")


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        pytest.param(None, _DEFAULT_NAME, id="unset"),
        pytest.param(_CUSTOM_NAME, _CUSTOM_NAME, id="configured"),
    ],
)
def test_login_validate_and_registry_agree_on_the_session_cookie_name(
    monkeypatch, configured, expected
):
    """The login sets the cookie under the configured name and every reader finds it."""
    name = _configure_session_cookie_name(monkeypatch, configured)

    login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)
    hop = _validate_hop_tokens({"Cookie": f"{name}={login.cookie}"}, login.store)

    assert name == expected
    assert login.cookie, login.set_cookie_headers
    assert hop.registry_context["auth_method"] == "oauth2", hop.registry_context
    assert _registry_sees_browser_session(name, login.cookie, hop.registry_context, login.store)


def test_validate_ignores_a_session_cookie_under_another_name(monkeypatch):
    """With a configured name, a live session cookie under the default name is refused."""
    _configure_session_cookie_name(monkeypatch, _CUSTOM_NAME)
    login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)
    live_cookie = server_module.signer.dumps(login.store.documents[0]["session_id"])

    with _session_store(login.store):
        response = _client().get(
            "/validate",
            headers={
                "Cookie": f"{_DEFAULT_NAME}={live_cookie}",
                "X-Original-URL": "https://gw.example.com/github/mcp",
            },
        )
    _client().cookies.clear()

    assert response.status_code == 401, response.text
