"""The session cookie the auth server sets is the one the registry reads.

The auth server sets the session cookie at login and reads it at ``/validate``;
the registry reads it when it resolves a browser session (the egress consent
routes, CSRF, logout). Both take its name from ``SESSION_COOKIE_NAME`` and
``SESSION_COOKIE_HOST_PREFIX``. These tests drive the real login callback and
``/validate`` and the registry's own session check, with the default name, a
configured one, and the ``__Host-`` prefix. The login takes the cookie's Domain
from ``SESSION_COOKIE_DOMAIN`` only, as the registry's logout does, and expires
the host-only variant when it sets one with a Domain, so a browser that kept one
from before a ``SESSION_COOKIE_DOMAIN`` change does not send two. Both services
take every attribute from one function, so their headers agree. The cookie's
Secure flag comes from ``SESSION_COOKIE_SECURE`` and its lifetime from
``SESSION_MAX_AGE_SECONDS``; the auth server refuses to start when the yml's
``session`` block sets any cookie key besides the shipped domain.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess  # nosec B404 - runs the service imports in a fresh interpreter
import sys
import time
from datetime import timedelta
from http.cookies import (
    Morsel,
    SimpleCookie,
)
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import (
    AsyncMock,
    patch,
)

import pytest
from itsdangerous import TimestampSigner
from starlette.requests import Request
from starlette.responses import Response

import registry.auth.dependencies as registry_dependencies
from registry.auth.session_cookie import clear_session_cookie
from registry.common.session_cookie import session_cookie_signing_salt
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
    host_prefix: bool = False,
) -> str:
    """Set the session cookie variables and load the name both services read.

    Args:
        monkeypatch: Sets the environment and the shared settings.
        configured: The ``SESSION_COOKIE_NAME`` value, or None to leave it unset.
        host_prefix: Whether to set ``SESSION_COOKIE_HOST_PREFIX=true``, with the
            Secure cookie and empty domain it requires.

    Returns:
        The effective session cookie name, ``__Host-`` included when the prefix is on.
    """
    if configured is None:
        monkeypatch.delenv("SESSION_COOKIE_NAME", raising=False)
    else:
        monkeypatch.setenv("SESSION_COOKIE_NAME", configured)
    if host_prefix:
        monkeypatch.setenv("SESSION_COOKIE_HOST_PREFIX", "true")
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "true")
        monkeypatch.delenv("SESSION_COOKIE_DOMAIN", raising=False)
    loaded = Settings()
    for field in (
        "session_cookie_name",
        "session_cookie_host_prefix",
        "session_cookie_secure",
        "session_cookie_domain",
    ):
        monkeypatch.setattr(server_module.settings, field, getattr(loaded, field))
    return loaded.session_cookie_effective_name


def _login_session_cookie(
    set_cookie_headers: list[str],
    name: str,
) -> Morsel:
    """Return the one ``Set-Cookie`` a login response sent for ``name`` with a value.

    Args:
        set_cookie_headers: Every ``Set-Cookie`` header of the login response.
        name: The session cookie name.

    Returns:
        The parsed cookie; an expiry of the other variant is skipped.
    """
    [morsel] = [
        morsel for morsel in _login_session_cookies(set_cookie_headers, name) if morsel.value
    ]
    return morsel


def _login_session_cookies(
    set_cookie_headers: list[str],
    name: str,
) -> list[Morsel]:
    """Return every ``Set-Cookie`` a login response sent for ``name``, parsed.

    Args:
        set_cookie_headers: Every ``Set-Cookie`` header of the login response.
        name: The session cookie name.

    Returns:
        The parsed cookies, in header order.
    """
    morsels = []
    for header in set_cookie_headers:
        parsed = SimpleCookie()
        parsed.load(header)
        morsels.extend(morsel for key, morsel in parsed.items() if key == name)
    return morsels


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


@pytest.mark.parametrize(
    ("host_prefix", "name", "secure"),
    [
        pytest.param(False, _DEFAULT_NAME, False, id="prefix-off"),
        pytest.param(True, f"__Host-{_DEFAULT_NAME}", True, id="prefix-on"),
    ],
)
def test_host_prefix_decides_the_login_cookie_name_and_attributes(
    monkeypatch, host_prefix, name, secure
):
    """The prefix names the cookie __Host-<name>: Secure, Path=/, no Domain.

    The test client talks plain HTTP, so without the prefix the cookie is not
    Secure, as today.
    """
    resolved = _configure_session_cookie_name(monkeypatch, None, host_prefix=host_prefix)

    login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)
    morsel = _login_session_cookie(login.set_cookie_headers, resolved)

    assert resolved == name
    assert (bool(morsel["secure"]), morsel["path"], morsel["domain"]) == (secure, "/", "")
    assert morsel["httponly"] is True


def test_validate_and_registry_read_the_host_prefixed_cookie(monkeypatch):
    """With the prefix on, /validate and the registry find the session under __Host-<name>."""
    name = _configure_session_cookie_name(monkeypatch, None, host_prefix=True)

    login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)
    hop = _validate_hop_tokens({"Cookie": f"{name}={login.cookie}"}, login.store)

    assert login.cookie, login.set_cookie_headers
    assert _registry_sees_browser_session(name, login.cookie, hop.registry_context, login.store)


@pytest.mark.parametrize(
    "signed_for",
    [
        pytest.param(None, id="the-cookie-the-prefix-replaced"),
        pytest.param(f"__Host-{_CUSTOM_NAME}", id="another-host-prefixed-name"),
    ],
)
def test_host_prefixed_cookie_refuses_a_value_signed_for_another_name(monkeypatch, signed_for):
    """With the prefix on, a live session's value signed under another name is refused.

    Turning the prefix on leaves the previous cookie in browsers, and it still
    reaches every host under the parent domain. Its value names a live session,
    so neither service may accept it as the ``__Host-`` cookie.
    """
    _configure_session_cookie_name(monkeypatch, None, host_prefix=True)
    login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)
    session_id = login.store.documents[0]["session_id"]
    salt = None if signed_for is None else session_cookie_signing_salt(signed_for, True)
    other = server_module.signer.dumps(session_id, salt=salt)

    assert _auth_server_accepts(login.cookie, login.store)
    assert _registry_accepts(login.cookie, login.store)
    assert not _auth_server_accepts(other, login.store)
    assert not _registry_accepts(other, login.store)


_EXAMPLE_DOMAIN: str = ".example.com"
# A domain oauth2_providers.yml names at run time that SESSION_COOKIE_DOMAIN does
# not. Startup refuses that pair; set after startup, the login must ignore it.
_YML_DOMAIN: str = ".other.example"


@pytest.mark.parametrize(
    ("host_prefix", "configured_domain", "expected"),
    [
        pytest.param(False, None, [("set", "")], id="host-only"),
        pytest.param(
            False,
            _EXAMPLE_DOMAIN,
            [("set", _EXAMPLE_DOMAIN), ("expire", "")],
            id="domain-expires-the-host-only-cookie",
        ),
        pytest.param(True, None, [("set", "")], id="host-prefix-has-one-variant"),
    ],
)
def test_login_expires_the_other_session_cookie_variant(
    monkeypatch, host_prefix, configured_domain, expected
):
    """The login sets the session cookie on SESSION_COOKIE_DOMAIN and expires the other variant.

    A cookie set with a Domain and a host-only one are two cookies to the
    browser, so after ``SESSION_COOKIE_DOMAIN`` changes a browser holds both
    and sends both, which both services refuse. Setting one variant and
    expiring the other at Path=/ leaves one. A ``__Host-`` cookie never has a
    Domain, so with the prefix there is nothing to expire. A ``session.domain``
    in ``oauth2_providers.yml`` plays no part.
    """
    name = _configure_session_cookie_name(monkeypatch, None, host_prefix=host_prefix)
    monkeypatch.setitem(server_module.OAUTH2_CONFIG["session"], "domain", _YML_DOMAIN)
    monkeypatch.setattr(server_module.settings, "session_cookie_domain", configured_domain)

    login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)
    morsels = _login_session_cookies(login.set_cookie_headers, name)

    assert login.status_code == 302, login.set_cookie_headers
    assert sorted(("set" if m.value else "expire", m["domain"]) for m in morsels) == sorted(
        expected
    )
    for morsel in morsels:
        assert morsel["path"] == "/"
        if not morsel.value:
            assert morsel["max-age"] == "0"


@pytest.mark.parametrize("configured_domain", [None, "", _EXAMPLE_DOMAIN])
def test_registry_logout_expires_the_cookie_the_login_set(monkeypatch, configured_domain):
    """The registry's logout expiry covers the Domain the login set the session cookie on.

    A browser ignores an expiry whose Domain differs from the cookie's, so a
    login and a logout that took the domain from different places could leave
    the user signed in after logout.
    """
    name = _configure_session_cookie_name(monkeypatch, None)
    monkeypatch.setitem(server_module.OAUTH2_CONFIG["session"], "domain", _YML_DOMAIN)
    monkeypatch.setattr(server_module.settings, "session_cookie_domain", configured_domain)

    login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)
    set_domain = _login_session_cookie(login.set_cookie_headers, name)["domain"]
    logout = Response()
    clear_session_cookie(logout, _browser_request(https=False))
    expired = {
        m["domain"] for m in _login_session_cookies(logout.headers.getlist("set-cookie"), name)
    }

    assert set_domain in expired, (set_domain, expired)


def _browser_request(
    https: bool,
) -> Request:
    """Return a registry request from a browser that reached the gateway over HTTPS or HTTP."""
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/auth/logout",
            "scheme": "https" if https else "http",
            "server": ("gw.example.com", 443 if https else 80),
            "headers": [],
        }
    )


# (SESSION_COOKIE_HOST_PREFIX, SESSION_COOKIE_SECURE, request over HTTPS, the
#  cookie is Secure). Settings refuse the prefix with SESSION_COOKIE_SECURE=false,
#  so those rows are absent.
_SECURE_FLAG_ROWS = [
    pytest.param(False, True, True, True, id="setting-on-https"),
    pytest.param(False, False, True, False, id="setting-off-https"),
    pytest.param(False, True, False, False, id="setting-on-http"),
    pytest.param(False, False, False, False, id="setting-off-http"),
    pytest.param(True, True, True, True, id="prefix-https"),
    pytest.param(True, True, False, True, id="prefix-http"),
]


@pytest.mark.parametrize(("host_prefix", "setting_secure", "https", "secure"), _SECURE_FLAG_ROWS)
def test_session_cookie_is_secure_on_https_exactly_when_the_setting_says(
    monkeypatch, host_prefix, setting_secure, https, secure
):
    """Every set and expiry of the session cookie, in both services, carries one Secure flag.

    On HTTPS the cookie is Secure exactly when ``SESSION_COOKIE_SECURE`` is
    true. On plain HTTP it is never Secure, because a browser drops a Secure
    Set-Cookie there, except with the ``__Host-`` prefix, which a browser
    accepts only with Secure. The login's cookie, the login's expiry of the
    host-only variant and the registry's expiries (logout and a refused
    duplicate) all follow it. A ``session.secure`` in the provider config
    plays no part, so it is set to the opposite of the setting here.
    """
    name = _configure_session_cookie_name(monkeypatch, None, host_prefix=host_prefix)
    monkeypatch.setattr(server_module.settings, "session_cookie_secure", setting_secure)
    domain = None if host_prefix else _EXAMPLE_DOMAIN
    monkeypatch.setattr(server_module.settings, "session_cookie_domain", domain)
    monkeypatch.setitem(server_module.OAUTH2_CONFIG["session"], "secure", not setting_secure)

    with patch.object(server_module, "is_request_https", return_value=https):
        login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)
    login_cookies = _login_session_cookies(login.set_cookie_headers, name)
    expiry = Response()
    clear_session_cookie(expiry, _browser_request(https))
    registry_expiries = _login_session_cookies(expiry.headers.getlist("set-cookie"), name)

    assert len(login_cookies) == (1 if host_prefix else 2), login.set_cookie_headers
    assert len(registry_expiries) == (1 if host_prefix else 2), registry_expiries
    for morsel in login_cookies + registry_expiries:
        assert bool(morsel["secure"]) is secure, morsel.OutputString()


# A session lifetime other than the 28800-second default, short enough that a
# test can sign a cookie on either side of it.
_LIFETIME: int = 600


def _without_expires(
    set_cookie_header: str,
) -> str:
    """Return a ``Set-Cookie`` header without its Expires date, which is the time it was sent."""
    return re.sub(r"expires=[^;]*; ", "", set_cookie_header)


def _cookie_attributes(
    morsel: Morsel,
) -> tuple[str, str, bool, bool, str]:
    """Return the attributes a browser matches and stores: Domain, Path, Secure, HttpOnly, SameSite."""
    return (
        morsel["domain"],
        morsel["path"],
        bool(morsel["secure"]),
        bool(morsel["httponly"]),
        morsel["samesite"].lower(),
    )


# (SESSION_COOKIE_NAME, SESSION_COOKIE_HOST_PREFIX, SESSION_COOKIE_SECURE,
#  request over HTTPS, SESSION_COOKIE_DOMAIN, SESSION_MAX_AGE_SECONDS).
_ATTRIBUTE_ROWS = [
    pytest.param(None, False, True, True, None, 28800, id="defaults-https"),
    pytest.param(None, False, True, True, _EXAMPLE_DOMAIN, _LIFETIME, id="domain-https"),
    pytest.param(_CUSTOM_NAME, False, False, True, _EXAMPLE_DOMAIN, 28800, id="secure-off-https"),
    pytest.param(None, False, True, False, _EXAMPLE_DOMAIN, 28800, id="domain-http"),
    pytest.param(_CUSTOM_NAME, True, True, True, None, _LIFETIME, id="prefix-https"),
    pytest.param(None, True, True, False, None, 28800, id="prefix-http"),
]


@pytest.mark.parametrize(
    ("configured", "host_prefix", "setting_secure", "https", "domain", "max_age"),
    _ATTRIBUTE_ROWS,
)
def test_both_services_send_the_same_session_cookie_headers(
    monkeypatch, configured, host_prefix, setting_secure, https, domain, max_age
):
    """The login's cookie and expiries match the registry's expiries, header for header.

    The login's expiry of the other variant is byte for byte one of the
    registry's expiries (logout and a refused duplicate), and the registry
    expires the cookie the login set with its exact Domain, Path, Secure,
    HttpOnly and SameSite. A browser ignores an expiry whose name, Domain or
    Path differ, so a split here leaves a user signed in after logout.
    """
    name = _configure_session_cookie_name(monkeypatch, configured, host_prefix=host_prefix)
    monkeypatch.setattr(server_module.settings, "session_cookie_secure", setting_secure)
    monkeypatch.setattr(server_module.settings, "session_cookie_domain", domain)
    monkeypatch.setattr(server_module.settings, "session_max_age_seconds", max_age)

    with patch.object(server_module, "is_request_https", return_value=https):
        login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)
    logout = Response()
    clear_session_cookie(logout, _browser_request(https))
    login_headers = [
        header for header in login.set_cookie_headers if _login_session_cookies([header], name)
    ]
    registry_headers = logout.headers.getlist("set-cookie")
    set_cookie = _login_session_cookie(login.set_cookie_headers, name)
    login_expiries = [h for h in login_headers if not _login_session_cookies([h], name)[0].value]

    assert set_cookie["max-age"] == str(max_age)
    assert set_cookie["domain"] == ("" if host_prefix else domain or "")
    assert len(registry_headers) == (2 if domain and not host_prefix else 1), registry_headers
    assert {_without_expires(h) for h in login_expiries} <= {
        _without_expires(h) for h in registry_headers
    }, (login_expiries, registry_headers)
    assert _cookie_attributes(set_cookie) in {
        _cookie_attributes(m) for m in _login_session_cookies(registry_headers, name)
    }, (set_cookie.OutputString(), registry_headers)


def test_session_lifetime_sets_the_stored_ttl_and_the_cookie_max_age(monkeypatch):
    """SESSION_MAX_AGE_SECONDS is the stored session's lifetime and the cookie's Max-Age.

    A ``session.max_age_seconds`` left in the provider config plays no part.
    """
    name = _configure_session_cookie_name(monkeypatch, None)
    monkeypatch.setattr(server_module.settings, "session_max_age_seconds", _LIFETIME)
    monkeypatch.setitem(server_module.OAUTH2_CONFIG["session"], "max_age_seconds", 28800)

    login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)
    [document] = login.store.documents

    assert _login_session_cookie(login.set_cookie_headers, name)["max-age"] == str(_LIFETIME)
    assert document["expires_at"] - document["created_at"] == timedelta(seconds=_LIFETIME)


def _aged_session_cookie(
    session_id: str,
    age_seconds: int,
) -> str:
    """Return the session cookie a login signed ``age_seconds`` ago for ``session_id``.

    Args:
        session_id: The id of a session in the store.
        age_seconds: How long ago the cookie was signed.

    Returns:
        The signed cookie value, with the signer's timestamp moved back.
    """
    issued = int(time.time()) - age_seconds
    with patch.object(TimestampSigner, "get_timestamp", return_value=issued):
        return server_module.signer.dumps(session_id)


def _auth_server_accepts(
    cookie: str,
    store: _FakeSessionCollection,
) -> bool:
    """Return whether the auth server's ``validate_session_cookie`` accepts ``cookie``."""
    with (
        _session_store(store),
        patch.object(server_module, "map_groups_to_scopes", AsyncMock(return_value=[])),
    ):
        try:
            asyncio.run(server_module.validate_session_cookie(cookie))
        except ValueError:
            return False
    return True


def _registry_accepts(
    cookie: str,
    store: _FakeSessionCollection,
) -> bool:
    """Return whether the registry's ``resolve_session_from_cookie`` accepts ``cookie``."""
    with (
        patch.object(registry_dependencies, "signer", server_module.signer),
        patch("registry.auth.session_store._get_collection", AsyncMock(return_value=store)),
    ):
        record = asyncio.run(registry_dependencies.resolve_session_from_cookie(cookie))
    return record is not None


@pytest.mark.parametrize(
    ("age_seconds", "accepted"),
    [
        pytest.param(_LIFETIME - 60, True, id="younger-than-the-lifetime"),
        pytest.param(_LIFETIME + 60, False, id="older-than-the-lifetime"),
    ],
)
def test_both_services_check_the_cookie_signature_against_the_session_lifetime(
    monkeypatch, age_seconds, accepted
):
    """The registry and the auth server refuse a cookie signed longer ago than the lifetime.

    The session the cookie names is still live in the store, so only the
    signature's age decides.
    """
    _configure_session_cookie_name(monkeypatch, None)
    monkeypatch.setattr(server_module.settings, "session_max_age_seconds", _LIFETIME)
    login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)
    cookie = _aged_session_cookie(login.store.documents[0]["session_id"], age_seconds)

    assert _registry_accepts(cookie, login.store) is accepted
    assert _auth_server_accepts(cookie, login.store) is accepted


def test_login_redirect_check_reads_the_domain_the_login_sets(monkeypatch):
    """With SESSION_COOKIE_DOMAIN only in settings, the redirect check uses the cookie's domain.

    ``Settings`` also loads ``.env``, which never reaches ``os.environ``, so a
    domain set only there must still widen the login's redirect check to the
    domain the login sets the session cookie on.
    """
    name = _configure_session_cookie_name(monkeypatch, None)
    for variable in (
        "SESSION_COOKIE_DOMAIN",
        "OAUTH2_ALLOWED_REDIRECT_URIS",
        "AUTH_SERVER_EXTERNAL_URL",
        "REGISTRY_URL",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(server_module.settings, "session_cookie_domain", _EXAMPLE_DOMAIN)
    target = f"https://registry{_EXAMPLE_DOMAIN}/dashboard"
    monkeypatch.setitem(server_module.OAUTH2_CONFIG["registry"], "success_redirect", target)

    login = _login("keycloak", _ACCOUNT_ID, _COLLIDING_PROFILE)

    assert login.status_code == 302, login.set_cookie_headers
    assert _login_session_cookie(login.set_cookie_headers, name)["domain"] == _EXAMPLE_DOMAIN
    assert login.location == target


def _shipped_config_with_session(
    monkeypatch: pytest.MonkeyPatch,
    **session_keys: Any,
) -> None:
    """Make ``load_oauth2_config`` read the shipped yml with ``session_keys`` in its session block."""
    real_safe_load = server_module.yaml.safe_load

    def _safe_load(stream: Any) -> Any:
        config = real_safe_load(stream)
        config["session"].update(session_keys)
        return config

    monkeypatch.setattr(server_module.yaml, "safe_load", _safe_load)


# (the yml session block's extra keys, every key the refusal must name with its
#  variable). The shipped block holds only domain: "${SESSION_COOKIE_DOMAIN}".
_IGNORED_SESSION_KEY_ROWS = [
    pytest.param({"secure": True}, {"secure": "SESSION_COOKIE_SECURE"}, id="secure"),
    pytest.param(
        {"max_age_seconds": 28800},
        {"max_age_seconds": "SESSION_MAX_AGE_SECONDS"},
        id="max-age",
    ),
    pytest.param(
        {"cookie_name": "mcp_oauth_session"},
        {"cookie_name": "SESSION_COOKIE_NAME"},
        id="cookie-name-as-formerly-shipped",
    ),
    pytest.param({"samesite": "lax"}, {"samesite": "SameSite=Lax"}, id="samesite"),
    pytest.param({"httponly": True}, {"httponly": "HttpOnly"}, id="httponly"),
    pytest.param({"domain": _YML_DOMAIN}, {"domain": "SESSION_COOKIE_DOMAIN"}, id="domain-edited"),
    pytest.param(
        {"domain": "${SOME_UNSET_VARIABLE}"},
        {"domain": "SESSION_COOKIE_DOMAIN"},
        id="domain-other-placeholder",
    ),
    pytest.param(
        {"secure": False, "cookie_name": _CUSTOM_NAME, "domain": _EXAMPLE_DOMAIN},
        {
            "secure": "SESSION_COOKIE_SECURE",
            "cookie_name": "SESSION_COOKIE_NAME",
            "domain": "SESSION_COOKIE_DOMAIN",
        },
        id="several-keys-in-one-message",
    ),
]


@pytest.mark.parametrize(("session_keys", "named"), _IGNORED_SESSION_KEY_ROWS)
@pytest.mark.parametrize("env_domain", [None, _EXAMPLE_DOMAIN], ids=["env-unset", "env-set"])
def test_startup_refuses_a_yml_session_key_the_auth_server_does_not_read(
    monkeypatch, session_keys, named, env_domain
):
    """A yml session block with any key besides the shipped domain stops the auth server.

    The settings are the one source of these values, so the yml value would be
    ignored without notice. Only an operator who mounts an edited copy of the
    yml can hit this: it ships in the image and no deployment surface mounts
    it. The one message names each key and the variable to set instead.
    """
    monkeypatch.delenv("SOME_UNSET_VARIABLE", raising=False)
    if env_domain is None:
        monkeypatch.delenv("SESSION_COOKIE_DOMAIN", raising=False)
    else:
        monkeypatch.setenv("SESSION_COOKIE_DOMAIN", env_domain)
    monkeypatch.setattr(server_module.settings, "session_cookie_domain", env_domain)
    _shipped_config_with_session(monkeypatch, **session_keys)

    with pytest.raises(ValueError) as refusal:
        server_module.load_oauth2_config()

    message = str(refusal.value)
    for key, variable in named.items():
        assert f"session.{key}" in message, message
        assert variable in message, message


@pytest.mark.parametrize(
    "session_keys",
    [
        pytest.param({}, id="as-shipped"),
        pytest.param({"domain": ""}, id="empty-domain"),
        pytest.param({"domain": None}, id="null-domain"),
    ],
)
@pytest.mark.parametrize(
    "env_domain", [None, "", _EXAMPLE_DOMAIN], ids=["env-unset", "env-empty", "env-set"]
)
def test_startup_accepts_the_shipped_yml_session_block(monkeypatch, session_keys, env_domain):
    """The shipped ``domain: "${SESSION_COOKIE_DOMAIN}"``, or an empty domain, loads.

    Whether SESSION_COOKIE_DOMAIN is set or not, and whatever it says: the
    placeholder expands to it, so it cannot disagree.
    """
    if env_domain is None:
        monkeypatch.delenv("SESSION_COOKIE_DOMAIN", raising=False)
    else:
        monkeypatch.setenv("SESSION_COOKIE_DOMAIN", env_domain)
    monkeypatch.setattr(server_module.settings, "session_cookie_domain", env_domain)
    _shipped_config_with_session(monkeypatch, **session_keys)

    assert "session" in server_module.load_oauth2_config()


_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
# Each service's startup import, as its container runs it.
_SERVICE_IMPORTS = [
    pytest.param("import server", id="auth-server"),
    pytest.param("import registry.main", id="registry"),
]


@pytest.mark.parametrize("service_import", _SERVICE_IMPORTS)
def test_both_services_refuse_to_start_with_the_prefix_and_a_domain(service_import):
    """With SESSION_COOKIE_HOST_PREFIX=true and a SESSION_COOKIE_DOMAIN, neither service starts.

    A browser drops a ``__Host-`` cookie that carries a Domain, so nobody
    could sign in. Each service is imported in a fresh interpreter, as its
    container does at startup.
    """
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(_REPO_ROOT), str(_REPO_ROOT / "auth_server")]),
        "SESSION_COOKIE_HOST_PREFIX": "true",
        "SESSION_COOKIE_SECURE": "true",
        "SESSION_COOKIE_DOMAIN": _EXAMPLE_DOMAIN,
    }

    result = subprocess.run(  # nosec B603 - fixed interpreter and fixed import statement
        [sys.executable, "-c", service_import],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0, result.stdout
    assert "SESSION_COOKIE_HOST_PREFIX=true needs" in result.stderr, result.stderr[-2000:]
    assert "SESSION_COOKIE_DOMAIN must be empty" in result.stderr, result.stderr[-2000:]
