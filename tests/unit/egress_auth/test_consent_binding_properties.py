"""Consent binding property: a callback stores a token only where the consent began.

The egress OAuth callback writes the upstream token into the vault bucket of the
principal named in the signed consent state. That write is safe only when the
browser completing the callback is the browser that started the consent, and
that browser is authenticated as that same principal. Without the principal
check, a consent link minted by one user and opened by another stores the
second user's upstream token in the first user's bucket. Without the browser
check, anyone holding a state bound to user V (an agent with V's bearer, or V's
browser history) finishes the provider step with their own account and makes
V's browser load the callback, so V's vault holds the attacker's account.

These tests drive the real ``/oauth2/egress/callback``, ``/egress-auth/initiate``
and ``/oauth2/egress/connect`` routes and the real ``EgressAuthService``. Each
``TestClient`` is one browser with its own cookie jar.
"""

import time
from datetime import (
    UTC,
    datetime,
)
from http.cookies import SimpleCookie
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import AsyncMock
from urllib.parse import (
    parse_qs,
    urlparse,
)

import httpx
import jwt as pyjwt
import pytest
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.testclient import TestClient
from pymongo.errors import ServerSelectionTimeoutError

import registry.api.egress_auth_routes as routes
import registry.api.egress_oauth_facade_routes as facade
from registry.egress_auth import (
    oauth_engine,
    state_codec,
)
from registry.egress_auth.factory import _MongoReplayGuard
from registry.egress_auth.providers import resolve_provider
from registry.egress_auth.schemas import OAuthState
from registry.egress_auth.service import EgressAuthService
from registry.secrets.interfaces import SecretStoreBase
from registry.utils.credential_encryption import encrypt_credential

_CONSENTING_USER = "alice"
_OTHER_USER = "mallory"
_AUTH_METHOD = "oauth2"
_SERVER_PATH = "/github-mcp"
_UPSTREAM = "https://api.example.com/mcp"
# The TestClient origin. The callback base carries the /api prefix the test app
# mounts the router under, so the browser-visible callback path the consent
# cookie is scoped to is the path these tests request.
_CALLBACK_BASE_URL = "http://testserver/api"
_CALLBACK_PATH = "/api/oauth2/egress/callback"
_CONSENTING_COOKIE = "session-of-alice"
_OTHER_COOKIE = "session-of-mallory"
_EXPIRED_COOKIE = "session-expired"
_NO_PRINCIPAL_COOKIE = "session-of-static-key"
# The registry's SECRET_KEY in these tests: it signs a real X-Internal-Token-Registry.
_REGISTRY_SECRET = "consent-binding-test-secret-0123456789abcdef"
_LIVE_SESSIONS: dict[str, dict] = {
    _CONSENTING_COOKIE: {
        "username": _CONSENTING_USER,
        "egress_user": _CONSENTING_USER,
        "auth_method": _AUTH_METHOD,
        "client_id": "web",
        "session_id": "sid-alice",
    },
    _OTHER_COOKIE: {
        "username": _OTHER_USER,
        "egress_user": _OTHER_USER,
        "auth_method": _AUTH_METHOD,
        "client_id": "web",
        "session_id": "sid-mallory",
    },
    # A live session that owns no per-user bucket: vault_principal is None.
    _NO_PRINCIPAL_COOKIE: {
        "username": _CONSENTING_USER,
        "egress_user": _CONSENTING_USER,
        "auth_method": "network-trusted",
        "client_id": "web",
        "session_id": "sid-static",
    },
}
# Cookies for which the callback refuses before the code exchange: no cookie, one
# that no longer resolves, and one that resolves to a caller with no principal.
_REFUSING_COOKIES = [
    pytest.param(None, id="no-session-cookie"),
    pytest.param(_EXPIRED_COOKIE, id="expired-session-cookie"),
    pytest.param(_NO_PRINCIPAL_COOKIE, id="session-without-principal"),
]
# The two routes a browser starts a consent at.
_CONSENT_STARTS = ["initiate", "connect"]


class _BrowserState(NamedTuple):
    """How the browser that completes the provider redirect looks to the registry.

    Attributes:
        cookie: Session cookie the browser presents, or None for no cookie.
        same_browser: True when it is the browser that started the consent, so it
            also carries the consent cookie that browser was given at consent start.
        live_principal: The principal the cookie resolves to, or None when the
            cookie is missing or no longer resolves.
    """

    cookie: str | None
    same_browser: bool
    live_principal: str | None


_BROWSER_STATES = [
    pytest.param(
        _BrowserState(cookie=None, same_browser=False, live_principal=None),
        id="no-session-cookie",
    ),
    pytest.param(
        _BrowserState(cookie=_EXPIRED_COOKIE, same_browser=False, live_principal=None),
        id="expired-session-cookie",
    ),
    pytest.param(
        _BrowserState(
            cookie=_CONSENTING_COOKIE, same_browser=True, live_principal=_CONSENTING_USER
        ),
        id="live-as-consenting-user",
    ),
    pytest.param(
        _BrowserState(
            cookie=_CONSENTING_COOKIE, same_browser=False, live_principal=_CONSENTING_USER
        ),
        id="other-browser-live-as-consenting-user",
    ),
    pytest.param(
        _BrowserState(cookie=_OTHER_COOKIE, same_browser=False, live_principal=_OTHER_USER),
        id="live-as-other-user",
    ),
]


class _RecordingStore(SecretStoreBase):
    """In-memory SecretStore that records every token write."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []
        self._data: dict[tuple[str, str, str, str, str], object] = {}

    async def put_token(self, auth_method, user_id, provider, server_path, token, *, purpose):
        self.writes.append((auth_method, user_id))
        self._data[(purpose, auth_method, user_id, provider, server_path)] = token

    async def get_token(self, auth_method, user_id, provider, server_path, *, purpose):
        return self._data.get((purpose, auth_method, user_id, provider, server_path))

    async def delete_token(self, auth_method, user_id, provider, server_path, *, purpose):
        self._data.pop((purpose, auth_method, user_id, provider, server_path), None)

    async def list_for_user(self, auth_method, user_id):
        return []


class _NonceRepo:
    """Stand-in for the operational repository's nonce collection.

    Behaves like ``consume_nonce`` on a reachable database until ``down`` is
    set, then raises the error pymongo raises when no server answers.
    """

    def __init__(self) -> None:
        self.down = False
        self._seen: set[str] = set()

    async def consume_nonce(self, nonce: str, ttl_seconds: int) -> bool:
        if self.down:
            raise ServerSelectionTimeoutError("no servers available")
        if nonce in self._seen:
            return False
        self._seen.add(nonce)
        return True


async def _resolve_session(
    request: Request,
    session: str | None = None,
) -> dict:
    """Stand-in for ``nginx_proxied_auth`` that resolves a session cookie.

    Like the real dependency called directly, it reads only ``session``: a
    route that calls it passes the cookie it read, or None for no session.

    Args:
        request: The incoming request.
        session: Session cookie value passed explicitly, as the routes do.

    Returns:
        A copy of the user context for a live session cookie.

    Raises:
        HTTPException: 401 for a missing, unknown or expired cookie, matching the
            real dependency.
    """
    if session in _LIVE_SESSIONS:
        return dict(_LIVE_SESSIONS[session])
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Session expired. Please login again via OAuth2.",
    )


# Bearer callers: the registry resolves them from the internal token /validate
# minted for the bearer, so the context carries no session id.
_ALICE_BEARER = "bearer-of-alice"
_BEARERS: dict[str, dict] = {
    _ALICE_BEARER: {
        "username": _CONSENTING_USER,
        "egress_user": _CONSENTING_USER,
        "auth_method": _AUTH_METHOD,
        "client_id": "dcr-agent",
        "session_id": None,
    },
}


async def _session_dependency(
    request: Request,
) -> dict:
    """FastAPI dependency form of ``nginx_proxied_auth`` for the initiate route.

    A session cookie wins over a bearer, as at ``/validate``, which checks the
    cookie first; a request without one is resolved from its bearer.

    Args:
        request: The incoming request.

    Returns:
        The user context for the request's session cookie, else its bearer.
    """
    authorization = request.headers.get("authorization", "")
    cookie = request.cookies.get(routes.settings.session_cookie_name)
    if authorization.startswith("Bearer ") and cookie is None:
        return dict(_BEARERS[authorization.removeprefix("Bearer ")])
    return await _resolve_session(request, cookie)


async def _resolve_session_from_cookie(
    cookie_value: str | None,
) -> dict | None:
    """Stand-in for the registry's session-store lookup of a session cookie.

    The routes resolve the browser's session through the registry's real
    ``enhanced_auth``, which reads the record this returns.

    Args:
        cookie_value: The session cookie the request carried.

    Returns:
        A copy of the session record for a live cookie, else None.
    """
    if cookie_value in _LIVE_SESSIONS:
        return {"provider": "github", "groups": [], **_LIVE_SESSIONS[cookie_value]}
    return None


def _registry_token(
    username: str,
    secret: str,
) -> str:
    """Return an ``X-Internal-Token-Registry`` for a bearer caller with no session.

    Args:
        username: The caller the token names (its ``sub`` and ``egress_user``).
        secret: The key it is signed with; the registry's ``SECRET_KEY`` makes
            it a token ``nginx_proxied_auth`` accepts.

    Returns:
        The signed token.
    """
    now = int(time.time())
    claims = {
        "iss": "mcp-auth-server",
        "aud": "mcp-registry-ui",
        "token_use": "mcp-registry-ui",
        "sub": username,
        "egress_user": username,
        "auth_method": _AUTH_METHOD,
        "client_id": "dcr-agent",
        "groups": [],
        "scopes": [],
        "iat": now,
        "exp": now + 60,
    }
    return pyjwt.encode(claims, secret, algorithm="HS256")


async def _fake_post_token(
    cfg,
    data,
    headers,
) -> dict:
    """Provider token endpoint stub: every code exchange succeeds.

    Args:
        cfg: Provider config (unused).
        data: Form body (unused).
        headers: Request headers (unused).

    Returns:
        A token endpoint response body.
    """
    return {
        "access_token": "gho_upstream_token_of_whoever_consented",
        "refresh_token": "rt",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": "repo",
    }


def _browser(
    app: FastAPI,
    cookie: str | None,
) -> TestClient:
    """Return a TestClient acting as one browser with its own cookie jar.

    Args:
        app: The app under test.
        cookie: Session cookie to hold, or None for none.

    Returns:
        A TestClient whose jar holds only that session cookie.
    """
    cookies = {routes.settings.session_cookie_name: cookie} if cookie is not None else None
    return TestClient(app, cookies=cookies)


def _extract_state(
    authorize_url: str,
) -> str:
    """Return the ``state`` query parameter of a provider authorize URL.

    Args:
        authorize_url: URL a consent start sent the browser to.

    Returns:
        The encoded consent state.
    """
    return parse_qs(urlparse(authorize_url).query)["state"][0]


def _start_consent(
    browser: TestClient,
    start: str,
) -> tuple[str, httpx.Response]:
    """Start a consent in ``browser`` and return its state and the start response.

    Args:
        browser: The browser that starts the consent.
        start: ``initiate`` (the Connected Accounts XHR) or ``connect`` (the
            elicitation front door, which 302s to the provider).

    Returns:
        The encoded consent state and the response that started it.
    """
    if start == "initiate":
        response = browser.post("/api/egress-auth/initiate", json={"server_path": _SERVER_PATH})
        assert response.status_code == 200, response.text
        return _extract_state(response.json()["authorize_url"]), response
    response = browser.get(
        "/oauth2/egress/connect", params={"server": _SERVER_PATH}, follow_redirects=False
    )
    assert response.status_code == 302, response.text
    return _extract_state(response.headers["location"]), response


def _complete(
    browser: TestClient,
    state_blob: str,
    code: str = "provider-code",
) -> httpx.Response:
    """Load the provider redirect to the callback in ``browser``.

    Args:
        browser: The browser the provider redirect lands in.
        state_blob: The consent state the provider echoes back.
        code: The authorization code the provider issued.

    Returns:
        The callback response.
    """
    return browser.get(_CALLBACK_PATH, params={"code": code, "state": state_blob})


def _consent_cookies(
    browser: TestClient,
) -> dict[str, str]:
    """Return every cookie in ``browser``'s jar other than the session cookie.

    Args:
        browser: The browser whose jar to read.

    Returns:
        Cookie name to value.
    """
    return {
        cookie.name: cookie.value
        for cookie in browser.cookies.jar
        if cookie.name != routes.settings.session_cookie_name
    }


@pytest.fixture(autouse=True)
def _reset_cipher():
    state_codec.reset_cipher_for_tests()
    yield
    state_codec.reset_cipher_for_tests()


@pytest.fixture
def nonce_repo():
    """The replay store behind the consent app's replay guard."""
    return _NonceRepo()


@pytest.fixture
def consent_app(monkeypatch, nonce_repo):
    """App with the real egress routes and service; IdP, provider and DB stubbed.

    Returns:
        Tuple of (app, recording store).
    """
    original_auth_dependency = routes.nginx_proxied_auth
    egress_oauth = {
        "provider": "github",
        "client_id": "Iv1.testclient",
        "client_secret_encrypted": encrypt_credential("ghs_testsecret"),
        "scopes": ["repo"],
    }
    server = {
        "path": _SERVER_PATH,
        "proxy_pass_url": _UPSTREAM,
        "egress_auth_mode": "oauth_user",
        "egress_oauth": egress_oauth,
    }
    store = _RecordingStore()
    svc = EgressAuthService(
        secret_store=store,
        callback_base_url=_CALLBACK_BASE_URL,
        replay_guard=_MongoReplayGuard(nonce_repo),
    )

    monkeypatch.setattr(routes.settings, "egress_auth_enabled", True)
    monkeypatch.setattr(routes.settings, "egress_oauth_callback_base_url", _CALLBACK_BASE_URL)
    monkeypatch.setattr(routes.server_service, "get_server_info", AsyncMock(return_value=server))
    monkeypatch.setattr(routes, "get_egress_auth_service", lambda: svc)
    monkeypatch.setattr(facade, "get_egress_auth_service", lambda: svc)
    # The routes resolve the browser's session themselves, from the cookie only,
    # through the registry's real enhanced_auth; only the stores behind it are
    # stubbed.
    monkeypatch.setattr(oauth_engine, "_post_token", _fake_post_token)
    monkeypatch.setattr(
        "registry.auth.dependencies.resolve_session_from_cookie", _resolve_session_from_cookie
    )
    monkeypatch.setattr(
        "registry.auth.dependencies.map_cognito_groups_to_scopes", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        "registry.auth.dependencies.resolve_scope_access",
        AsyncMock(return_value=SimpleNamespace(servers=[], tools={})),
    )
    monkeypatch.setattr(
        "registry.auth.dependencies.get_ui_permissions_for_user", AsyncMock(return_value={})
    )
    monkeypatch.setenv("SECRET_KEY", _REGISTRY_SECRET)

    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.include_router(facade.router)
    app.dependency_overrides[original_auth_dependency] = _session_dependency
    app.dependency_overrides[routes.verify_csrf_token_flexible] = lambda: None
    return app, store


@pytest.mark.unit
@pytest.mark.parametrize("start", _CONSENT_STARTS)
@pytest.mark.parametrize("browser_state", _BROWSER_STATES)
def test_callback_writes_only_in_the_starting_browser_live_as_its_principal(
    consent_app, start, browser_state
):
    """Property: the callback writes P's bucket iff it runs in P's starting browser, live as P.

    The consenting user starts consent from their own browser; the provider
    redirect is then completed by a browser in each possible session state. A
    vault write may happen only when that browser is the one that started the
    consent and is authenticated as the principal bound in the consent state,
    and then only into that principal's bucket.
    """
    app, store = consent_app
    consenting_browser = _browser(app, _CONSENTING_COOKIE)
    state_blob, _ = _start_consent(consenting_browser, start)

    if browser_state.same_browser:
        completing_browser = consenting_browser
    else:
        completing_browser = _browser(app, browser_state.cookie)
    _complete(completing_browser, state_blob)

    may_write = browser_state.same_browser and browser_state.live_principal == _CONSENTING_USER
    expected_writes = [(_AUTH_METHOD, _CONSENTING_USER)] if may_write else []
    assert store.writes == expected_writes, (
        f"consent started at {start} for {_CONSENTING_USER!r}, completing browser "
        f"{browser_state}: vault writes {store.writes}, expected {expected_writes}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("start", _CONSENT_STARTS)
def test_account_injection_into_the_principals_browser_writes_nothing(consent_app, start):
    """An attacker's code paired with V's state, loaded in V's browser, writes nothing.

    The attacker holds V's session cookie outside V's browser (a bearer cannot
    start a consent at all), so the consent it starts is bound to V. It completes the provider
    step with its own account and makes V's live browser load the callback with
    its code. V's browser never started that consent, so it holds no consent
    cookie for it, and the callback must refuse before the code exchange.
    """
    app, store = consent_app
    attacker_client = _browser(app, _CONSENTING_COOKIE)
    state_blob, _ = _start_consent(attacker_client, start)
    victims_browser = _browser(app, _CONSENTING_COOKIE)

    refused = _complete(victims_browser, state_blob, code="code-for-the-attackers-account")

    assert refused.status_code == 400, refused.text
    assert store.writes == [], (
        f"V's browser completed a consent another client started; vault writes {store.writes}"
    )


@pytest.mark.unit
def test_consent_cookie_completes_only_its_own_consent(consent_app):
    """The cookie one consent set never completes a different consent.

    One browser holds the cookie of consent 1. It cannot complete consent 2 (or
    3), which another client started: neither as it is, nor with consent 1's
    cookie value presented under the name the other consent's cookie has.
    """
    app, store = consent_app
    browser = _browser(app, _CONSENTING_COOKIE)
    _, first = _start_consent(browser, "initiate")
    [first_value] = first.cookies.values()
    other_client = _browser(app, _CONSENTING_COOKIE)
    second_state, _ = _start_consent(other_client, "initiate")
    third_state, third = _start_consent(other_client, "initiate")
    [third_name] = third.cookies.keys()

    as_is = _complete(browser, second_state)
    browser.cookies.set(third_name, first_value, path=_CALLBACK_PATH)
    renamed = _complete(browser, third_state)

    assert (as_is.status_code, renamed.status_code) == (400, 400), (as_is.text, renamed.text)
    assert store.writes == [], f"consent 1's cookie completed another consent: {store.writes}"


@pytest.mark.unit
@pytest.mark.parametrize("start", _CONSENT_STARTS)
def test_parallel_consents_in_one_browser_both_complete(consent_app, start):
    """Two consents started in one browser both complete, in either order.

    Each consent sets its own cookie, so starting the second does not displace
    the first. Each callback clears the cookie of its own consent.
    """
    app, store = consent_app
    browser = _browser(app, _CONSENTING_COOKIE)
    first_state, _ = _start_consent(browser, start)
    second_state, _ = _start_consent(browser, start)
    assert len(_consent_cookies(browser)) == 2, _consent_cookies(browser)

    second = _complete(browser, second_state)
    first = _complete(browser, first_state)

    assert (first.status_code, second.status_code) == (200, 200), (first.text, second.text)
    assert store.writes == [(_AUTH_METHOD, _CONSENTING_USER)] * 2
    assert _consent_cookies(browser) == {}


@pytest.mark.unit
def test_state_without_a_browser_binding_is_refused(consent_app):
    """A consent state that binds no browser (minted before the upgrade) never writes.

    The state carries everything else a consent state needs and the browser is
    live as its principal; only the browser binding is missing. The refusal
    also burns the state.
    """
    app, store = consent_app
    unbound = state_codec.encode_state(
        OAuthState(
            user_id=_CONSENTING_USER,
            auth_method=_AUTH_METHOD,
            provider="github",
            server_path=_SERVER_PATH,
            nonce="pre-upgrade-nonce",
            issued_at=datetime.now(UTC).isoformat(),
            approved_upstreams=[_UPSTREAM],
            token_url=resolve_provider({"provider": "github"}).token_url,
        )
    )
    browser = _browser(app, _CONSENTING_COOKIE)

    refused = _complete(browser, unbound)

    assert refused.status_code == 400, refused.text
    assert store.writes == []


@pytest.mark.unit
@pytest.mark.parametrize("refusing_cookie", _REFUSING_COOKIES)
def test_refused_callback_burns_the_state(consent_app, refusing_cookie):
    """A code and state the callback refused cannot be replayed by the initiator.

    The refusing browser's history keeps ``?code=...&state=...``. If that URL is
    handed back to the user who minted the state, their own live browser passes
    the principal and browser checks, so the refusal must already have consumed
    the nonce.
    """
    app, store = consent_app
    consenting_browser = _browser(app, _CONSENTING_COOKIE)
    state_blob, _ = _start_consent(consenting_browser, "initiate")
    refused = _complete(_browser(app, refusing_cookie), state_blob, "code-of-the-refusing-browser")
    assert refused.status_code == 401, refused.text

    _complete(consenting_browser, state_blob, "code-of-the-refusing-browser")

    assert store.writes == [], (
        f"callback refused for cookie {refusing_cookie!r}, then the same code and "
        f"state replayed from {_CONSENTING_USER!r}'s browser wrote {store.writes}"
    )


@pytest.mark.unit
def test_browser_mismatch_burns_the_state(consent_app):
    """A callback refused for the wrong browser cannot be replayed in the right one.

    V's own browser starts the consent; the code and state are first loaded in
    another browser live as V, which is refused for the missing consent cookie.
    Replaying the same URL in V's starting browser must not write either.
    """
    app, store = consent_app
    consenting_browser = _browser(app, _CONSENTING_COOKIE)
    state_blob, _ = _start_consent(consenting_browser, "initiate")
    refused = _complete(_browser(app, _CONSENTING_COOKIE), state_blob)
    assert refused.status_code == 400, refused.text

    _complete(consenting_browser, state_blob)

    assert store.writes == []


@pytest.mark.unit
@pytest.mark.parametrize("start", _CONSENT_STARTS)
def test_provider_error_without_code_burns_the_state_and_clears_its_cookie(consent_app, start):
    """A provider redirect with no code (the user clicked Cancel) consumes the consent.

    The provider answers ``?error=access_denied&state=S``. The callback burns S
    and expires the consent cookie, so a later ``?code=C&state=S`` loaded in the
    starting browser, even with that browser's consent cookie put back, writes
    nothing. Otherwise anyone who learned S could finish the provider step with
    their own account and send the starting browser to the callback.
    """
    app, store = consent_app
    browser = _browser(app, _CONSENTING_COOKIE)
    state_blob, started = _start_consent(browser, start)
    [(cookie_name, cookie_value)] = started.cookies.items()

    cancelled = browser.get(_CALLBACK_PATH, params={"error": "access_denied", "state": state_blob})
    cleared = _consent_cookies(browser)
    browser.cookies.set(cookie_name, cookie_value, path=_CALLBACK_PATH)
    _complete(browser, state_blob, code="code-for-the-attackers-account")

    assert cancelled.status_code == 400, cancelled.text
    assert cleared == {}, f"consent cookie survived a callback with no code: {cleared}"
    assert store.writes == [], f"state replayed after a provider error wrote {store.writes}"


@pytest.mark.unit
def test_initiate_without_the_browser_session_hands_out_no_binding(consent_app):
    """A bearer caller with no session cookie gets no consent state and no cookie.

    A client holding V's bearer (a DCR agent) skips CSRF, which applies only to
    cookie callers. If initiate answered it, the client would hold a state bound
    to V together with the matching consent secret, and could inject its own
    account wherever it can plant a cookie in V's browser. The context the route
    acts on must be the session the request's cookie names; otherwise 403.
    """
    app, _ = consent_app
    client = _browser(app, None)

    response = client.post(
        "/api/egress-auth/initiate",
        json={"server_path": _SERVER_PATH},
        headers={"Authorization": f"Bearer {_ALICE_BEARER}"},
    )

    assert response.status_code == 403, response.text
    assert "authorize_url" not in response.text
    assert dict(response.cookies) == {}, response.headers.get_list("set-cookie")


@pytest.mark.unit
@pytest.mark.parametrize(
    "token_secret",
    [
        pytest.param(_REGISTRY_SECRET, id="real-registry-token"),
        pytest.param("not-the-registry-secret-0123456789abcdef", id="forged-registry-token"),
    ],
)
def test_registry_token_is_never_a_browser_session(consent_app, token_secret):
    """An ``X-Internal-Token-Registry`` names no browser on any consent route.

    A client sends a registry token naming mallory, real or forged, and no
    session cookie: initiate refuses it, connect sends it to the login, and a
    callback it loads (holding the consent cookie of alice's consent) is
    refused as signed out. With alice's real session cookie beside the same
    token, every route acts for alice, so the token never chooses the browser
    principal.
    """
    app, store = consent_app
    headers = {"X-Internal-Token-Registry": _registry_token(_OTHER_USER, token_secret)}
    token_only = TestClient(app, headers=headers)
    alice = _browser(app, _CONSENTING_COOKIE)
    state_blob, started = _start_consent(alice, "initiate")
    for name, value in started.cookies.items():
        token_only.cookies.set(name, value, path=_CALLBACK_PATH)

    initiate = token_only.post("/api/egress-auth/initiate", json={"server_path": _SERVER_PATH})
    connect = token_only.get(
        "/oauth2/egress/connect", params={"server": _SERVER_PATH}, follow_redirects=False
    )
    callback = _complete(token_only, state_blob)

    assert initiate.status_code == 403, initiate.text
    assert connect.status_code == 302, connect.text
    assert "/oauth2/login/" in connect.headers["location"], connect.headers["location"]
    assert callback.status_code == 401, callback.text
    assert store.writes == []

    alice.headers.update(headers)
    state_blob, _ = _start_consent(alice, "connect")
    connected = _complete(alice, state_blob)

    assert connected.status_code == 200, connected.text
    assert store.writes == [(_AUTH_METHOD, _CONSENTING_USER)]


def _consent_set_cookie(
    response: httpx.Response,
) -> SimpleCookie:
    """Parse the one consent ``Set-Cookie`` a consent start returned.

    Args:
        response: The initiate or connect response.

    Returns:
        The parsed cookie (one morsel).
    """
    [header] = response.headers.get_list("set-cookie")
    parsed = SimpleCookie()
    parsed.load(header)
    return parsed


@pytest.mark.unit
@pytest.mark.parametrize("start", _CONSENT_STARTS)
@pytest.mark.parametrize(
    ("base_url", "forwarded_proto", "secure"),
    [
        pytest.param("https://gw.example.com", None, True, id="https"),
        pytest.param("http://gw.example.com", "https", True, id="tls-ends-at-the-proxy"),
        pytest.param("http://gw.example.com", None, False, id="plain-http"),
    ],
)
def test_consent_cookie_attributes(
    consent_app, monkeypatch, start, base_url, forwarded_proto, secure
):
    """The consent cookie reaches the callback, from a cross-site redirect, and only it.

    With a callback base under a path (``https://gw.example.com/registry``) and
    a shared ``SESSION_COOKIE_DOMAIN``, the cookie is HttpOnly, SameSite=Lax
    (the provider's redirect back is a top-level cross-site GET), scoped to the
    callback's full path and the session cookie's domain, lives for the state
    TTL, and is Secure exactly when the browser reached the gateway over HTTPS.
    """
    app, _ = consent_app
    monkeypatch.setattr(
        routes.settings, "egress_oauth_callback_base_url", "https://gw.example.com/registry"
    )
    monkeypatch.setattr(routes.settings, "session_cookie_domain", ".example.com")
    monkeypatch.setattr(routes.settings, "session_cookie_secure", True)
    headers = {"X-Forwarded-Proto": forwarded_proto} if forwarded_proto else {}
    browser = TestClient(
        app,
        base_url=base_url,
        headers=headers,
        cookies={routes.settings.session_cookie_name: _CONSENTING_COOKIE},
    )

    _, started = _start_consent(browser, start)

    [(name, morsel)] = _consent_set_cookie(started).items()
    assert name.startswith("egress_consent_"), name
    assert morsel.value
    assert morsel["httponly"] is True
    assert morsel["samesite"].lower() == "lax"
    assert morsel["path"] == "/registry/oauth2/egress/callback"
    assert morsel["max-age"] == str(routes.settings.egress_state_ttl_seconds)
    assert morsel["domain"] == ".example.com"
    assert bool(morsel["secure"]) is secure


@pytest.mark.unit
@pytest.mark.parametrize(
    "completing_session",
    [
        pytest.param(None, id="refused-before-the-service-no-session"),
        pytest.param(_OTHER_COOKIE, id="refused-by-the-service-other-principal"),
        pytest.param(_CONSENTING_COOKIE, id="connected"),
    ],
)
def test_callback_expires_the_consent_cookie_whatever_the_outcome(consent_app, completing_session):
    """Every callback response for a decoded state expires that consent's cookie.

    The starting browser completes its own consent after its session changed:
    signed out (refused before the service runs), signed in as someone else
    (refused by the service), or unchanged (connected). In each case the
    response deletes the cookie by its name and path.
    """
    app, _ = consent_app
    browser = _browser(app, _CONSENTING_COOKIE)
    state_blob, started = _start_consent(browser, "initiate")
    [cookie_name] = started.cookies.keys()
    browser.cookies.delete(routes.settings.session_cookie_name)
    if completing_session is not None:
        browser.cookies.set(routes.settings.session_cookie_name, completing_session)

    response = _complete(browser, state_blob)

    [(name, morsel)] = _consent_set_cookie(response).items()
    assert name == cookie_name
    assert morsel["max-age"] == "0"
    assert morsel["path"] == _CALLBACK_PATH
    assert _consent_cookies(browser) == {}


# Callbacks that reach the replay store: one per refusal that burns the state,
# plus the completion in the starting browser. Each is (query without the
# state, session cookie the browser holds at the callback, server registration
# the callback reads).
_REPLAY_STORE_CALLBACKS = [
    pytest.param({"code": "provider-code"}, _CONSENTING_COOKIE, "configured", id="completion"),
    pytest.param({"error": "access_denied"}, _CONSENTING_COOKIE, "configured", id="no-code"),
    pytest.param({"code": "provider-code"}, None, "configured", id="no-session-cookie"),
    pytest.param({"code": "provider-code"}, _EXPIRED_COOKIE, "configured", id="expired-session"),
    pytest.param(
        {"code": "provider-code"}, _NO_PRINCIPAL_COOKIE, "configured", id="session-no-principal"
    ),
    pytest.param({"code": "provider-code"}, _CONSENTING_COOKIE, "gone", id="server-gone"),
    pytest.param({"code": "provider-code"}, _CONSENTING_COOKIE, "no-oauth", id="server-no-oauth"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("query", "session", "server_state"), _REPLAY_STORE_CALLBACKS)
@pytest.mark.parametrize("same_browser", [True, False], ids=["starting-browser", "other-browser"])
def test_callback_decodes_and_consumes_its_state_once(
    consent_app, monkeypatch, nonce_repo, query, session, server_state, same_browser
):
    """Every callback decodes its state once and consumes its nonce once.

    Whatever the outcome (connected, no code, no or unusable session, a server
    gone or unconfigured, the wrong browser), the callback reads the state with
    one decode and records its nonce with one replay-store call, before any
    refusal.
    """
    app, _ = consent_app
    starting_browser = _browser(app, _CONSENTING_COOKIE)
    state_blob, _ = _start_consent(starting_browser, "initiate")
    browser = starting_browser if same_browser else _browser(app, None)
    browser.cookies.delete(routes.settings.session_cookie_name)
    if session is not None:
        browser.cookies.set(routes.settings.session_cookie_name, session)
    configured = routes.server_service.get_server_info.return_value
    if server_state == "gone":
        monkeypatch.setattr(routes.server_service, "get_server_info", AsyncMock(return_value=None))
    elif server_state == "no-oauth":
        unconfigured = {k: v for k, v in configured.items() if k != "egress_oauth"}
        monkeypatch.setattr(
            routes.server_service, "get_server_info", AsyncMock(return_value=unconfigured)
        )
    nonce = state_codec.decode_state(state_blob).nonce
    decrypts: list[bytes] = []
    consumed: list[str] = []
    cipher = state_codec._get_aesgcm()
    real_consume = nonce_repo.consume_nonce

    class _CountingCipher:
        """The state cipher, counting decrypts: every decode of a state is one."""

        def decrypt(self, iv: bytes, data: bytes, associated_data: bytes | None) -> bytes:
            decrypts.append(data)
            return cipher.decrypt(iv, data, associated_data)

    async def _counting_consume(recorded: str, ttl_seconds: int) -> bool:
        consumed.append(recorded)
        return await real_consume(recorded, ttl_seconds)

    monkeypatch.setattr(state_codec, "_get_aesgcm", _CountingCipher)
    monkeypatch.setattr(nonce_repo, "consume_nonce", _counting_consume)

    browser.get(_CALLBACK_PATH, params={**query, "state": state_blob})

    assert len(decrypts) == 1, f"state decoded {len(decrypts)} times"
    assert consumed == [nonce]


def _callback_with_replay_store_down(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    nonce_repo: _NonceRepo,
    query: dict[str, str],
    session: str | None,
    server_state: str,
) -> tuple[TestClient, str, str, httpx.Response]:
    """Start a consent, then load its callback while the replay store is down.

    Args:
        app: The app under test.
        monkeypatch: Patches the server registration the callback reads.
        nonce_repo: The replay store; set down before the callback.
        query: Callback query parameters other than ``state``.
        session: Session cookie the starting browser holds at the callback.
        server_state: ``configured``, ``gone`` or ``no-oauth``.

    Returns:
        The starting browser, the consent state, its cookie name, and the
        callback response.
    """
    browser = TestClient(
        app,
        cookies={routes.settings.session_cookie_name: _CONSENTING_COOKIE},
        raise_server_exceptions=False,
    )
    state_blob, started = _start_consent(browser, "initiate")
    [cookie_name] = started.cookies.keys()
    browser.cookies.delete(routes.settings.session_cookie_name)
    if session is not None:
        browser.cookies.set(routes.settings.session_cookie_name, session)
    configured = routes.server_service.get_server_info.return_value
    if server_state == "gone":
        monkeypatch.setattr(routes.server_service, "get_server_info", AsyncMock(return_value=None))
    elif server_state == "no-oauth":
        unconfigured = {k: v for k, v in configured.items() if k != "egress_oauth"}
        monkeypatch.setattr(
            routes.server_service, "get_server_info", AsyncMock(return_value=unconfigured)
        )
    nonce_repo.down = True

    response = browser.get(_CALLBACK_PATH, params={**query, "state": state_blob})

    nonce_repo.down = False
    monkeypatch.setattr(
        routes.server_service, "get_server_info", AsyncMock(return_value=configured)
    )
    return browser, state_blob, cookie_name, response


def _record_code_exchanges(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict]:
    """Record every request the code exchange sends to the provider token endpoint.

    Args:
        monkeypatch: Replaces the token endpoint stub.

    Returns:
        The list each exchange's form body is appended to.
    """
    exchanges: list[dict] = []

    async def _recording_post_token(cfg, data, headers) -> dict:
        exchanges.append(data)
        return await _fake_post_token(cfg, data, headers)

    monkeypatch.setattr(oauth_engine, "_post_token", _recording_post_token)
    return exchanges


@pytest.mark.unit
@pytest.mark.parametrize(("query", "session", "server_state"), _REPLAY_STORE_CALLBACKS)
def test_replay_store_outage_refuses_the_callback_and_expires_its_cookie(
    consent_app, monkeypatch, nonce_repo, query, session, server_state
):
    """A callback whose nonce the replay store cannot record stores nothing.

    The page is a 503 asking the user to connect again, the consent cookie is
    expired on the callback path, and no code reaches the provider.
    """
    app, store = consent_app
    exchanges = _record_code_exchanges(monkeypatch)

    _, _, cookie_name, response = _callback_with_replay_store_down(
        app, monkeypatch, nonce_repo, query, session, server_state
    )

    assert response.status_code == 503, response.text
    assert "temporarily unavailable" in response.text
    [(name, morsel)] = _consent_set_cookie(response).items()
    assert (name, morsel["max-age"], morsel["path"]) == (cookie_name, "0", _CALLBACK_PATH)
    assert exchanges == []
    assert store.writes == []


@pytest.mark.unit
@pytest.mark.parametrize(("query", "session", "server_state"), _REPLAY_STORE_CALLBACKS)
def test_replay_after_a_replay_store_outage_writes_nothing(
    consent_app, monkeypatch, nonce_repo, query, session, server_state
):
    """The state the outage left unburned cannot complete once the store is back.

    The starting browser signs in again as the consenting user and replays the
    same code and state. Its consent cookie is gone, so the replay is refused
    before the code exchange.
    """
    app, store = consent_app
    browser, state_blob, _, _ = _callback_with_replay_store_down(
        app, monkeypatch, nonce_repo, query, session, server_state
    )
    browser.cookies.delete(routes.settings.session_cookie_name)
    browser.cookies.set(routes.settings.session_cookie_name, _CONSENTING_COOKIE)
    exchanges = _record_code_exchanges(monkeypatch)

    replay = _complete(browser, state_blob)

    assert replay.status_code == 400, replay.text
    assert exchanges == []
    assert store.writes == [], f"replay after a replay store outage wrote {store.writes}"


async def _raise_unexpected(*args, **kwargs) -> None:
    """Stand-in for a callback dependency that fails with an unhandled error.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError("unexpected failure")


# Points in the callback that raise an error the route does not handle. Each
# names the attribute patched to raise: the session resolution and server lookup
# run before the nonce is consumed; the completion raises from handle_callback
# before the code exchange; the replay store raising a non-database error also
# makes the route's own attempt to burn the state fail.
_UNEXPECTED_ERROR_POINTS = [
    pytest.param("session-resolution", id="session-resolution"),
    pytest.param("server-lookup", id="server-lookup"),
    pytest.param("completion", id="completion"),
    pytest.param("replay-store", id="replay-store-and-burn"),
]


def _callback_with_unexpected_error(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    nonce_repo: _NonceRepo,
    point: str,
) -> tuple[TestClient, str, str, httpx.Response]:
    """Start a consent, then load its callback while ``point`` raises RuntimeError.

    The patch is undone before returning, so a replay sees a working registry.

    Args:
        app: The app under test.
        monkeypatch: Source of the scoped patch context.
        nonce_repo: The replay store behind the service.
        point: Which entry of ``_UNEXPECTED_ERROR_POINTS`` raises.

    Returns:
        The starting browser, the consent state, its cookie name, and the
        callback response.
    """
    browser = TestClient(
        app,
        cookies={routes.settings.session_cookie_name: _CONSENTING_COOKIE},
        raise_server_exceptions=False,
    )
    state_blob, started = _start_consent(browser, "initiate")
    [cookie_name] = started.cookies.keys()
    with monkeypatch.context() as patch:
        if point == "session-resolution":
            patch.setattr("registry.auth.dependencies.enhanced_auth", _raise_unexpected)
        elif point == "server-lookup":
            patch.setattr(routes.server_service, "get_server_info", _raise_unexpected)
        elif point == "completion":
            patch.setattr(routes.get_egress_auth_service(), "handle_callback", _raise_unexpected)
        else:
            patch.setattr(nonce_repo, "consume_nonce", _raise_unexpected)
        response = _complete(browser, state_blob)
    return browser, state_blob, cookie_name, response


@pytest.mark.unit
@pytest.mark.parametrize("point", _UNEXPECTED_ERROR_POINTS)
def test_unexpected_callback_error_refuses_and_expires_the_consent_cookie(
    consent_app, monkeypatch, nonce_repo, point
):
    """An error the callback does not handle still expires the consent cookie.

    The page is the retryable 503 asking the user to connect again, not a 500,
    and no code reaches the provider.
    """
    app, store = consent_app
    exchanges = _record_code_exchanges(monkeypatch)

    _, _, cookie_name, response = _callback_with_unexpected_error(
        app, monkeypatch, nonce_repo, point
    )

    assert response.status_code == 503, response.text
    assert "start connecting again" in response.text
    [(name, morsel)] = _consent_set_cookie(response).items()
    assert (name, morsel["max-age"], morsel["path"]) == (cookie_name, "0", _CALLBACK_PATH)
    assert exchanges == []
    assert store.writes == []


@pytest.mark.unit
@pytest.mark.parametrize("point", _UNEXPECTED_ERROR_POINTS)
def test_replay_after_an_unexpected_callback_error_writes_nothing(
    consent_app, monkeypatch, nonce_repo, point
):
    """The starting browser replaying the same code and state stores nothing.

    The browser is still signed in as the consenting user and the registry
    works again, so only the expired cookie or the burned state stops it.
    """
    app, store = consent_app
    browser, state_blob, _, _ = _callback_with_unexpected_error(app, monkeypatch, nonce_repo, point)
    exchanges = _record_code_exchanges(monkeypatch)

    replay = _complete(browser, state_blob)

    assert replay.status_code == 400, replay.text
    assert exchanges == []
    assert store.writes == [], f"replay after an unexpected error wrote {store.writes}"
