"""Both services refuse a ``Cookie`` header that carries the session cookie twice.

Another host under the parent domain can set a cookie with the session cookie's
name for that parent domain, and the browser then sends both. The browser
orders them by Path length and then by creation time, so whoever sets the
second cookie decides which one comes first and which one comes last. Picking
either is unsafe, so the auth server's ``/validate`` treats a duplicated session
cookie as no session cookie at all, and every registry read of the session
cookie refuses it.

Two accounts log in through the real login callback, and the header carries
both live session cookies in either order, so a reader that picked one would
authenticate as that account. The tests drive the real ``/validate`` and the
registry's real routes; only the session store, the scope repository and the
egress service are replaced.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import (
    AsyncMock,
    patch,
)

import httpx
import pytest
from fastapi.testclient import TestClient
from itsdangerous import URLSafeTimedSerializer
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

import registry.api.egress_auth_routes as egress_routes
import registry.auth.dependencies as registry_dependencies
import registry.auth.routes as registry_routes
from registry.audit import (
    AuditLogger,
    AuditMiddleware,
)
from registry.auth.csrf import generate_csrf_token
from tests.auth_server.unit.test_egress_vault_identity_properties import (
    _COLLIDING_PROFILE,
    _HOP_BODY,
    _HOP_ORIGIN,
    _HOP_ORIGINAL_URL,
    _HOP_SCOPE,
    _HOP_UPSTREAM,
    _client,
    _FakeSessionCollection,
    _login,
    _registry_context,
    _registry_cookie_context,
    _session_store,
    _signed_bearer,
    _validate_hop_tokens,
    server_module,
)

pytestmark = [pytest.mark.unit, pytest.mark.auth]

_ALICE_ID: str = "account-alice"
_BOB_ID: str = "account-bob"
_CAROL_ID: str = "account-carol"
_SERVER_PATH: str = "/github-mcp"
_AUTHORIZE_URL: str = "https://provider.example.com/authorize?state=s"


@dataclass
class _Sessions:
    """Two live sessions in one store, and the cookie of each."""

    alice: str
    bob: str
    store: _FakeSessionCollection

    def session_id(
        self,
        cookie: str,
    ) -> str:
        """Return the session id a session cookie carries."""
        return server_module.signer.loads(cookie)


def _cookie_header(
    *values: str,
) -> str:
    """Return a ``Cookie`` header with one session cookie pair per value, in order."""
    name = server_module.settings.session_cookie_name
    return "; ".join(f"{name}={value}" for value in values)


@pytest.fixture(autouse=True)
def _no_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep /validate's M2M group enrichment and the registry's audit sink off MongoDB."""
    monkeypatch.setattr("mongodb_groups_enrichment.should_enrich_groups", lambda result: False)
    monkeypatch.setattr(AuditLogger, "log_event", AsyncMock(return_value=None))


@pytest.fixture
def sessions() -> _Sessions:
    """Log Alice and Bob in through the real callback and keep both sessions."""
    alice = _login("keycloak", _ALICE_ID, _COLLIDING_PROFILE)
    bob = _login("keycloak", _BOB_ID, _COLLIDING_PROFILE)
    assert alice.cookie and bob.cookie, (alice.set_cookie_headers, bob.set_cookie_headers)
    alice.store.documents.extend(bob.store.documents)
    return _Sessions(alice=alice.cookie, bob=bob.cookie, store=alice.store)


@pytest.fixture(params=["alice-first", "alice-last"])
def duplicated(
    request: pytest.FixtureRequest,
    sessions: _Sessions,
) -> str:
    """A ``Cookie`` header carrying Alice's and Bob's live session cookies."""
    if request.param == "alice-first":
        return _cookie_header(sessions.alice, sessions.bob)
    return _cookie_header(sessions.bob, sessions.alice)


def _validate(
    store: _FakeSessionCollection,
    *cookie_fields: str,
) -> httpx.Response:
    """Send a browser MCP POST's ``/validate`` with only ``Cookie`` header fields as credential.

    Each of ``cookie_fields`` goes in its own ``Cookie`` header field, in order,
    as an HTTP/2 client may send them. The request carries the gateway's own
    Origin, so /validate's cross-origin cookie check passes.
    """
    headers = [
        ("X-Original-URL", _HOP_ORIGINAL_URL),
        ("X-Original-Method", "POST"),
        ("Origin", _HOP_ORIGIN),
        ("X-Body", _HOP_BODY),
        ("X-Resolved-Upstream", _HOP_UPSTREAM),
        ("X-Registry-Api-Auth", "1"),
        ("X-Validate-Source-Secret", server_module.settings.auth_server_nginx_marker_secret),
        *(("Cookie", field) for field in cookie_fields),
    ]
    client = _client()
    with (
        _session_store(store),
        patch.object(
            server_module,
            "_gateway_trusted_origins",
            return_value=frozenset({_HOP_ORIGIN}),
        ),
        patch.object(server_module, "map_groups_to_scopes", AsyncMock(return_value=[_HOP_SCOPE])),
        patch.object(server_module, "validate_server_tool_access", AsyncMock(return_value=True)),
        patch.object(server_module, "get_mcp_logger", return_value=None),
    ):
        response = client.get("/validate", headers=headers)
    client.cookies.clear()
    return response


def test_validate_treats_a_duplicated_session_cookie_as_no_cookie(sessions, duplicated):
    """/validate answers a duplicated session cookie exactly as it answers none."""
    refused = _validate(sessions.store, duplicated)
    without_cookie = _validate(sessions.store)

    assert refused.status_code == 401, refused.text
    assert (refused.status_code, refused.json()) == (
        without_cookie.status_code,
        without_cookie.json(),
    )


def test_validate_refuses_a_duplicate_in_a_second_cookie_field(sessions):
    """A duplicate sent in a second ``Cookie`` header field is refused like one in the first.

    HTTP/2 lets a client split its cookies over several fields, and a proxy
    that does not merge them passes the fields on separately.
    """
    refused = _validate(
        sessions.store, _cookie_header(sessions.alice), _cookie_header(sessions.bob)
    )
    without_cookie = _validate(sessions.store)

    assert refused.status_code == 401, refused.text
    assert refused.json() == without_cookie.json()


def test_validate_finds_the_session_cookie_in_a_second_cookie_field(sessions):
    """A session cookie sent once, in the second ``Cookie`` field, validates as that session."""
    response = _validate(sessions.store, "theme=dark", _cookie_header(sessions.alice))

    assert response.status_code == 200, response.text
    context = _registry_context(response.headers["X-Internal-Token-Registry"], sessions.store)
    assert context["session_id"] == sessions.session_id(sessions.alice)


def test_validate_accepts_one_session_cookie(sessions):
    """The same session cookie sent once validates as that session."""
    hop = _validate_hop_tokens({"Cookie": _cookie_header(sessions.alice)}, sessions.store)

    assert hop.registry_context["session_id"] == sessions.session_id(sessions.alice)


def test_validate_falls_through_to_the_bearer_beside_a_duplicated_cookie(sessions, duplicated):
    """With a duplicated session cookie, /validate authenticates the bearer as if no cookie came.

    A session cookie wins over a bearer at /validate, so before the refusal the
    request resolved as one of the two sessions.
    """
    bearer = _signed_bearer("keycloak", _CAROL_ID, _COLLIDING_PROFILE, {"groups": ["hop-group"]})
    headers = {"Cookie": duplicated, "Authorization": f"Bearer {bearer}"}

    hop = _validate_hop_tokens(headers, sessions.store, bearer_provider="keycloak")

    assert not hop.registry_context.get("session_id"), hop.registry_context
    assert hop.registry_context["egress_user"] == _CAROL_ID


@pytest.fixture
def registry_app(
    monkeypatch: pytest.MonkeyPatch,
    sessions: _Sessions,
) -> Iterator[Any]:
    """The registry app, resolving sessions from the logins' store.

    The registry verifies the cookie with the auth server's signer, as a
    deployment that shares ``SECRET_KEY`` does. Groups map to no scopes.
    """
    from registry.main import app

    monkeypatch.setattr(registry_dependencies, "signer", server_module.signer)
    monkeypatch.setattr(
        "registry.auth.session_store._get_collection", AsyncMock(return_value=sessions.store)
    )
    monkeypatch.setattr(
        registry_dependencies, "map_cognito_groups_to_scopes", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        registry_dependencies,
        "resolve_scope_access",
        AsyncMock(return_value=SimpleNamespace(servers=[], tools={})),
    )
    monkeypatch.setattr(
        registry_dependencies, "get_ui_permissions_for_user", AsyncMock(return_value={})
    )
    app.dependency_overrides.clear()
    yield app
    app.dependency_overrides.clear()


def _registry_get(
    app: Any,
    path: str,
    *cookie_fields: str,
) -> httpx.Response:
    """GET ``path`` on the registry, one ``Cookie`` field per argument, not following redirects."""
    headers = [("Cookie", field) for field in cookie_fields]
    return TestClient(app).get(path, headers=headers, follow_redirects=False)


# Registry routes that read the session cookie, with the status each answers
# for a live session sent once and for a refused one.
_SESSION_ROUTES = [
    pytest.param("/api/auth/me", 200, 401, id="nginx_proxied_auth-cookie-path"),
    pytest.param("/api/config", 200, 401, id="enhanced_auth"),
    pytest.param("/api/auth/csrf-token", 200, 401, id="csrf-token"),
]


@pytest.mark.parametrize(("path", "accepted", "refused"), _SESSION_ROUTES)
def test_registry_session_routes_accept_one_session_cookie(
    registry_app, sessions, path, accepted, refused
):
    """Each route accepts the live session sent once."""
    response = _registry_get(registry_app, path, _cookie_header(sessions.alice))

    assert response.status_code == accepted, response.text


@pytest.mark.parametrize(("path", "accepted", "refused"), _SESSION_ROUTES)
def test_registry_session_routes_refuse_a_duplicated_session_cookie(
    registry_app, duplicated, path, accepted, refused
):
    """Each route answers a duplicated session cookie as it answers no session."""
    response = _registry_get(registry_app, path, duplicated)

    assert response.status_code == refused, response.text


@pytest.mark.parametrize(("path", "accepted", "refused"), _SESSION_ROUTES)
def test_registry_session_routes_refuse_a_duplicate_in_a_second_cookie_field(
    registry_app, sessions, path, accepted, refused
):
    """Each route refuses a duplicate that arrives in a second ``Cookie`` header field."""
    response = _registry_get(
        registry_app, path, _cookie_header(sessions.alice), _cookie_header(sessions.bob)
    )

    assert response.status_code == refused, response.text


@pytest.mark.parametrize(("path", "accepted", "refused"), _SESSION_ROUTES)
def test_registry_session_routes_find_the_session_cookie_in_a_second_cookie_field(
    registry_app, sessions, path, accepted, refused
):
    """Each route accepts the live session sent once, in the second ``Cookie`` header field."""
    response = _registry_get(registry_app, path, "theme=dark", _cookie_header(sessions.alice))

    assert response.status_code == accepted, response.text


def test_dashboard_sends_a_duplicated_session_cookie_to_login(registry_app, duplicated):
    """The dashboard redirects a duplicated session cookie to /login, as it does no cookie."""
    response = _registry_get(registry_app, "/api/", duplicated)

    assert response.status_code == 302, response.text
    assert response.headers["location"] == "/login"


def test_registry_raw_cookie_path_reads_one_session_cookie(sessions):
    """``enhanced_auth`` resolves the live session sent once."""
    context = _registry_cookie_context(sessions.alice, sessions.store)

    assert context["session_id"] == sessions.session_id(sessions.alice)


def test_logout_post_demands_csrf_for_one_session_cookie(registry_app, sessions):
    """A logout POST from a live session without a CSRF token is refused (403)."""
    response = TestClient(registry_app).post(
        "/api/auth/logout",
        headers={"Cookie": _cookie_header(sessions.alice)},
        follow_redirects=False,
    )

    assert response.status_code == 403, response.text


def _no_auth_server(
    *args: Any,
    **kwargs: Any,
) -> None:
    """Stand in for ``httpx.AsyncClient`` in the logout route: the auth server is unreachable."""
    raise OSError("no auth server")


def _logout(
    app: Any,
    cookie_header: str,
) -> tuple[httpx.Response, AsyncMock]:
    """GET logout with ``cookie_header``; return the response and the session delete mock.

    The browser's logout link is a GET, which takes no CSRF token. Only the
    logout route's ``httpx`` is replaced, so the app's own clients still build.
    """
    delete_session = AsyncMock(return_value=True)
    with (
        patch("registry.auth.session_store.delete_session", delete_session),
        patch.object(registry_routes, "httpx", SimpleNamespace(AsyncClient=_no_auth_server)),
    ):
        response = _registry_get(app, "/api/auth/logout", cookie_header)
    return response, delete_session


def _deleted_session_ids(
    delete_session: AsyncMock,
) -> set[str]:
    """Return the session ids a ``delete_session`` mock was awaited with."""
    return {awaited.args[0] for awaited in delete_session.await_args_list}


def test_logout_deletes_every_session_a_duplicated_cookie_names(registry_app, sessions, duplicated):
    """Logout with a duplicated session cookie deletes the session of each valid copy.

    Neither copy is trusted as the caller's session, but the user asked to sign
    out, so both sessions end.
    """
    response, delete_session = _logout(registry_app, duplicated)

    assert response.status_code == 303, response.text
    assert response.headers["location"] == "/login"
    assert _deleted_session_ids(delete_session) == {
        sessions.session_id(sessions.alice),
        sessions.session_id(sessions.bob),
    }


def test_logout_post_refuses_a_duplicated_cookie_and_expires_it(registry_app, duplicated):
    """A logout POST with a duplicated session cookie fails CSRF, yet expires both cookies.

    CSRF never skips a request that carries the session cookie, and a refused
    duplicate resolves to no session, so the POST gets a 403 and ends no
    session. The response still expires the cookie, so the browser is signed
    out; the GET logout ends the sessions.
    """
    delete_session = AsyncMock(return_value=True)
    with patch("registry.auth.session_store.delete_session", delete_session):
        response = TestClient(registry_app).post(
            "/api/auth/logout", headers={"Cookie": duplicated}, follow_redirects=False
        )

    name = server_module.settings.session_cookie_effective_name
    expiries = [h for h in response.headers.get_list("set-cookie") if h.startswith(f"{name}=")]
    assert response.status_code == 403, response.text
    assert expiries, response.headers
    delete_session.assert_not_awaited()


@pytest.mark.parametrize(
    "forged",
    [
        pytest.param("not-a-signed-value", id="unsigned"),
        pytest.param("bob-session-signed-with-another-key", id="other-key"),
    ],
)
def test_logout_ignores_a_forged_copy_beside_a_valid_one(registry_app, sessions, forged):
    """Only a copy whose signature the signer accepts names a session to delete.

    The other-key copy carries Bob's real session id under a signature from a
    key the gateway does not hold, so his session must survive.
    """
    if forged == "bob-session-signed-with-another-key":
        other_key = "Fq8v3Lp0Wm7Xr2Tn5Yb9Kc4Hd6Jf1Ga8Qe3Us0Vt"  # pragma: allowlist secret
        forged = URLSafeTimedSerializer(other_key).dumps(sessions.session_id(sessions.bob))

    response, delete_session = _logout(registry_app, _cookie_header(forged, sessions.alice))

    assert response.status_code == 303, response.text
    assert _deleted_session_ids(delete_session) == {sessions.session_id(sessions.alice)}


def test_logout_logs_the_copy_count_and_no_value(registry_app, sessions, duplicated, caplog):
    """The logout of a duplicated cookie logs how many copies came, never a cookie value."""
    with caplog.at_level(logging.INFO, logger="registry.auth.routes"):
        _logout(registry_app, duplicated)

    messages = [r.getMessage() for r in caplog.records if r.name == "registry.auth.routes"]
    assert any("2 times" in message for message in messages), messages
    for message in messages:
        assert sessions.alice not in message and sessions.bob not in message


def _logout_lookups(
    app: Any,
    cookie_header: str,
) -> tuple[list[str], AsyncMock]:
    """GET logout with ``cookie_header``; return the session ids looked up and the delete mock.

    The lookup spy wraps the real session-store read, so every copy that
    passes the signature check reaches the store and is counted once.
    """
    lookup = AsyncMock(wraps=registry_dependencies._store_resolve_session)
    with patch.object(registry_dependencies, "_store_resolve_session", lookup):
        _, delete_session = _logout(app, cookie_header)
    return [awaited.args[0] for awaited in lookup.await_args_list], delete_session


def test_logout_looks_up_identical_copies_once(registry_app, sessions):
    """A header repeating one signed value looks the session up once and still ends it.

    Logout needs no authentication or CSRF token for a duplicated cookie, so
    without deduplication one request could make the registry read the
    session store once per copy.
    """
    lookups, delete_session = _logout_lookups(registry_app, _cookie_header(*[sessions.alice] * 6))

    assert lookups == [sessions.session_id(sessions.alice)]
    assert _deleted_session_ids(delete_session) == {sessions.session_id(sessions.alice)}


def test_logout_resolves_at_most_the_cap_of_distinct_copies(registry_app, caplog):
    """Past the cap, logout looks up only the first distinct copies and logs the rest dropped."""
    from registry.auth.routes import _MAX_LOGOUT_SESSION_COPIES

    session_ids = [f"unknown-session-{index}" for index in range(_MAX_LOGOUT_SESSION_COPIES + 3)]
    copies = [server_module.signer.dumps(session_id) for session_id in session_ids]

    with caplog.at_level(logging.INFO, logger="registry.auth.routes"):
        lookups, _ = _logout_lookups(registry_app, _cookie_header(*copies))

    assert lookups == session_ids[:_MAX_LOGOUT_SESSION_COPIES]
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == "registry.auth.routes" and r.levelno == logging.WARNING
    ]
    assert warnings, "no warning when the cap dropped copies"
    for message in warnings:
        assert not any(copy in message for copy in copies), message


def test_logout_looks_up_each_of_two_distinct_copies(registry_app, sessions, duplicated):
    """Two distinct live copies are each looked up once, below the cap."""
    lookups, _ = _logout_lookups(registry_app, duplicated)

    assert sorted(lookups) == sorted(
        [sessions.session_id(sessions.alice), sessions.session_id(sessions.bob)]
    )


def test_logout_get_deletes_the_session_sent_once(registry_app, sessions):
    """Logout deletes the server-side session of the one session cookie it received."""
    response, delete_session = _logout(registry_app, _cookie_header(sessions.alice))

    assert response.status_code == 303, response.text
    delete_session.assert_awaited_once_with(sessions.session_id(sessions.alice))


def _health_websocket_accepts(
    app: Any,
    cookie_header: str | None,
    query_session: str | None = None,
) -> bool:
    """Open the health WebSocket and report whether it authenticated the caller.

    The health service is replaced by one that accepts the socket, records it
    and closes it, so an authenticated connection ends at once.

    Args:
        app: The registry app.
        cookie_header: The handshake's ``Cookie`` header, or None to send none.
        query_session: A session cookie value to put in the URL under the
            session cookie's name, or None.
    """
    accepted: list[bool] = []

    async def _accept_and_close(websocket: Any) -> bool:
        accepted.append(True)
        await websocket.accept()
        await websocket.close()
        return False

    url = "/api/health/ws/health_status"
    if query_session is not None:
        url += f"?{server_module.settings.session_cookie_name}={query_session}"
    headers = {} if cookie_header is None else {"Cookie": cookie_header}
    with patch("registry.health.routes.health_service.add_websocket_connection", _accept_and_close):
        try:
            with TestClient(app).websocket_connect(url, headers=headers) as websocket:
                websocket.receive()
        except WebSocketDisconnect:
            pass
    return accepted == [True]


def test_health_websocket_accepts_one_session_cookie(registry_app, sessions):
    """The health WebSocket authenticates the live session sent once."""
    assert _health_websocket_accepts(registry_app, _cookie_header(sessions.alice))


def test_health_websocket_refuses_a_duplicated_session_cookie(registry_app, duplicated):
    """The health WebSocket closes on a duplicated session cookie before accepting it."""
    assert not _health_websocket_accepts(registry_app, duplicated)


def test_health_websocket_refuses_a_session_in_the_query_string(registry_app, sessions):
    """A live session sent only in the URL does not authenticate the health WebSocket.

    A URL lands in access logs, so a session credential must never travel in one.
    """
    assert not _health_websocket_accepts(registry_app, None, query_session=sessions.alice)


def test_health_websocket_ignores_a_query_session_beside_a_duplicated_cookie(
    registry_app, sessions, duplicated
):
    """A refused duplicated cookie is not rescued by a live session in the URL."""
    assert not _health_websocket_accepts(registry_app, duplicated, query_session=sessions.alice)


def _audit_identity(
    cookie_header: str,
    log_dir: str,
) -> tuple[str, str]:
    """Return the username and credential type the audit middleware records."""
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/version",
            "headers": [(b"cookie", cookie_header.encode())],
        }
    )
    middleware = AuditMiddleware(app=None, audit_logger=AuditLogger(log_dir=log_dir))
    identity = asyncio.run(middleware._extract_identity(request))
    return identity.username, identity.credential_type


def test_audit_identity_reads_one_session_cookie(registry_app, sessions, tmp_path):
    """The audit record names the user of the live session sent once."""
    identity = _audit_identity(_cookie_header(sessions.alice), str(tmp_path))

    assert identity == ("alice@example.com", "session_cookie")


def test_audit_identity_refuses_a_duplicated_session_cookie(registry_app, duplicated, tmp_path):
    """The audit record of a duplicated session cookie is anonymous, with no cookie credential."""
    assert _audit_identity(duplicated, str(tmp_path)) == ("anonymous", "none")


@pytest.fixture
def initiate_app(
    monkeypatch: pytest.MonkeyPatch,
    registry_app: Any,
    sessions: _Sessions,
) -> Any:
    """The registry app with egress enabled and Alice resolved as the caller.

    ``nginx_proxied_auth`` returns Alice's session context, as the internal
    token ``/validate`` minted for her would; the route's own browser-session
    check then reads the request's session cookie.
    """
    context = _registry_cookie_context(sessions.alice, sessions.store)
    server = {
        "path": _SERVER_PATH,
        "egress_auth_mode": "oauth_user",
        "egress_oauth": {"provider": "github", "client_id": "Iv1.test"},
    }
    service = SimpleNamespace(build_consent_url=AsyncMock(return_value=_AUTHORIZE_URL))
    monkeypatch.setattr(egress_routes.settings, "egress_auth_enabled", True)
    monkeypatch.setattr(
        egress_routes.server_service, "get_server_info", AsyncMock(return_value=server)
    )
    monkeypatch.setattr(egress_routes, "get_egress_auth_service", lambda: service)
    registry_app.dependency_overrides[registry_dependencies.nginx_proxied_auth] = lambda: context
    return registry_app


def _initiate(
    app: Any,
    cookie_header: str,
    session_id: str,
) -> httpx.Response:
    """Start a consent at ``/api/egress-auth/initiate`` with a valid CSRF token."""
    return TestClient(app).post(
        "/api/egress-auth/initiate",
        json={"server_path": _SERVER_PATH},
        headers={"Cookie": cookie_header, "X-CSRF-Token": generate_csrf_token(session_id)},
    )


def test_initiate_starts_a_consent_for_one_session_cookie(initiate_app, sessions):
    """The browser holding Alice's session cookie once starts her consent."""
    response = _initiate(
        initiate_app, _cookie_header(sessions.alice), sessions.session_id(sessions.alice)
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"authorize_url": _AUTHORIZE_URL}


def test_initiate_refuses_a_duplicated_session_cookie(initiate_app, sessions, duplicated):
    """A duplicated session cookie is not the browser session, even with Alice's beside it."""
    response = _initiate(initiate_app, duplicated, sessions.session_id(sessions.alice))

    assert response.status_code == 403, response.text
    assert "authorize_url" not in response.text
