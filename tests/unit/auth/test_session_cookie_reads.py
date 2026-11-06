"""The registry reads the session cookie through one reader, which refuses duplicates.

Every registry route and dependency that takes the session cookie gets it from
``request_session_cookie``, never from a FastAPI ``Cookie`` parameter (which
keeps the last of two cookies with one name). The outermost registry middleware
removes the session cookie from the ``Cookie`` fields every handler sees, so a read
that bypasses the shared reader gets None while every other cookie keeps its
bytes. When a browser request is refused for a duplicated session cookie and
the registry can answer it (a 401, the dashboard's redirect to /login, and
logout), the response expires the host-only cookie and the cookie on
``SESSION_COOKIE_DOMAIN`` at Path=/. Logout expires both whatever it received.
"""

from collections.abc import Iterator
from http.cookies import SimpleCookie
from typing import (
    Annotated,
    Any,
)
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import (
    Cookie,
    Request,
    WebSocket,
)
from fastapi.dependencies.models import Dependant
from fastapi.routing import (
    APIRoute,
    APIWebSocketRoute,
    iter_route_contexts,
)
from fastapi.testclient import TestClient
from starlette.requests import HTTPConnection

from registry.audit import AuditLogger
from registry.auth.session_cookie import (
    request_session_cookie,
    session_cookie_refused,
)
from registry.core.config import settings

pytestmark = [pytest.mark.unit, pytest.mark.auth]

_PLAIN_NAME: str = "mcp_gateway_session"
_DOMAIN: str = ".example.com"


@pytest.fixture(autouse=True)
def _no_audit_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the registry's audit middleware off MongoDB."""
    monkeypatch.setattr(AuditLogger, "log_event", AsyncMock(return_value=None))


def _dependants(
    dependant: Dependant,
) -> Iterator[Dependant]:
    """Yield ``dependant`` and every dependency below it."""
    yield dependant
    for child in dependant.dependencies:
        yield from _dependants(child)


def _route_dependants(
    app: Any,
) -> Iterator[tuple[str, Dependant]]:
    """Yield each HTTP and WebSocket route's path and its effective dependant.

    ``iter_route_contexts`` flattens included routers, and a route context's
    ``dependant`` carries the dependencies its routers add.
    """
    for route_context in iter_route_contexts(app.routes):
        dependant = getattr(route_context, "dependant", None)
        if dependant is not None:
            yield route_context.path, dependant


def test_no_route_reads_the_session_cookie_as_a_cookie_parameter():
    """No route or dependency declares ``Cookie(alias=<session cookie name>)``.

    The walk covers each route's whole dependency graph, so a dependency that
    another dependency pulls in is checked too, and it finds the shared reader
    on the routes known to take the session cookie.
    """
    from registry.auth.session_cookie import request_session_cookie
    from registry.main import app

    declared: list[str] = []
    readers: set[str] = set()
    for path, route_dependant in _route_dependants(app):
        for dependant in _dependants(route_dependant):
            if dependant.call is request_session_cookie:
                readers.add(path)
            for param in dependant.cookie_params:
                if param.alias == settings.session_cookie_name:
                    declared.append(f"{path}: {getattr(dependant.call, '__name__', '?')}")

    assert declared == []
    assert {
        "/api/",
        "/api/auth/logout",
        "/api/auth/csrf-token",
        "/api/auth/me",
        "/api/config",
    } <= readers


@pytest.fixture
def cookie_settings(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """Configure the session cookie name and domain; return the name."""
    domain = request.param
    monkeypatch.setattr(settings, "session_cookie_name", _PLAIN_NAME)
    monkeypatch.setattr(settings, "session_cookie_secure", True)
    monkeypatch.setattr(settings, "session_cookie_domain", domain)
    return settings.session_cookie_name


# Domain -> the Domain of every session cookie expiry expected.
_COOKIE_SETTINGS = [
    pytest.param(None, {""}, id="host-only"),
    pytest.param(_DOMAIN, {"", _DOMAIN}, id="domain"),
]


def _session_cookie_expiries(
    response: httpx.Response,
    name: str,
) -> dict[str, SimpleCookie]:
    """Return the session cookie ``Set-Cookie`` headers of a response, keyed by Domain.

    Every one must be an expiry (empty value, Max-Age=0) at Path=/.
    """
    expiries = {}
    for header in response.headers.get_list("set-cookie"):
        parsed = SimpleCookie()
        parsed.load(header)
        if name not in parsed:
            continue
        morsel = parsed[name]
        assert (morsel.value, morsel["max-age"], morsel["path"]) == ("", "0", "/"), header
        assert morsel["domain"] not in expiries, response.headers.get_list("set-cookie")
        expiries[morsel["domain"]] = morsel
    return expiries


# Browser routes that can answer a refused request, with the status they answer.
_BROWSER_ROUTES = [
    pytest.param("GET", "/api/auth/me", 401, id="nginx_proxied_auth-401"),
    pytest.param("GET", "/api/config", 401, id="enhanced_auth-401"),
    pytest.param("GET", "/api/auth/csrf-token", 401, id="csrf-token-401"),
    pytest.param("GET", "/api/", 302, id="dashboard-redirect"),
    pytest.param("GET", "/api/auth/logout", 303, id="logout-get"),
    # CSRF refuses a POST whose session cookie resolves to no session.
    pytest.param("POST", "/api/auth/logout", 403, id="logout-post-csrf-refused"),
]


@pytest.mark.parametrize(
    ("cookie_settings", "domains"), _COOKIE_SETTINGS, indirect=["cookie_settings"]
)
@pytest.mark.parametrize(("method", "path", "status"), _BROWSER_ROUTES)
def test_refused_browser_request_expires_both_session_cookies(
    cookie_settings, domains, method, path, status
):
    """A browser route that refuses a duplicated session cookie expires every variant.

    With ``SESSION_COOKIE_DOMAIN`` set, that is the host-only cookie and the one
    on the domain; otherwise only the host-only one.
    """
    from registry.main import app

    cookie_header = f"{cookie_settings}=first; {cookie_settings}=second"

    response = TestClient(app).request(
        method, path, headers={"Cookie": cookie_header}, follow_redirects=False
    )

    assert response.status_code == status, response.text
    expiries = _session_cookie_expiries(response, cookie_settings)
    assert set(expiries) == domains


@pytest.mark.parametrize(("method", "path", "status"), _BROWSER_ROUTES[:4])
def test_a_rejected_single_session_cookie_is_not_expired(monkeypatch, method, path, status):
    """A route that rejects a session cookie sent once leaves the browser's cookie alone.

    Only a refused duplicate expires the session cookie; a 401 for any other
    reason (here a cookie that does not resolve) sets no cookie.
    """
    from registry.main import app

    monkeypatch.setattr(settings, "session_cookie_domain", _DOMAIN)
    cookie_header = f"{settings.session_cookie_name}=not-a-signed-session"

    response = TestClient(app).request(
        method, path, headers={"Cookie": cookie_header}, follow_redirects=False
    )

    assert response.status_code == status, response.text
    assert _session_cookie_expiries(response, settings.session_cookie_name) == {}


_PROBE_PATH: str = "/__session-cookie-probe"
_WS_PROBE_PATH: str = "/__session-cookie-probe-ws"


def _probe_report(
    connection: HTTPConnection,
    parameter: str | None,
) -> dict[str, Any]:
    """Report what a handler sees of the session cookie through each way of reading it."""
    name = settings.session_cookie_name
    return {
        "starlette": connection.cookies.get(name),
        "parameter": parameter,
        "reader": request_session_cookie(connection),
        "refused": session_cookie_refused(connection),
        "cookie_fields": connection.headers.getlist("cookie"),
    }


@pytest.fixture
def probe_app() -> Iterator[Any]:
    """The registry app with an HTTP and a WebSocket route that report the session cookie.

    Each route reads the cookie the ways a future handler might: Starlette's
    ``cookies``, a FastAPI ``Cookie(alias=...)`` parameter, and the shared
    reader. The routes go first so the SPA catch-all cannot answer them.
    """
    from registry.main import app

    name = settings.session_cookie_name

    async def _http_probe(
        request: Request,
        parameter: Annotated[str | None, Cookie(alias=name)] = None,
    ) -> dict[str, Any]:
        return _probe_report(request, parameter)

    async def _ws_probe(
        websocket: WebSocket,
        parameter: Annotated[str | None, Cookie(alias=name)] = None,
    ) -> None:
        await websocket.accept()
        await websocket.send_json(_probe_report(websocket, parameter))
        await websocket.close()

    routes = [
        APIRoute(_PROBE_PATH, _http_probe, methods=["GET"]),
        APIWebSocketRoute(_WS_PROBE_PATH, _ws_probe),
    ]
    app.router.routes[0:0] = routes
    try:
        yield app
    finally:
        for route in routes:
            app.router.routes.remove(route)


def _probe(
    app: Any,
    *cookie_fields: str,
) -> dict[str, Any]:
    """GET the HTTP probe with one ``Cookie`` header field per argument."""
    response = TestClient(app).get(_PROBE_PATH, headers=[("Cookie", f) for f in cookie_fields])
    assert response.status_code == 200, response.text
    return response.json()


def test_handlers_see_no_session_cookie_but_the_shared_reader_does(probe_app):
    """``request.cookies`` and a ``Cookie`` parameter get None; the shared reader gets the value.

    So a handler that reads the session cookie past the shared reader fails
    closed instead of picking one of two copies.
    """
    name = settings.session_cookie_name

    report = _probe(probe_app, f"theme=dark; {name}=session-value")

    assert (report["starlette"], report["parameter"]) == (None, None)
    assert (report["reader"], report["refused"]) == ("session-value", False)


@pytest.mark.parametrize(
    ("cookie_fields", "expected"),
    [
        pytest.param(
            ["theme=dark; {name}=s1; egress_consent_d1=a=b"],
            ["theme=dark; egress_consent_d1=a=b"],
            id="middle",
        ),
        pytest.param(
            ['{name}=s1; egress_consent_d1="q v"'], ['egress_consent_d1="q v"'], id="first"
        ),
        pytest.param(["{name}=s1"], [], id="only-pair-drops-the-field"),
        pytest.param(["theme=dark", "{name}=s1"], ["theme=dark"], id="second-field"),
        pytest.param(["theme=dark;lang=en"], ["theme=dark;lang=en"], id="no-session-unchanged"),
        pytest.param(
            ["old_{name}=x; {name}_old=y; {name}"],
            ["old_{name}=x; {name}_old=y; {name}"],
            id="look-alikes-unchanged",
        ),
    ],
)
def test_other_cookies_reach_handlers_byte_identical(probe_app, cookie_fields, expected):
    """Only session cookie pairs leave the ``Cookie`` fields; every other pair keeps its bytes."""
    name = settings.session_cookie_name

    report = _probe(probe_app, *(field.format(name=name) for field in cookie_fields))

    assert report["cookie_fields"] == [field.format(name=name) for field in expected]


@pytest.mark.parametrize(
    "cookie_fields",
    [
        pytest.param(["{name}=first; {name}=second"], id="one-field"),
        pytest.param(["{name}=first", "theme=dark; {name}=second"], id="two-fields"),
    ],
)
def test_a_duplicated_session_cookie_is_still_refused(probe_app, cookie_fields):
    """Stripping the pairs does not hide a duplicate from the shared reader, which refuses it."""
    name = settings.session_cookie_name

    report = _probe(probe_app, *(field.format(name=name) for field in cookie_fields))

    assert (report["reader"], report["refused"]) == (None, True)
    assert (report["starlette"], report["parameter"]) == (None, None)
    assert all(name not in field for field in report["cookie_fields"]), report


def test_websocket_handlers_see_no_session_cookie_but_the_shared_reader_does(probe_app):
    """A WebSocket handshake gets the same treatment as an HTTP request."""
    name = settings.session_cookie_name
    headers = {"Cookie": f"{name}=session-value; theme=dark"}

    with TestClient(probe_app).websocket_connect(_WS_PROBE_PATH, headers=headers) as websocket:
        report = websocket.receive_json()

    assert (report["starlette"], report["parameter"]) == (None, None)
    assert (report["reader"], report["refused"]) == ("session-value", False)
    assert report["cookie_fields"] == ["theme=dark"]
