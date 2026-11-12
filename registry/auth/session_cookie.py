"""The registry's read of the session cookie, and the expiry of a refused one.

Every registry read of the session cookie goes through ``request_session_cookie``,
which joins every ``Cookie`` header field with ``join_cookie_fields`` and reads
the session cookie with ``registry/common/session_cookie.py``, as the auth
server's ``/validate`` does. Both services therefore refuse the same
duplicated session cookie, including one an HTTP/2 client sent in a second
field. Starlette's ``request.cookies``, and so a FastAPI ``Cookie`` parameter,
keep the last of two cookies with one name, so neither may read it.

``SessionCookieIsolationMiddleware``, the outermost of the registry's own
middleware, enforces that: it reads the session cookie once into the scope cache
``request_session_cookie`` answers from, then removes every session cookie pair
from the ``Cookie`` header fields every registry middleware and handler sees. A
handler that reads the session cookie any other way gets None. OpenTelemetry's
ASGI middleware wraps the whole app and still sees the raw header, so
``OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST`` must never list
``Cookie``.

The same middleware expires both session cookie variants on every response to a
request whose session cookie was refused as a duplicate, unless the response
already sets or expires the session cookie itself (logout does). Every expiry
carries the attributes the auth server's login gives the cookie, from
``session_cookie_attributes``.
"""

import logging
from typing import (
    Annotated,
    NamedTuple,
)

from fastapi import Depends
from starlette.datastructures import Headers
from starlette.requests import HTTPConnection
from starlette.responses import Response
from starlette.types import (
    ASGIApp,
    Message,
    Receive,
    Scope,
    Send,
)

from ..common.session_cookie import (
    SessionCookieAttributes,
    join_cookie_fields,
    log_refused_session_cookie,
    session_cookie_attributes,
    session_cookie_values,
)
from ..core.config import settings
from ..utils.request_utils import is_request_https

logger = logging.getLogger(__name__)

# ASGI scope key the request's read is cached under, so a request whose cookie
# several readers check (CSRF, auth, audit) is parsed and logged once.
_SCOPE_KEY: str = "registry.session_cookie"


class _SessionCookieRead(NamedTuple):
    """Every value a request carried under the session cookie name, in header order."""

    values: tuple[str, ...]

    @property
    def refused(self) -> bool:
        """Whether the request carried the session cookie more than once."""
        return len(self.values) > 1

    @property
    def value(self) -> str | None:
        """The session cookie when the request carried it exactly once, else None."""
        return self.values[0] if len(self.values) == 1 else None


def _read_scope(
    scope: Scope,
    headers: Headers,
) -> _SessionCookieRead:
    """Read the session cookie of a request's scope once and cache the result in it.

    Args:
        scope: The ASGI scope of the HTTP request or WebSocket.
        headers: The scope's headers.

    Returns:
        Every value the request carried under the session cookie name.
    """
    cached = scope.get(_SCOPE_KEY)
    if cached is None:
        cookie_header = join_cookie_fields(headers.getlist("cookie"))
        cached = _SessionCookieRead(
            values=tuple(
                session_cookie_values(cookie_header, settings.session_cookie_effective_name)
            )
        )
        if cached.refused:
            log_refused_session_cookie(len(cached.values))
        scope[_SCOPE_KEY] = cached
    return cached


def _read(
    connection: HTTPConnection,
) -> _SessionCookieRead:
    """Return the cached read of the session cookie of a request or WebSocket."""
    return _read_scope(connection.scope, connection.headers)


def _is_session_pair(
    segment: bytes,
    name: str,
) -> bool:
    """Return True when one ``;``-separated ``Cookie`` segment is a session cookie pair.

    It matches the pairs ``session_cookie_values`` reads: the exact name, with an ``=``.
    """
    pair_name, separator, _ = segment.decode("latin-1").strip().partition("=")
    return bool(separator) and pair_name.strip() == name


def _without_session_pairs(
    headers: list[tuple[bytes, bytes]],
    name: str,
) -> list[tuple[bytes, bytes]] | None:
    """Return the headers with every session cookie pair removed, or None when there was none.

    Every other pair keeps its bytes. A ``Cookie`` field left with no pair is dropped.

    Args:
        headers: The ASGI scope's raw headers.
        name: The session cookie name.

    Returns:
        The new header list, or None when no field carried a session cookie pair.
    """
    stripped: list[tuple[bytes, bytes]] = []
    changed = False
    for key, value in headers:
        if key.lower() != b"cookie":
            stripped.append((key, value))
            continue
        segments = value.split(b";")
        kept = [segment for segment in segments if not _is_session_pair(segment, name)]
        if len(kept) == len(segments):
            stripped.append((key, value))
            continue
        changed = True
        remaining = b";".join(kept).lstrip()
        if remaining:
            stripped.append((key, remaining))
    return stripped if changed else None


def _session_cookie_attributes(
    connection: HTTPConnection,
) -> SessionCookieAttributes:
    """Return the session cookie attributes for a response to ``connection``.

    Args:
        connection: The request the response answers; it decides whether the
            browser reached the gateway over HTTPS.

    Returns:
        The attributes the auth server's login uses for the same request.
    """
    return session_cookie_attributes(
        name=settings.session_cookie_effective_name,
        is_https=is_request_https(connection),
        secure=settings.session_cookie_secure,
        domain=settings.session_cookie_domain,
        host_prefix=settings.session_cookie_host_prefix,
        max_age=settings.session_max_age_seconds,
    )


def _expiry_headers(
    connection: HTTPConnection,
) -> list[tuple[bytes, bytes]]:
    """Return the ``Set-Cookie`` headers ``clear_session_cookie`` sends for ``connection``."""
    response = Response()
    clear_session_cookie(response, connection)
    return [(key, value) for key, value in response.raw_headers if key == b"set-cookie"]


def _sets_session_cookie(
    headers: list[tuple[bytes, bytes]],
    name: str,
) -> bool:
    """Return True when the response headers already set or expire the session cookie."""
    prefix = f"{name}=".encode("latin-1")
    return any(key.lower() == b"set-cookie" and value.startswith(prefix) for key, value in headers)


def request_session_cookie(
    connection: HTTPConnection,
) -> str | None:
    """Return the request's session cookie, or None when absent or sent more than once.

    Use it as a FastAPI dependency (``Depends(request_session_cookie)``) or
    call it with the request or WebSocket.

    Args:
        connection: The HTTP request or WebSocket.

    Returns:
        The session cookie value, or None.
    """
    return _read(connection).value


def request_carries_session_cookie(
    connection: HTTPConnection,
) -> bool:
    """Return True when the request carried the session cookie at all, once or more.

    A request that carries it is a browser request with ambient credentials,
    even when the cookie is refused as a duplicate or does not resolve to a
    session.

    Args:
        connection: The HTTP request or WebSocket.

    Returns:
        Whether the ``Cookie`` header named the session cookie at least once.
    """
    return bool(_read(connection).values)


def session_cookie_refused(
    connection: HTTPConnection,
) -> bool:
    """Return True when the request carried the session cookie more than once.

    Args:
        connection: The HTTP request or WebSocket.

    Returns:
        Whether the request's session cookie was refused as a duplicate.
    """
    return _read(connection).refused


def request_session_cookie_copies(
    connection: HTTPConnection,
) -> list[str]:
    """Return every value the request carried under the session cookie name, in order.

    The values are unverified: none may be trusted as the caller's session.
    Only logout uses them, to end each session a duplicated cookie names after
    checking its signature.

    Args:
        connection: The HTTP request or WebSocket.

    Returns:
        The values; empty when the request carried no session cookie.
    """
    return list(_read(connection).values)


# The request's session cookie as a route or dependency parameter:
# ``session: SessionCookie = None``. A direct call passes the value instead.
SessionCookie = Annotated[str | None, Depends(request_session_cookie)]


def clear_session_cookie(
    response: Response,
    connection: HTTPConnection,
) -> None:
    """Expire the session cookie: the host-only one, and the one on SESSION_COOKIE_DOMAIN.

    A browser ignores a deletion unless its name, Path and Domain match the
    cookie's, and keeps a host-only cookie and one with a Domain as two cookies.
    Expiring both at Path=/ clears whichever the browser holds, including one
    left from before a SESSION_COOKIE_DOMAIN change. A cookie on a domain the
    deployment does not name, such as one another host under the parent domain
    set, cannot be expired from here. Every expiry carries the attributes the
    login gives the cookie (``session_cookie_attributes``).

    Args:
        response: The response to add the expiry to.
        connection: The request the response answers; it decides whether the
            browser reached the gateway over HTTPS.
    """
    for expiry in _session_cookie_attributes(connection).expiries:
        response.delete_cookie(**expiry)


class SessionCookieIsolationMiddleware:
    """Read the session cookie once, then hide it from everything downstream.

    Register it last, so it is the first entry of ``app.user_middleware`` and
    runs before the audit middleware and every other registry middleware.
    OpenTelemetry's ASGI middleware, which ``FastAPIInstrumentor.instrument_app``
    or the ``opentelemetry-instrument`` wrapper adds, wraps the whole app and
    runs before this one; it records request headers only when
    ``OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST`` lists them, and
    that list must never include ``Cookie``, or every span carries the raw
    session cookie.

    For each HTTP request and WebSocket handshake this middleware caches the
    shared reader's result in the scope, where ``request_session_cookie`` and
    ``session_cookie_refused`` find it, and removes every session cookie pair
    from the scope's ``Cookie`` header fields. Every other cookie, the egress
    consent cookies included, keeps its bytes. A later
    ``request.cookies.get(...)`` or ``Cookie(alias=...)`` parameter therefore
    sees no session cookie, and cannot pick one of two copies the shared reader
    refused.

    When the request's session cookie was refused as a duplicate, every HTTP
    response gets the expiry of both session cookie variants, whatever its
    status (a 401 from a dependency, a 403 from CSRF, the dashboard's redirect
    to /login), unless it already sets or expires the session cookie.
    """

    def __init__(
        self,
        app: ASGIApp,
    ) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        read = _read_scope(scope, Headers(scope=scope))
        name = settings.session_cookie_effective_name
        stripped = _without_session_pairs(scope["headers"], name)
        if stripped is not None:
            scope["headers"] = stripped
        if scope["type"] != "http" or not read.refused:
            await self.app(scope, receive, send)
            return

        async def _send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if not _sets_session_cookie(headers, name):
                    logger.info("Expiring the refused duplicated session cookie")
                    headers.extend(_expiry_headers(HTTPConnection(scope)))
                    message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, _send)
