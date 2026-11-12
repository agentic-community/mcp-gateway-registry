"""Property: the isolation middleware hides the session cookie and keeps every other cookie.

``SessionCookieIsolationMiddleware`` removes each session cookie pair from the
scope's ``Cookie`` header fields and caches the shared reader's result. The
design holds only if, for any header a client can send, Starlette's parser
finds no session cookie afterwards, parses every other pair of every field
exactly as before, and the cache holds what ``single_session_cookie`` reads
from the original fields. The generated headers mix the session cookie with
look-alike names, values with spaces, ``=`` and latin-1 bytes, segments with
no ``=``, empty segments, stray whitespace and several ``Cookie`` fields.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import (
    given,
    settings,
)
from hypothesis import strategies as st
from starlette.requests import (
    Request,
    cookie_parser,
)
from starlette.types import Scope

from registry.auth.session_cookie import (
    SessionCookieIsolationMiddleware,
    request_session_cookie,
)
from registry.common.session_cookie import (
    join_cookie_fields,
    single_session_cookie,
)
from registry.core.config import settings as registry_settings

pytestmark = [pytest.mark.unit, pytest.mark.auth]

_NAME: str = registry_settings.session_cookie_name
# Names a careless matcher would take for the session cookie.
_LOOK_ALIKES: tuple[str, ...] = (
    f"old_{_NAME}",
    f"{_NAME}_x",
    _NAME.upper(),
    _NAME[:-1],
    f"__Host-{_NAME}",
    f"{_NAME}\x00",
)
_OTHER_NAMES: tuple[str, ...] = ("theme", "egress_consent_abc", "a", "")
# Whitespace Python's str.strip removes, including the non-ASCII latin-1 ones.
_WHITESPACE = st.text(alphabet=" \t\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0", max_size=3)
# Any latin-1 character except ";", which ends the pair.
_VALUES = st.text(
    alphabet=st.characters(codec="latin-1", exclude_characters=";"),
    max_size=12,
)


@st.composite
def _pair(
    draw: st.DrawFn,
) -> str:
    """Draw one ``name=value`` segment, padded with whitespace around the name."""
    name = draw(st.sampled_from((_NAME, *_LOOK_ALIKES, *_OTHER_NAMES)))
    return f"{draw(_WHITESPACE)}{name}{draw(_WHITESPACE)}={draw(_VALUES)}"


_SEGMENTS = st.one_of(
    _pair(),
    _pair(),
    _VALUES,
    st.just(_NAME),
    st.just(""),
    _WHITESPACE,
)
_FIELDS = st.lists(
    st.lists(_SEGMENTS, max_size=6).map(";".join),
    min_size=1,
    max_size=4,
)
_OTHER_HEADERS: list[tuple[bytes, bytes]] = [
    (b"host", b"registry.example.com"),
    (b"user-agent", b"test"),
]


def _run_middleware(
    headers: list[tuple[bytes, bytes]],
) -> Scope:
    """Run the isolation middleware on an HTTP scope; return the scope the app received."""
    received: list[Scope] = []

    async def _app(scope: Scope, receive: object, send: object) -> None:
        received.append(scope)

    scope: Scope = {"type": "http", "method": "GET", "path": "/", "headers": headers}
    asyncio.run(SessionCookieIsolationMiddleware(_app)(scope, None, None))
    assert len(received) == 1
    return received[0]


def _without_name(
    parsed: dict[str, str],
) -> dict[str, str]:
    """Return a parsed cookie dict without the session cookie name."""
    return {key: value for key, value in parsed.items() if key != _NAME}


@settings(max_examples=200, derandomize=True, database=None, report_multiple_bugs=False)
@given(fields=_FIELDS, cookie_first=st.booleans())
def test_stripping_hides_the_session_cookie_and_keeps_every_other_pair(fields, cookie_first):
    """Property: no session cookie after stripping, every other pair intact, cache matches."""
    cookie_headers = [(b"cookie", field.encode("latin-1")) for field in fields]
    headers = cookie_headers + _OTHER_HEADERS if cookie_first else _OTHER_HEADERS + cookie_headers

    scope = _run_middleware(list(headers))

    request = Request(scope)
    assert _NAME not in request.cookies, scope["headers"]

    stripped_fields = [
        value.decode("latin-1") for key, value in scope["headers"] if key == b"cookie"
    ]
    expected = [_without_name(cookie_parser(field)) for field in fields]
    actual = [cookie_parser(field) for field in stripped_fields]
    assert [parsed for parsed in actual if parsed] == [parsed for parsed in expected if parsed]

    other_headers = [header for header in scope["headers"] if header[0] != b"cookie"]
    assert other_headers == _OTHER_HEADERS

    assert request_session_cookie(request) == single_session_cookie(
        join_cookie_fields(fields), _NAME
    )
