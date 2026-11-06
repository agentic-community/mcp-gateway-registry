"""Property: CSRF never skips a request that carries the session cookie.

``verify_csrf_token_flexible`` and ``verify_csrf_token_header_only`` skip the
check only for a request with no session cookie at all (a bearer-token
client). A request that carries the cookie must either present a CSRF token
for the session the shared reader resolved, or be refused. The generated
requests carry zero to three copies of the session cookie, each valid,
signed for an unknown or expired session, signed too long ago, unsigned,
signed over a legacy dict payload, empty, or naming a session whose store
read raises, beside other cookies and with or without a CSRF token.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from hypothesis import (
    given,
    settings,
)
from hypothesis import strategies as st
from itsdangerous import TimestampSigner, URLSafeTimedSerializer
from starlette.requests import Request

import registry.auth.session_store as session_store
from registry.auth.csrf import (
    generate_csrf_token,
    verify_csrf_token_flexible,
    verify_csrf_token_header_only,
)
from registry.core.config import settings as registry_settings

pytestmark = [pytest.mark.unit, pytest.mark.auth]

_NAME: str = registry_settings.session_cookie_name
_VALID_SESSION_ID: str = "sid-valid"
_UNKNOWN_SESSION_ID: str = "sid-unknown"
_EXPIRED_SESSION_ID: str = "sid-expired"
_RAISING_SESSION_ID: str = "sid-raises"
_SIGNER = URLSafeTimedSerializer(registry_settings.secret_key)


def _signed_long_ago(
    payload: Any,
) -> str:
    """Sign ``payload`` with a timestamp older than the session lifetime."""
    issued = int(time.time()) - registry_settings.session_max_age_seconds - 60
    with patch.object(TimestampSigner, "get_timestamp", return_value=issued):
        return _SIGNER.dumps(payload)


# Every kind of session cookie copy a browser or attacker can send.
_COPIES: dict[str, str] = {
    "valid": _SIGNER.dumps(_VALID_SESSION_ID),
    "unknown_session": _SIGNER.dumps(_UNKNOWN_SESSION_ID),
    "expired_session": _SIGNER.dumps(_EXPIRED_SESSION_ID),
    "store_raises": _SIGNER.dumps(_RAISING_SESSION_ID),
    "signed_too_long_ago": _signed_long_ago(_VALID_SESSION_ID),
    "unsigned": "not-a-signed-value",
    "legacy_dict": _SIGNER.dumps({"username": "alice"}),
    "empty": "",
}
_TOKENS: dict[str, str | None] = {
    "none": None,
    "for_session": generate_csrf_token(_VALID_SESSION_ID),
    "for_other_session": generate_csrf_token(_UNKNOWN_SESSION_ID),
    "garbage": "garbage-token",
}


class _Collection:
    """The session store collection: one live session, one expired, one failing read."""

    async def find_one(
        self,
        query: dict[str, str],
    ) -> dict[str, Any] | None:
        session_id = query["session_id"]
        if session_id == _RAISING_SESSION_ID:
            raise ConnectionError("store unavailable")
        if session_id == _VALID_SESSION_ID:
            return {
                "session_id": session_id,
                "username": "alice",
                "expires_at": datetime.now(UTC) + timedelta(hours=1),
            }
        if session_id == _EXPIRED_SESSION_ID:
            return {
                "session_id": session_id,
                "username": "alice",
                "expires_at": datetime.now(UTC) - timedelta(seconds=1),
            }
        return None


def _request(
    copies: list[str],
    other_cookies: bool,
    token: str | None,
) -> Request:
    """Build a POST request carrying ``copies`` of the session cookie and ``token``."""
    pairs = [f"{_NAME}={_COPIES[copy]}" for copy in copies]
    if other_cookies:
        pairs = ["theme=dark", *pairs, "egress_consent_abc=x"]
    headers: list[tuple[bytes, bytes]] = [(b"host", b"registry.example.com")]
    if pairs:
        headers.append((b"cookie", "; ".join(pairs).encode("latin-1")))
    if token is not None:
        headers.append((b"x-csrf-token", token.encode("latin-1")))
    scope = {"type": "http", "method": "POST", "path": "/", "headers": headers}
    return Request(scope)


def _outcome(
    check: Any,
    request: Request,
) -> str:
    """Run a CSRF dependency; return "passed" or "refused" (a 403)."""
    with patch.object(session_store, "_collection", _Collection()):
        try:
            asyncio.run(check(request))
        except HTTPException as exc:
            assert exc.status_code == 403, exc
            return "refused"
    return "passed"


@settings(max_examples=200, derandomize=True, database=None, report_multiple_bugs=False)
@given(
    copies=st.lists(st.sampled_from(sorted(_COPIES)), max_size=3),
    other_cookies=st.booleans(),
    token=st.sampled_from(sorted(_TOKENS)),
    check=st.sampled_from((verify_csrf_token_flexible, verify_csrf_token_header_only)),
)
def test_csrf_never_skips_a_request_that_carries_the_session_cookie(
    copies,
    other_cookies,
    token,
    check,
):
    """Property: with a session cookie, CSRF passes only on the resolved session's token."""
    request = _request(copies, other_cookies, _TOKENS[token])

    outcome = _outcome(check, request)

    if not copies:
        assert outcome == "passed"
    elif copies == ["valid"] and token == "for_session":
        assert outcome == "passed"
    else:
        assert outcome == "refused", (copies, token)
