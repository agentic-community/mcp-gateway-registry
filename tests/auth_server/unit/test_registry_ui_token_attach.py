"""Unit tests for _attach_registry_ui_token, the registry /api/ hop token mint.

- It mints only when nginx force-set the marker secret alongside
  X-Registry-Api-Auth, so a direct auth-server caller gets no token.
- A session-backed caller's token carries no groups (the registry resolves
  them from the session store); a bearer caller's token carries them all
  (issue #1939).
- A group set too large for nginx's /validate header buffer fails /validate
  with a 500 instead of minting.
- A /validate response whose headers would overflow that buffer is refused.
"""

import os
from unittest.mock import patch

import jwt as pyjwt
import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from auth_server.server import (  # noqa: E402
    _attach_registry_ui_token,
    _enforce_validate_header_budget,
    settings,
)

pytestmark = pytest.mark.unit

_SECRET = "test-secret-key-that-is-definitely-long-enough-32b"
_GROUPS = ["team-a", "finance-visibility-only"]
# 1,000 GUID-sized groups mint a token of about 53 KB, over the 40 KiB limit.
_OVERSIZED_GROUPS = [f"00000000-0000-4000-8000-{i:012d}" for i in range(1000)]


class _Req:
    def __init__(self, headers):
        self.headers = headers


class _Resp:
    def __init__(self):
        self.headers = {}


def _attach(
    headers: dict[str, str],
    session_id: str = "",
    groups: list[str] = _GROUPS,
) -> _Resp:
    resp = _Resp()
    with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
        _attach_registry_ui_token(
            _Req(headers),
            resp,
            subject="alice",
            session_id=session_id,
            groups=list(groups),
            auth_method="keycloak",
            client_id="cid",
        )
    return resp


def _nginx_headers() -> dict[str, str]:
    return {
        "X-Registry-Api-Auth": "1",
        "X-Validate-Source-Secret": settings.auth_server_nginx_marker_secret,
    }


def _claims(resp: _Resp) -> dict:
    token = resp.headers["X-Internal-Token-Registry"]
    return pyjwt.decode(token, options={"verify_signature": False})


@pytest.mark.parametrize("marker", [None, "wrong-marker-secret"])
def test_no_mint_when_nginx_marker_missing_or_mismatched(marker):
    headers = {"X-Registry-Api-Auth": "1"}
    if marker is not None:
        headers["X-Validate-Source-Secret"] = marker

    resp = _attach(headers)

    assert "X-Internal-Token-Registry" not in resp.headers


def test_bearer_caller_token_carries_all_groups():
    resp = _attach(_nginx_headers())

    assert _claims(resp)["groups"] == _GROUPS


def test_session_caller_token_stays_small_whatever_the_group_count():
    resp = _attach(_nginx_headers(), session_id="sess-123", groups=_OVERSIZED_GROUPS)

    claims = _claims(resp)
    assert (claims["session_id"], claims["groups"]) == ("sess-123", [])
    assert len(resp.headers["X-Internal-Token-Registry"]) < 1024


def test_oversized_bearer_groups_fail_validate_with_500():
    """A token too large for nginx's buffer is a server-side limit, so /validate
    answers 500 instead of omitting the token (which the registry reports as 401)."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        _attach(_nginx_headers(), groups=_OVERSIZED_GROUPS)

    assert exc_info.value.status_code == 500


def test_validate_response_over_the_header_budget_is_refused():
    from fastapi.responses import JSONResponse

    response = JSONResponse({"valid": True})
    response.headers["X-Scopes"] = "s" * (40 * 1024)
    response.headers["X-Internal-Token"] = "t" * (30 * 1024)

    result = _enforce_validate_header_budget(response, "alice")

    assert (result.status_code, "X-Scopes" in result.headers) == (500, False)


def test_validate_response_within_the_header_budget_is_returned_unchanged():
    from fastapi.responses import JSONResponse

    response = JSONResponse({"valid": True})
    response.headers["X-Scopes"] = "s" * (40 * 1024)

    assert _enforce_validate_header_budget(response, "alice") is response
