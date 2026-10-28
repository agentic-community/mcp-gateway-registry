"""JWT-size guard for the registry-UI internal token (kiro review item #2).

Binding groups into the token reintroduces the large-group-membership pressure that
#971 (server-side OAuth session store) solved for cookies. Session-backed callers
carry no groups in the token (``_attach_registry_ui_token`` drops them because the
registry resolves them from the session store; see
tests/auth_server/unit/test_registry_ui_token_attach.py). Bearer/static-token
callers carry groups in the claim, so their token grows with group count. This
test records the size at 50/200/500 groups so a regression (or an unexpectedly
large machine identity) is visible, and documents that the /validate
proxy_buffer_size in the nginx templates must accommodate the upper bound.
"""

import os
from unittest.mock import patch

import pytest

from auth_server.internal_request_token import (
    INTERNAL_TOKEN_MAX_BYTES,
    mint_registry_ui_token,
)

_SECRET = "size-benchmark-secret-key"

# Two different numbers bound this token. nginx reads the /validate response into
# the 64 KiB `location = /validate` proxy_buffer_size, and the auth server refuses
# to mint above INTERNAL_TOKEN_MAX_BYTES (40 KiB) to leave room for the other
# auth headers. Separately, a bearer whose groups arrive inline is limited by the
# 32 KiB inbound header cap, which yields a registry token near 32 KiB at most;
# 500 realistic groups should stay under that inline bound.
_INLINE_BEARER_TOKEN_BOUND = 32 * 1024


@pytest.fixture(autouse=True)
def _secret_env():
    with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
        yield


def _groups(n: int) -> list[str]:
    # Realistic-ish group names (IdP groups are often long DN-like strings).
    return [f"group-{i:04d}-some-realistic-org-unit-name" for i in range(n)]


class TestBearerTokenSizeBounds:
    @pytest.mark.parametrize("n,expected_under", [(50, 4 * 1024), (200, 12 * 1024)])
    def test_bearer_token_size_within_documented_bound(self, n: int, expected_under: int) -> None:
        token = mint_registry_ui_token(
            subject="svc-account",
            session_id="",
            groups=_groups(n),
            auth_method="network-trusted",
            client_id="key-1",
        )
        size = len(token)
        # Record the size so a regression is visible in the assertion message.
        assert size < expected_under, (
            f"{n}-group bearer token is {size} bytes (expected < {expected_under}). "
            f"If real machine identities approach this many groups, raise the "
            f"`location = /validate` proxy_buffer_size in the nginx templates, "
            f"INTERNAL_TOKEN_MAX_BYTES, and this bound."
        )

    def test_500_groups_documents_upper_bound(self) -> None:
        # 500 groups is well beyond a realistic machine identity; this asserts the
        # token stays under the inline-bearer bound, below the mint cap.
        token = mint_registry_ui_token(
            subject="svc-account",
            session_id="",
            groups=_groups(500),
            auth_method="network-trusted",
            client_id="key-1",
        )
        assert len(token) < _INLINE_BEARER_TOKEN_BOUND < INTERNAL_TOKEN_MAX_BYTES, (
            f"500-group token is {len(token)} bytes, over the {_INLINE_BEARER_TOKEN_BOUND}-byte "
            f"inline-bearer bound. A bearer/static caller with this many groups needs either a "
            f"larger `location = /validate` proxy_buffer_size and INTERNAL_TOKEN_MAX_BYTES, "
            f"or the #971 server-side claim-store pattern."
        )
