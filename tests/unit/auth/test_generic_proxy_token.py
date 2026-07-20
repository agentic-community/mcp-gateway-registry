"""Unit tests for the generic-proxy internal token (auth_server/internal_request_token).

Covers mint round-trip (the TWO-claim shape: entity_type + full registered path)
and verify_generic_proxy_token's accept/reject paths, with emphasis on the two
security properties that distinguish it from the mcp-proxy token:
- cross-type replay is rejected (a skill token can't be used on an /a2a_agent/ route);
- the path guard binds the FULL multi-segment registered path and allows a
  sub-path UNDER it but NOT a sibling (skills/proxy-demo vs skills/proxy-demo-evil).
"""

import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import jwt as pyjwt
import pytest
from fastapi import HTTPException

from auth_server.internal_request_token import (
    GENERIC_PROXY_AUDIENCE,
    GENERIC_PROXY_TOKEN_USE,
    MCP_PROXY_AUDIENCE,
    _mint_internal_token,
    mint_generic_proxy_token,
    verify_generic_proxy_token,
)

_SECRET = "test-secret-key-for-testing-only"


def _request(headers, entity_type="skill", entity_path="skills/proxy-demo"):
    lower = {k.lower(): v for k, v in headers.items()}
    return SimpleNamespace(
        headers=SimpleNamespace(get=lambda k, default=None: lower.get(k.lower(), default)),
        path_params={"entity_type": entity_type, "entity_path": entity_path},
        state=SimpleNamespace(),
    )


@pytest.fixture(autouse=True)
def _secret_env():
    with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
        yield


def _token(
    entity_type="skill",
    registered_path="skills/proxy-demo",
    upstream="https://backend.example/",
    subject="alice",
):
    return mint_generic_proxy_token(
        subject=subject,
        scopes=["s/read"],
        entity_type=entity_type,
        registered_path=registered_path,
        upstream_url=upstream,
    )


class TestMint:
    def test_binds_entity_type_and_full_path_as_two_claims(self):
        tok = _token(entity_type="skill", registered_path="skills/proxy-demo")
        claims = pyjwt.decode(tok, _SECRET, algorithms=["HS256"], audience=GENERIC_PROXY_AUDIENCE)
        assert claims["entity_type"] == "skill"
        assert claims["server"] == "skills/proxy-demo"  # FULL path, not first segment
        assert claims["upstream_url"] == "https://backend.example/"
        assert claims["token_use"] == GENERIC_PROXY_TOKEN_USE
        assert claims["aud"] == GENERIC_PROXY_AUDIENCE

    def test_empty_subject_refused(self):
        with pytest.raises(ValueError):
            _token(subject="")

    def test_streaming_claim_defaults_false(self):
        tok = _token()
        claims = pyjwt.decode(tok, _SECRET, algorithms=["HS256"], audience=GENERIC_PROXY_AUDIENCE)
        assert claims["streaming"] is False

    def test_streaming_claim_bound_when_true(self):
        # The hop reads this SIGNED claim (never a forgeable header) to decide
        # whether to stream, so minting must carry the opt-in verbatim.
        tok = mint_generic_proxy_token(
            subject="alice",
            scopes=["s/read"],
            entity_type="custom-llm",
            registered_path="custom-llm/chat",
            upstream_url="https://llm.example/",
            streaming=True,
        )
        claims = pyjwt.decode(tok, _SECRET, algorithms=["HS256"], audience=GENERIC_PROXY_AUDIENCE)
        assert claims["streaming"] is True


class TestVerifyAccept:
    async def test_exact_path_accepted(self):
        tok = _token()
        req = _request({"X-Internal-Token-Generic": tok})
        await verify_generic_proxy_token(req)  # no raise
        assert req.state.generic_proxy_claims["server"] == "skills/proxy-demo"

    async def test_subpath_under_bound_accepted(self):
        tok = _token(registered_path="skills/proxy-demo")
        req = _request({"X-Internal-Token-Generic": tok}, entity_path="skills/proxy-demo/sub/x")
        await verify_generic_proxy_token(req)  # sub-resource of the bound entity


class TestVerifyReject:
    async def test_missing_token(self):
        with pytest.raises(HTTPException) as e:
            await verify_generic_proxy_token(_request({}))
        assert e.value.status_code == 401

    async def test_wrong_audience_mcp_token_rejected(self):
        # An mcp-proxy-audience token must not verify on the generic hop.
        mcp_tok = _mint_internal_token(
            audience=MCP_PROXY_AUDIENCE,
            subject="alice",
            scopes=[],
            extra_claims={
                "server": "skills/proxy-demo",
                "upstream_url": "https://b/",
                "entity_type": "skill",
                "token_use": "mcp-proxy",
            },
        )
        with pytest.raises(HTTPException) as e:
            await verify_generic_proxy_token(_request({"X-Internal-Token-Generic": mcp_tok}))
        assert e.value.status_code == 401

    async def test_wrong_token_use_rejected(self):
        tok = _mint_internal_token(
            audience=GENERIC_PROXY_AUDIENCE,
            subject="alice",
            scopes=[],
            extra_claims={
                "server": "skills/proxy-demo",
                "upstream_url": "https://b/",
                "entity_type": "skill",
                "token_use": "something-else",
            },
        )
        with pytest.raises(HTTPException) as e:
            await verify_generic_proxy_token(_request({"X-Internal-Token-Generic": tok}))
        assert e.value.status_code == 401

    async def test_missing_upstream_rejected(self):
        tok = _mint_internal_token(
            audience=GENERIC_PROXY_AUDIENCE,
            subject="alice",
            scopes=[],
            extra_claims={
                "server": "skills/proxy-demo",
                "entity_type": "skill",
                "token_use": GENERIC_PROXY_TOKEN_USE,
            },
        )
        with pytest.raises(HTTPException) as e:
            await verify_generic_proxy_token(_request({"X-Internal-Token-Generic": tok}))
        assert e.value.status_code == 401

    async def test_expired_rejected(self):
        # Mint with the clock far in the past so exp is already elapsed when the
        # verifier (which uses the real clock) decodes it.
        with patch(
            "auth_server.internal_request_token.time.time", return_value=time.time() - 10000
        ):
            tok = _token()
        with pytest.raises(HTTPException) as e:
            await verify_generic_proxy_token(_request({"X-Internal-Token-Generic": tok}))
        assert e.value.status_code == 401

    async def test_cross_type_replay_rejected(self):
        """A token bound to (skill, skills/proxy-demo) must NOT verify on an
        /a2a_agent/ route with the same path."""
        tok = _token(entity_type="skill", registered_path="skills/proxy-demo")
        req = _request(
            {"X-Internal-Token-Generic": tok},
            entity_type="a2a_agent",
            entity_path="skills/proxy-demo",
        )
        with pytest.raises(HTTPException) as e:
            await verify_generic_proxy_token(req)
        assert e.value.status_code == 401

    async def test_sibling_path_rejected(self):
        """skills/proxy-demo token must NOT authorize skills/proxy-demo-evil
        (prefix match must be on a segment boundary)."""
        tok = _token(registered_path="skills/proxy-demo")
        req = _request({"X-Internal-Token-Generic": tok}, entity_path="skills/proxy-demo-evil")
        with pytest.raises(HTTPException) as e:
            await verify_generic_proxy_token(req)
        assert e.value.status_code == 401

    async def test_different_entity_path_rejected(self):
        tok = _token(registered_path="skills/proxy-demo")
        req = _request({"X-Internal-Token-Generic": tok}, entity_path="skills/other")
        with pytest.raises(HTTPException) as e:
            await verify_generic_proxy_token(req)
        assert e.value.status_code == 401

    async def test_tampered_signature_rejected(self):
        tok = _token()
        tampered = tok[:-4] + ("aaaa" if not tok.endswith("aaaa") else "bbbb")
        with pytest.raises(HTTPException) as e:
            await verify_generic_proxy_token(_request({"X-Internal-Token-Generic": tampered}))
        assert e.value.status_code == 401

    async def test_dotdot_segment_rejected(self):
        """A dot-dot escape under the bound prefix must be rejected even though it
        literally starts with '<bound>/'."""
        tok = _token(registered_path="skills/proxy-demo")
        req = _request(
            {"X-Internal-Token-Generic": tok},
            entity_path="skills/proxy-demo/../../secret",
        )
        with pytest.raises(HTTPException) as e:
            await verify_generic_proxy_token(req)
        assert e.value.status_code == 401


class TestClaimOverrideProtection:
    def test_extra_claims_cannot_override_reserved_base_claims(self):
        """A caller passing a reserved key in extra_claims must NOT override the
        base claim (aud/exp/iss). Base claims win regardless of extra order."""
        tok = _mint_internal_token(
            audience=GENERIC_PROXY_AUDIENCE,
            subject="alice",
            scopes=[],
            extra_claims={
                "aud": "attacker-audience",  # attempt to hijack audience
                "iss": "attacker",
                "exp": 99999999999,  # attempt to extend lifetime
                "server": "skills/x",
                "upstream_url": "https://b/",
                "entity_type": "skill",
                "token_use": GENERIC_PROXY_TOKEN_USE,
            },
        )
        claims = pyjwt.decode(tok, _SECRET, algorithms=["HS256"], audience=GENERIC_PROXY_AUDIENCE)
        assert claims["aud"] == GENERIC_PROXY_AUDIENCE  # base won, not "attacker-audience"
        assert claims["iss"] == "mcp-auth-server"
        assert claims["exp"] != 99999999999  # base exp won


class TestTtlKnob:
    def test_generic_ttl_falls_back_to_shared(self):
        from auth_server.internal_request_token import _generic_ttl_seconds, _ttl_seconds

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GENERIC_PROXY_TOKEN_TTL_SECONDS", None)
            assert _generic_ttl_seconds() == _ttl_seconds()

    def test_generic_ttl_own_value_and_floor(self):
        from auth_server.internal_request_token import _generic_ttl_seconds

        with patch.dict(os.environ, {"GENERIC_PROXY_TOKEN_TTL_SECONDS": "120"}, clear=False):
            assert _generic_ttl_seconds() == 120
        with patch.dict(os.environ, {"GENERIC_PROXY_TOKEN_TTL_SECONDS": "1"}, clear=False):
            assert _generic_ttl_seconds() == 5  # clamped to floor
