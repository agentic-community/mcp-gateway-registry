"""Guards that the gateway advertises a /oauth/token endpoint the nginx front
door actually routes to the auth-server -- not one that falls through to the
registry catch-all and 404s (regression guard for the token-proxy routing gap).

Verifies per location block (not just a bare count) that EVERY /oauth/token
route proxies to the auth-server and is not gated by auth_request, so a stray
route pointing elsewhere can't satisfy the invariant."""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CONFS = [
    _ROOT / "docker" / "nginx_rev_proxy_http_and_https.conf",
    _ROOT / "docker" / "nginx_rev_proxy_http_only.conf",
]
_AUTH_UPSTREAM = "proxy_pass http://{{AUTH_SERVER_HOST}}:{{AUTH_SERVER_PORT}}/oauth/token;"


def _oauth_token_location_bodies(text: str) -> list[str]:
    """Return the body of each `location = /oauth/token { ... }` block.

    Line-based (not regex) so the ``{{AUTH_SERVER_HOST}}`` template braces in the
    body can't confuse brace matching. Location bodies here have no nested blocks.
    """
    bodies: list[str] = []
    inside = False
    cur: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "location = /oauth/token {":
            inside, cur = True, []
        elif inside and stripped == "}":
            bodies.append("\n".join(cur))
            inside = False
        elif inside:
            cur.append(line)
    return bodies


@pytest.mark.parametrize("conf", _CONFS, ids=lambda p: p.name)
def test_oauth_token_routed_to_auth_server_in_every_server_block(conf):
    text = conf.read_text()
    # The cognito callback is an unconditional, exactly-once-per-server{} landmark
    # (unlike the conditionally-templated Keycloak/PingFederate blocks), so it is a
    # reliable per-server-block counter.
    server_blocks = text.count("location /oauth2/callback/cognito {")
    assert server_blocks >= 1

    bodies = _oauth_token_location_bodies(text)
    assert len(bodies) == server_blocks, (
        f"{conf.name}: /oauth/token routes ({len(bodies)}) != server blocks ({server_blocks})"
    )
    for body in bodies:
        assert _AUTH_UPSTREAM in body, f"{conf.name}: /oauth/token not routed to the auth-server"
        # It IS the token exchange (like /oauth2/callback/*), so it must not be
        # gated by an auth_request subrequest, and must forward the request body.
        assert "auth_request" not in body
        assert "proxy_pass_request_body on;" in body
