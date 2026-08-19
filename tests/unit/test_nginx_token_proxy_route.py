"""Guards that the gateway advertises a /oauth/token endpoint the nginx front
door actually routes to the auth-server -- not one that falls through to the
registry catch-all and 404s (regression guard for the token-proxy routing gap)."""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CONFS = [
    _ROOT / "docker" / "nginx_rev_proxy_http_and_https.conf",
    _ROOT / "docker" / "nginx_rev_proxy_http_only.conf",
]


@pytest.mark.parametrize("conf", _CONFS, ids=lambda p: p.name)
def test_oauth_token_routed_to_auth_server_in_every_server_block(conf):
    text = conf.read_text()
    # One auth-server /oauth2/callback/cognito block exists per server{} block;
    # the /oauth/token route must appear in each of them (both the HTTP and the
    # HTTPS server blocks of the http+https conf).
    server_blocks = text.count("location /oauth2/callback/cognito {")
    token_routes = text.count("location = /oauth/token {")
    assert server_blocks >= 1
    assert token_routes == server_blocks, (
        f"{conf.name}: /oauth/token routes ({token_routes}) != server blocks ({server_blocks})"
    )
    # ...and it must proxy to the auth-server upstream, never the registry.
    assert "proxy_pass http://{{AUTH_SERVER_HOST}}:{{AUTH_SERVER_PORT}}/oauth/token;" in text
