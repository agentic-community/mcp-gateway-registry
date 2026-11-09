"""The base nginx configs return the JSON-RPC error body for a refused MCP
request (issue #2006)."""

from pathlib import Path

import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
BASE_CONFIGS: tuple[str, ...] = (
    "docker/nginx_rev_proxy_http_only.conf",
    "docker/nginx_rev_proxy_http_and_https.conf",
)


def _forbidden_blocks(
    text: str,
) -> list[str]:
    """Return the body of every @forbidden_error location."""
    blocks = []
    for part in text.split("location @forbidden_error {")[1:]:
        blocks.append(part.split("\n    }\n", 1)[0])
    return blocks


@pytest.mark.parametrize("config", BASE_CONFIGS)
def test_declares_an_empty_default_for_the_error_body(config):
    text = (PROJECT_ROOT / config).read_text()

    assert 'map $host $mcp_error_body {\n    default "";\n}' in text


@pytest.mark.parametrize("config", BASE_CONFIGS)
def test_forbidden_error_returns_the_body_after_the_throttle_check(config):
    blocks = _forbidden_blocks((PROJECT_ROOT / config).read_text())

    assert blocks
    for block in blocks:
        throttle = block.index('if ($rl_throttled = "1")')
        body = block.index('if ($mcp_error_body != "") {\n            return 403 $mcp_error_body;')
        fallback = block.index('return 403 \'{"error": "Access forbidden"}\';')
        assert throttle < body < fallback
