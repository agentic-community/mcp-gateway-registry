"""The front-door nginx templates must not emit absolute redirects.

Issue #1606: the #1501 fix made generated location blocks trailing-slash
terminated, so a request to a registered MCP server *without* the trailing
slash now hits nginx's automatic "add the slash" 301. With ``absolute_redirect``
at its nginx default (on), that Location is rebuilt from the listener that
served the request -- ``http://`` and the internal container port 8080/8443 --
which is not routable from outside the cluster, and TLS terminates upstream
anyway. MCP clients follow the redirect and hang until they time out.

``absolute_redirect off`` makes nginx send a relative Location, so the client
stays on the origin it dialed. It has to be set in every server block: the
internal port is baked into the redirect by whichever listener answered.
"""

import re
from pathlib import Path

import pytest

_DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"
_TEMPLATES = (
    "nginx_rev_proxy_http_and_https.conf",
    "nginx_rev_proxy_http_only.conf",
)


def _server_blocks(text: str) -> list[str]:
    """Split a config into its top-level ``server { ... }`` blocks.

    Top-level only: the templates nest ``location``/``if`` blocks several levels
    deep, so this brace-counts from each ``server {`` at column 0 rather than
    trying to match with a regex.
    """
    blocks = []
    for match in re.finditer(r"^server \{", text, re.MULTILINE):
        depth = 0
        for index in range(match.start(), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(text[match.start() : index + 1])
                    break
    return blocks


@pytest.mark.parametrize("template", _TEMPLATES)
def test_every_server_block_disables_absolute_redirect(template: str) -> None:
    text = (_DOCKER_DIR / template).read_text(encoding="utf-8")

    blocks = _server_blocks(text)
    assert blocks, f"no server blocks found in {template}"

    for block in blocks:
        listen = re.search(r"^\s*listen ([^;]+);", block, re.MULTILINE)
        listen_desc = listen.group(1) if listen else "unknown"
        assert re.search(r"^\s*absolute_redirect off;", block, re.MULTILINE), (
            f"{template}: server block listening on {listen_desc!r} does not set "
            "'absolute_redirect off', so nginx's automatic trailing-slash 301 "
            "would leak the internal scheme and port to MCP clients (#1606)"
        )


@pytest.mark.parametrize("template", _TEMPLATES)
def test_absolute_redirect_is_never_re_enabled(template: str) -> None:
    """A later ``absolute_redirect on`` would silently undo this per-scope."""
    text = (_DOCKER_DIR / template).read_text(encoding="utf-8")

    assert not re.search(r"^\s*absolute_redirect\s+on;", text, re.MULTILINE)
