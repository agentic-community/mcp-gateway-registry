"""Static/favicon nginx locations must be ROOT_PATH-prefixed like other routes.

Issue #1660: in path-based routing mode (ROOT_PATH set, e.g. ``/registry``),
the frontend's rewritten HTML requests assets at ``{ROOT_PATH}/static/...``
and ``{ROOT_PATH}/favicon.ico``. The nginx locations serving those assets
directly via ``alias`` were bare ``/static/`` and ``/favicon.ico``, so a
ROOT_PATH-prefixed request never matched them; it fell through to the
catch-all ``location /``, which proxies the unstripped path to the FastAPI
backend and gets back the SPA's cached index.html instead of the real asset.
Every other backend route in these templates already uses a
``{{ROOT_PATH}}``-prefixed location for exactly this reason.
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
    """Split a config into its top-level ``server { ... }`` blocks."""
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
def test_every_server_block_prefixes_static_location(template: str) -> None:
    text = (_DOCKER_DIR / template).read_text(encoding="utf-8")

    blocks = _server_blocks(text)
    assert blocks, f"no server blocks found in {template}"

    for block in blocks:
        listen = re.search(r"^\s*listen ([^;]+);", block, re.MULTILINE)
        listen_desc = listen.group(1) if listen else "unknown"
        assert re.search(r"^\s*location \{\{ROOT_PATH\}\}/static/ \{", block, re.MULTILINE), (
            f"{template}: server block listening on {listen_desc!r} does not "
            "prefix its /static/ location with {{ROOT_PATH}}, so a path-mode "
            "request for a static asset falls through to the SPA fallback (#1660)"
        )


@pytest.mark.parametrize("template", _TEMPLATES)
def test_every_server_block_prefixes_favicon_location(template: str) -> None:
    text = (_DOCKER_DIR / template).read_text(encoding="utf-8")

    blocks = _server_blocks(text)
    assert blocks, f"no server blocks found in {template}"

    for block in blocks:
        listen = re.search(r"^\s*listen ([^;]+);", block, re.MULTILINE)
        listen_desc = listen.group(1) if listen else "unknown"
        assert re.search(
            r"^\s*location = \{\{ROOT_PATH\}\}/favicon\.ico \{", block, re.MULTILINE
        ), (
            f"{template}: server block listening on {listen_desc!r} does not "
            "prefix its /favicon.ico location with {{ROOT_PATH}}, so a path-mode "
            "request for it falls through to the SPA fallback (#1660)"
        )


@pytest.mark.parametrize("template", _TEMPLATES)
def test_bare_static_and_favicon_locations_are_gone(template: str) -> None:
    """A later edit re-introducing the unprefixed form would silently regress this."""
    text = (_DOCKER_DIR / template).read_text(encoding="utf-8")

    assert not re.search(r"^\s*location /static/ \{", text, re.MULTILINE)
    assert not re.search(r"^\s*location = /favicon\.ico \{", text, re.MULTILINE)
