"""Static-template invariant: every auth_request location carries a 401 handler.

The gateway's OAuth discovery contract is that EVERY 401 produced by a data
plane location protected by ``auth_request /validate`` carries the RFC 9728
``WWW-Authenticate: ... resource_metadata=...`` challenge, so a spec-strict MCP
client can bootstrap authorization from any authenticated surface.

Since gympass/mcp-gateway-chart#931 the auth-server ``/validate`` subrequest no
longer sends a ``WWW-Authenticate`` of its own (it is internal; the copied
header used to duplicate the error-page challenge). The client-visible
challenge is therefore produced ONLY by the location's ``error_page 401 = @...``
handler, and the named locations (@auth_error / @ard_auth_error) set it via
``more_set_headers``. Without the directive a failed /validate yields nginx's
default headerless 401, the exact regression this test pins: the http_only
``{{ROOT_PATH}}/api/`` location lost its challenge exactly this way when the
subrequest copy was removed.

These tests parse the raw .conf templates (no nginx binary needed, same style
as test_nginx_generic_markers.py). They fail loudly at CI if a new
auth_request location is added without a 401 handler, if an existing handler
is dropped, or if a named handler stops emitting WWW-Authenticate.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"
_TEMPLATES = (
    "nginx_rev_proxy_http_only.conf",
    "nginx_rev_proxy_http_and_https.conf",
)

# Directives always start their own line in these templates, so anchoring on
# ^\s* keeps prose comments (which mention auth_request /validate in passing)
# from reading as real directives.
_AUTH_REQUEST = re.compile(r"^\s*auth_request\s+/validate\s*;", re.MULTILINE)
_ERROR_PAGE_401 = re.compile(r"^\s*error_page\s+401\s*=\s*(@[A-Za-z0-9_-]+)\s*;", re.MULTILINE)
_WWW_AUTHENTICATE = re.compile(r"WWW-Authenticate", re.MULTILINE)


def _extract_location_blocks(conf_text: str) -> list[tuple[str, str]]:
    """Return (location-header-line, brace-matched-body) for every location.

    The block opener is the LAST ``{`` on a ``location`` line, because the
    header itself carries template placeholders (``{{ROOT_PATH}}``) whose
    braces are balanced pairs. Brace matching then nests, so
    ``set_by_lua_block``/``if`` bodies stay inside their enclosing location and
    ``{{PLACEHOLDER}}`` pairs keep depth balanced. Named locations
    (``location @name {``) are included: they are the error_page targets this
    test must resolve.
    """
    blocks: list[tuple[str, str]] = []
    for match in re.finditer(r"^[ \t]*location[ \t]+.*\{[ \t]*$", conf_text, re.MULTILINE):
        line = match.group(0)
        open_index = match.start() + line.rstrip().rfind("{")
        depth = 1
        index = open_index + 1
        while index < len(conf_text) and depth > 0:
            char = conf_text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            index += 1
        header = line.rstrip()[: line.rstrip().rfind("{")].strip()
        blocks.append((header, conf_text[open_index + 1 : index - 1]))
    return blocks


def _named_location_body(conf_text: str, name: str) -> str | None:
    """Body of ``location @name { ... }``, or None when the block is absent."""
    for header, body in _extract_location_blocks(conf_text):
        if header.split()[-1].rstrip("{").strip() == name:
            return body
    return None


def _auth_request_locations(conf_text: str) -> list[tuple[str, str]]:
    """Location blocks (header, body) that gate on ``auth_request /validate``."""
    return [
        (header, body)
        for header, body in _extract_location_blocks(conf_text)
        if _AUTH_REQUEST.search(body)
    ]


@pytest.fixture
def http_only_text():
    return (_DOCKER_DIR / _TEMPLATES[0]).read_text()


@pytest.fixture
def http_and_https_text():
    return (_DOCKER_DIR / _TEMPLATES[1]).read_text()


class TestEveryAuthRequestLocationHasChallenge:
    """Invariant: every ``auth_request /validate`` location renders an
    ``error_page 401 = @handler`` whose named location sets WWW-Authenticate.

    A location dropping the directive, or pointing it at a handler without the
    header, silently ships headerless 401s on that surface (MCP clients cannot
    discover the PRM and hang at login).
    """

    @pytest.mark.parametrize(
        "conf_fixture",
        ["http_only_text", "http_and_https_text"],
    )
    def test_all_auth_request_locations_declare_error_page_401(self, conf_fixture, request):
        conf_text = request.getfixturevalue(conf_fixture)
        blocks = _auth_request_locations(conf_text)
        assert blocks, f"{conf_fixture}: no auth_request /validate locations found"
        for header, body in blocks:
            handlers = _ERROR_PAGE_401.findall(body)
            assert handlers, (
                f"{conf_fixture}: location {header!r} runs auth_request /validate "
                "but declares no `error_page 401 = @...` handler: since #931 the "
                "/validate subrequest sends no WWW-Authenticate, so this surface "
                "would answer a headerless 401"
            )

    @pytest.mark.parametrize(
        "conf_fixture",
        ["http_only_text", "http_and_https_text"],
    )
    def test_401_handlers_emit_www_authenticate(self, conf_fixture, request):
        conf_text = request.getfixturevalue(conf_fixture)
        checked: set[str] = set()
        for header, body in _auth_request_locations(conf_text):
            for target in _ERROR_PAGE_401.findall(body):
                if target in checked:
                    continue
                checked.add(target)
                handler = _named_location_body(conf_text, target)
                assert handler is not None, (
                    f"{conf_fixture}: location {header!r} error-pages 401 to "
                    f"{target!r}, which does not exist in this template"
                )
                assert _WWW_AUTHENTICATE.search(handler), (
                    f"{conf_fixture}: named location {target!r} must set a "
                    "WWW-Authenticate challenge (RFC 9728 §5.1), otherwise every "
                    "401 routed to it is headerless"
                )
        assert checked, f"{conf_fixture}: no error_page 401 handler targets found"

    def test_http_only_covers_the_api_location(self, http_only_text):
        """Pin the specific #931 regression: the http_only {{ROOT_PATH}}/api/
        location (no trailing-slash-less variant) must keep its 401 handler."""
        api_blocks = [
            (header, body)
            for header, body in _auth_request_locations(http_only_text)
            if header.endswith("{{ROOT_PATH}}/api/")
        ]
        assert len(api_blocks) == 1, "expected exactly one http_only /api/ auth location"
        assert _ERROR_PAGE_401.search(api_blocks[0][1]), (
            "http_only /api/ lost its error_page 401 handler (the #931 regression)"
        )
