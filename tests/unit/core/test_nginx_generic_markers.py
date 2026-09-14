"""Template invariants for the generic-proxy markers in the nginx conf files.

The generic-hop mint discriminator + marker-spoof defense
depends on a set of nginx variables being (a) declared http-scope with an empty
default and (b) forwarded via ``proxy_set_header`` in EVERY ``location = /validate``
block. The LLD mandates a render-static assertion that the forwarded-header set is
identical across all three /validate blocks (one in http_only, two in
http_and_https) — otherwise a future refactor dropping an override from one block
would let a client spoof the marker to that block and mint a token bound to an
attacker-controlled upstream (SSRF).

These tests parse the raw .conf files (no nginx binary needed) and lock in that
invariant. They fail loudly at CI if a /validate block is added/edited without the
generic markers, or if a required http-scope map default is removed.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"
_HTTP_ONLY = _DOCKER_DIR / "nginx_rev_proxy_http_only.conf"
_HTTP_AND_HTTPS = _DOCKER_DIR / "nginx_rev_proxy_http_and_https.conf"

# The three generic markers that MUST be forwarded, sourced from the same-named
# nginx variable, in every /validate block.
_GENERIC_MARKER_FORWARDS = (
    "proxy_set_header X-Resolved-Generic-Upstream $generic_backend_url;",
    "proxy_set_header X-Generic-Proxy-Kind $generic_proxy_kind;",
    "proxy_set_header X-Entity-Path $entity_path;",
)

# The http-scope maps that must declare each marker variable with an empty
# default (so a request through a location that does NOT set them forwards "").
_GENERIC_MARKER_MAPS = (
    "map $host $generic_backend_url {",
    "map $host $generic_proxy_kind {",
    "map $host $entity_path {",
)


def _extract_validate_blocks(conf_text: str) -> list[str]:
    """Return the body text of each real ``location = /validate { ... }`` block.

    Brace-matched extraction (a comment mentioning the block in prose is not a
    real block and is skipped because it has no following ``{``).
    """
    blocks: list[str] = []
    for m in re.finditer(r"location\s+=\s+/validate\s*\{", conf_text):
        i = m.end()  # just past the opening brace
        depth = 1
        while i < len(conf_text) and depth > 0:
            if conf_text[i] == "{":
                depth += 1
            elif conf_text[i] == "}":
                depth -= 1
            i += 1
        blocks.append(conf_text[m.end() : i - 1])
    return blocks


@pytest.fixture
def http_only_text():
    return _HTTP_ONLY.read_text()


@pytest.fixture
def http_and_https_text():
    return _HTTP_AND_HTTPS.read_text()


class TestValidateBlockCount:
    def test_http_only_has_one_validate_block(self, http_only_text):
        assert len(_extract_validate_blocks(http_only_text)) == 1

    def test_http_and_https_has_two_validate_blocks(self, http_and_https_text):
        # HTTP server + HTTPS server each have their own /validate block.
        assert len(_extract_validate_blocks(http_and_https_text)) == 2


class TestEveryValidateBlockForwardsGenericMarkers:
    """Invariant: all three /validate blocks forward the identical marker
    set. A block missing one is the marker-spoof SSRF hole."""

    @pytest.mark.parametrize(
        "conf_fixture",
        ["http_only_text", "http_and_https_text"],
    )
    def test_all_blocks_forward_all_markers(self, conf_fixture, request):
        conf_text = request.getfixturevalue(conf_fixture)
        blocks = _extract_validate_blocks(conf_text)
        assert blocks, "no /validate blocks found"
        for idx, block in enumerate(blocks):
            for forward in _GENERIC_MARKER_FORWARDS:
                assert forward in block, (
                    f"{conf_fixture} /validate block #{idx} is missing "
                    f"required marker forward: {forward!r}"
                )

    def test_marker_set_is_identical_across_all_three_blocks(
        self, http_only_text, http_and_https_text
    ):
        # Collect, per block, which of the generic marker forwards it contains;
        # assert every block contains the full set (so the sets are identical).
        all_blocks = _extract_validate_blocks(http_only_text) + _extract_validate_blocks(
            http_and_https_text
        )
        assert len(all_blocks) == 3
        per_block_sets = [
            {f for f in _GENERIC_MARKER_FORWARDS if f in block} for block in all_blocks
        ]
        expected = set(_GENERIC_MARKER_FORWARDS)
        for idx, present in enumerate(per_block_sets):
            assert present == expected, f"/validate block #{idx} marker set drifted: {present}"


class TestMarkerMapsDeclared:
    @pytest.mark.parametrize(
        "conf_fixture",
        ["http_only_text", "http_and_https_text"],
    )
    def test_all_marker_maps_declared_with_empty_default(self, conf_fixture, request):
        conf_text = request.getfixturevalue(conf_fixture)
        for map_decl in _GENERIC_MARKER_MAPS:
            assert map_decl in conf_text, f"{conf_fixture} missing http-scope map: {map_decl!r}"
        # Each marker map must default to empty string (so a non-generic request
        # forwards "" and the generic mint never fires for it).
        for var in ("generic_backend_url", "generic_proxy_kind", "entity_path"):
            block = re.search(r"map \$host \$" + var + r"\s*\{([^}]*)\}", conf_text, re.DOTALL)
            assert block, f"{conf_fixture}: no map body for ${var}"
            assert 'default "";' in block.group(1), (
                f"{conf_fixture}: map ${var} must have empty default"
            )


class TestGenericUpstreamSeparateFromBackendUrl:
    """Double-mint guard at the template level: no /validate block may forward
    the generic upstream from $backend_url (must be $generic_backend_url)."""

    @pytest.mark.parametrize(
        "conf_fixture",
        ["http_only_text", "http_and_https_text"],
    )
    def test_generic_upstream_not_sourced_from_backend_url(self, conf_fixture, request):
        conf_text = request.getfixturevalue(conf_fixture)
        assert "X-Resolved-Generic-Upstream $backend_url;" not in conf_text
