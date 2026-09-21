"""Vault-key canonicalization tests.

An Auth0 ``sub`` (``auth0|abc``), a Keycloak username with a space/accent,
and a multi-segment ``server_path``must all encode to separator-free segments
so they cannot collide with the ``/`` (OpenBao path) or ``|`` (map-key) delimiters,
and must round-trip identically.
"""

import pytest

from registry.egress_auth.service import is_per_user_auth_method
from registry.secrets import keys


@pytest.mark.unit
class TestEncodeSegment:
    @pytest.mark.parametrize(
        "raw",
        [
            "auth0|abc123",  # Auth0 sub -- the "|" collision case
            "alice smith",  # Keycloak username with a space
            "café",  # non-ASCII (NFC)
            "user@example.com",  # Entra UPN
            "a+b",
            "/github-mcp/mcp",  # server_path with slashes
            "plain",
        ],
    )
    def test_encoded_segment_has_no_separators(self, raw: str) -> None:
        enc = keys.encode_segment(raw)
        assert "/" not in enc, f"{raw!r} encoded to {enc!r} containing '/'"
        assert "|" not in enc, f"{raw!r} encoded to {enc!r} containing '|'"

    @pytest.mark.parametrize(
        "raw",
        [
            "auth0|abc123",
            "alice smith",
            "café",
            "user@example.com",
            "a+b",
            "/github-mcp/mcp",
            "/github",
            "plain",
        ],
    )
    def test_encoded_segment_is_hvac_and_vault_safe(self, raw: str) -> None:
        # Regression: percent-encoding emitted "%", which the OpenBao HTTP client
        # (hvac) re-quoted ("%2F" -> "%252F"), producing a path OpenBao would not
        # match -> every vault read failed. base64url has none of the reserved
        # characters, so an encoded segment must contain no "%", "=", "/", or "|".
        enc = keys.encode_segment(raw)
        for bad in ("%", "=", "/", "|"):
            assert bad not in enc, f"{raw!r} encoded to {enc!r} containing {bad!r}"

    @pytest.mark.parametrize(
        "raw",
        [
            "auth0|abc123",
            "alice smith",
            "café",
            "user@example.com",
            "/github-mcp/mcp",
            "/github",
            "plain",
            "",
        ],
    )
    def test_encode_decode_roundtrip(self, raw: str) -> None:
        assert keys.decode_segment(keys.encode_segment(raw)) == raw

    def test_nfc_normalization_collapses_equivalent_forms(self) -> None:
        # NFC and NFD representations of "é" must encode identically, else a
        # Keycloak user logging in from two clients could silently get two vaults.
        nfc = "café"  # é as single code point
        nfd = "café"  # e + combining acute accent
        assert keys.encode_segment(nfc) == keys.encode_segment(nfd)

    def test_distinct_values_stay_distinct(self) -> None:
        assert keys.encode_segment("auth0|alice") != keys.encode_segment("auth0|bob")


@pytest.mark.unit
class TestKeyComposition:
    def test_user_principal_two_segments(self) -> None:
        p = keys.user_principal("oauth2", "auth0|abc")
        # exactly one "/" (the segment join); neither segment contains "/" or "|"
        assert p.count("/") == 1
        auth_part, user_part = p.split("/")
        assert "|" not in user_part

    def test_map_key_one_delimiter(self) -> None:
        k = keys.map_key("github", "/github-mcp/mcp")
        assert k.count(keys.MAP_KEY_DELIMITER) == 1
        provider_part, server_part = k.split(keys.MAP_KEY_DELIMITER)
        assert "/" not in server_part  # server_path slashes were encoded

    def test_openbao_path_segment_count(self) -> None:
        path = keys.openbao_path(
            "mcp/egress", "oauth2", "auth0|abc", "github", "/x/y", purpose=keys.EGRESS_PURPOSE
        )
        # prefix has one "/", then 4 encoded segments each joined by "/"
        # mcp/egress/<auth>/<user>/<provider>/<server>  => 5 slashes total
        assert path.count("/") == 5


@pytest.mark.unit
class TestPurposeNamespace:
    """`purpose` selects a disjoint address space, and egress keeps its old addresses."""

    def test_egress_prefix_is_byte_identical_to_the_pre_purpose_layout(self) -> None:
        """The no-migration guarantee. Do not "tidy" this into always appending.

        Every entry written before purposes existed is an egress entry, so if the egress
        prefix ever gains a segment, every already-vaulted credential becomes unreadable
        and every user has to re-consent. The pre-purpose builder was simply
        ``prefix.strip("/")``, so that is what egress must still produce.
        """
        for prefix in ("egress", "/egress", "egress/", "mcp/egress-tokens"):
            assert keys.namespaced_prefix(prefix, keys.EGRESS_PURPOSE) == prefix.strip("/")

    def test_discovery_gets_its_own_space(self) -> None:
        egress = keys.namespaced_prefix("egress", keys.EGRESS_PURPOSE)
        discovery = keys.namespaced_prefix("egress", keys.DISCOVERY_PURPOSE)
        assert discovery != egress
        # Nested under the configured prefix, so existing vault policies / IAM scoping
        # that grant the prefix keep working without a deployment change.
        assert discovery.startswith(f"{egress}/")

    def test_the_two_spaces_cannot_collide(self) -> None:
        """Depth alone separates them: egress is 4 segments past the prefix, discovery 5."""
        args = ("oauth2", "alice", "github", "/github-mcp")
        egress = keys.openbao_path("egress", *args, purpose=keys.EGRESS_PURPOSE)
        discovery = keys.openbao_path("egress", *args, purpose=keys.DISCOVERY_PURPOSE)
        assert egress != discovery
        assert discovery.count("/") == egress.count("/") + 1
        # No egress path can ever equal a discovery path, for ANY principal: a collision
        # would need an auth_method segment equal to the encoded purpose, i.e. an
        # auth_method literally named "discovery", which is not a per-user method.
        assert not is_per_user_auth_method(keys.DISCOVERY_PURPOSE)

    def test_purpose_segment_is_encoded_like_every_other_segment(self) -> None:
        discovery = keys.namespaced_prefix("egress", keys.DISCOVERY_PURPOSE)
        segment = discovery.rsplit("/", 1)[-1]
        assert segment == keys.encode_segment(keys.DISCOVERY_PURPOSE)
        assert keys.decode_segment(segment) == keys.DISCOVERY_PURPOSE
