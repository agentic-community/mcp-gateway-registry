"""Tests for URL normalization helpers used by registration deduplication."""

import pytest

from registry.utils.url_normalize import (
    ENTITY_TYPE_AGENT,
    ENTITY_TYPE_SERVER,
    ENTITY_TYPE_SKILL,
    normalize_identity_url,
    normalize_proxy_url,
)


class TestNormalizeProxyUrl:
    @pytest.mark.parametrize(
        "url,expected",
        [
            (
                "https://Example.COM/",
                {"scheme": "https", "host": "example.com", "port": None, "path": ""},
            ),
            (
                "https://example.com:443/foo",
                {"scheme": "https", "host": "example.com", "port": None, "path": "/foo"},
            ),
            (
                "http://example.com:80/foo/",
                {"scheme": "http", "host": "example.com", "port": None, "path": "/foo"},
            ),
            (
                "https://example.com:8443/api?x=1#frag",
                {"scheme": "https", "host": "example.com", "port": 8443, "path": "/api"},
            ),
            (
                "https://example.com",
                {"scheme": "https", "host": "example.com", "port": None, "path": ""},
            ),
            (
                "  https://example.com/x  ",
                {"scheme": "https", "host": "example.com", "port": None, "path": "/x"},
            ),
        ],
    )
    def test_normalizes(self, url: str, expected: dict) -> None:
        assert normalize_proxy_url(url) == expected

    @pytest.mark.parametrize("url", [None, "", "not a url", "://nohost", "/relative/only"])
    def test_invalid_inputs_return_none(self, url: str | None) -> None:
        assert normalize_proxy_url(url) is None

    def test_urlparse_exception_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # urlparse is normally extremely permissive but documented to be
        # capable of raising. Cover the except-branch by patching it.
        from registry.utils import url_normalize as mod

        def _raise(_: str) -> None:
            raise ValueError("synthetic parse error")

        monkeypatch.setattr(mod, "urlparse", _raise)
        assert mod.normalize_proxy_url("https://example.com/x") is None


class TestNormalizeIdentityUrl:
    """Tests for the per-entity-type identity URL canonicalizer.

    Used by DuplicateCheckService to find exact-URL collisions on registration.
    """

    @pytest.mark.parametrize("entity_type", [ENTITY_TYPE_SERVER, ENTITY_TYPE_AGENT])
    @pytest.mark.parametrize(
        "a,b",
        [
            ("http://api.example.com/mcp", "https://api.example.com/mcp"),
            ("https://Example.COM/mcp", "https://example.com/mcp/"),
            ("https://example.com:443/mcp", "https://example.com/mcp"),
            ("http://example.com:80/mcp", "http://example.com/mcp"),
            ("https://example.com/mcp?x=1", "https://example.com/mcp"),
            ("https://example.com/mcp#frag", "https://example.com/mcp"),
        ],
    )
    def test_http_equivalence_classes(
        self,
        entity_type: str,
        a: str,
        b: str,
    ) -> None:
        """URLs that should denote the same service collapse to one identity."""
        normalized_a = normalize_identity_url(a, entity_type)
        normalized_b = normalize_identity_url(b, entity_type)
        assert normalized_a is not None
        assert normalized_a == normalized_b

    @pytest.mark.parametrize("entity_type", [ENTITY_TYPE_SERVER, ENTITY_TYPE_AGENT])
    @pytest.mark.parametrize(
        "a,b",
        [
            ("https://example.com:8443/mcp", "https://example.com:9000/mcp"),
            ("https://a.example.com/mcp", "https://b.example.com/mcp"),
            ("https://example.com/foo", "https://example.com/bar"),
        ],
    )
    def test_http_distinct_identities(
        self,
        entity_type: str,
        a: str,
        b: str,
    ) -> None:
        """URLs that denote different services produce different identities."""
        assert normalize_identity_url(a, entity_type) != normalize_identity_url(b, entity_type)

    def test_scheme_collapses_explicitly_for_servers(self) -> None:
        """Explicit equivalence-class assertion (kiro round-2 ask).

        Scheme is collapsed: http and https forms produce the same
        identity so a registration of one collides with the other.
        """
        http_form = normalize_identity_url("http://x/mcp", ENTITY_TYPE_SERVER)
        https_form = normalize_identity_url("https://x/mcp", ENTITY_TYPE_SERVER)
        assert http_form == https_form

    @pytest.mark.parametrize(
        "entity_type",
        [ENTITY_TYPE_SERVER, ENTITY_TYPE_AGENT, ENTITY_TYPE_SKILL],
    )
    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/mcp",
            "https://example.com:8443/mcp/sub/path",
            "https://github.com/org/repo/blob/main/.claude/skills/foo/SKILL.md",
        ],
    )
    def test_idempotent_when_input_is_already_canonical(
        self,
        entity_type: str,
        url: str,
    ) -> None:
        """Property: feeding a normalized identity through urlparse-able
        re-input still yields the same identity. We can't strictly assert
        ``f(f(x)) == f(x)`` because the output isn't a parseable URL —
        but normalizing two equivalent inputs must yield the same string.
        """
        first = normalize_identity_url(url, entity_type)
        second = normalize_identity_url(url + ("/" if not url.endswith("/") else ""), entity_type)
        assert first is not None
        assert first == second

    @pytest.mark.parametrize(
        "a,b",
        [
            (
                "https://github.com/org/repo.git",
                "https://github.com/org/repo",
            ),
            (
                "https://github.com/org/repo/",
                "https://github.com/org/repo",
            ),
            (
                "https://GitHub.com/org/repo",
                "https://github.com/org/repo",
            ),
            (
                "http://github.com/org/repo",
                "https://github.com/org/repo",
            ),
        ],
    )
    def test_skill_github_equivalence_classes(self, a: str, b: str) -> None:
        """Skill identity URLs collapse common GitHub URL variations."""
        normalized_a = normalize_identity_url(a, ENTITY_TYPE_SKILL)
        normalized_b = normalize_identity_url(b, ENTITY_TYPE_SKILL)
        assert normalized_a is not None
        assert normalized_a == normalized_b

    def test_skill_path_case_preserved(self) -> None:
        """GitHub paths are case-sensitive; the identity must preserve case."""
        upper = normalize_identity_url(
            "https://github.com/Org/Repo/blob/main/SKILL.md",
            ENTITY_TYPE_SKILL,
        )
        lower = normalize_identity_url(
            "https://github.com/org/repo/blob/main/SKILL.md",
            ENTITY_TYPE_SKILL,
        )
        assert upper != lower

    @pytest.mark.parametrize(
        "entity_type",
        [ENTITY_TYPE_SERVER, ENTITY_TYPE_AGENT, ENTITY_TYPE_SKILL],
    )
    @pytest.mark.parametrize("url", [None, "", "not a url", "/relative", "://nohost"])
    def test_invalid_inputs_return_none(
        self,
        entity_type: str,
        url: str | None,
    ) -> None:
        assert normalize_identity_url(url, entity_type) is None

    def test_unknown_entity_type_returns_none(self) -> None:
        assert normalize_identity_url("https://x.example.com/", "unknown_type") is None
