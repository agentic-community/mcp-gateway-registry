"""Unit tests for ARD mapping helpers (pure functions, no I/O)."""

import re

from registry.schemas.ard_models import ArdCatalogEntry
from registry.services import ard_mapping

# Both regexes a conformant identifier must satisfy.
_SCHEMA_RE = re.compile(r"^urn:air:[a-zA-Z0-9.-]+(:[a-zA-Z0-9._-]+)+$")
_TOOL_RE = re.compile(r"^urn:air:([a-zA-Z0-9.-]+)(?::([a-zA-Z0-9._:-]+))?:([a-zA-Z0-9._-]+)$")


class TestSanitizeName:
    """Tests for _sanitize_name."""

    def test_strips_prefix_and_slashes(self):
        assert ard_mapping._sanitize_name("/servers/github/") == "github"

    def test_replaces_unsafe_chars(self):
        assert ard_mapping._sanitize_name("/agents/My Agent!") == "My-Agent"

    def test_empty_when_only_unsafe(self):
        assert ard_mapping._sanitize_name("/skills/!!!") == ""


class TestBuildUrn:
    """Tests for _build_urn double validation."""

    def test_valid_urn_passes_both_regexes(self):
        urn = ard_mapping._build_urn("registry.example.com", "server", "github")
        assert urn == "urn:air:registry.example.com:server:github"
        assert _SCHEMA_RE.match(urn)
        assert _TOOL_RE.match(urn)

    def test_empty_name_returns_none(self):
        assert ard_mapping._build_urn("registry.example.com", "server", "") is None


class TestRepresentativeQueries:
    """Tests for _derive_representative_queries 2-5 bound."""

    def test_returns_none_when_fewer_than_two(self):
        assert ard_mapping._derive_representative_queries([], None) is None

    def test_single_tag_no_description_is_none(self):
        # Only one derivable query -> omit (schema requires >= 2).
        assert ard_mapping._derive_representative_queries(["solo"], None) is None

    def test_two_tags_yield_two_queries(self):
        out = ard_mapping._derive_representative_queries(["vcs", "github"], None)
        assert out == ["vcs tools", "github tools"]

    def test_caps_at_five(self):
        tags = ["a", "b", "c"]
        out = ard_mapping._derive_representative_queries(tags, "First. Second.")
        assert out is not None
        assert 2 <= len(out) <= 5

    def test_tag_plus_description_when_one_tag(self):
        out = ard_mapping._derive_representative_queries(
            ["solo"], "Does a useful thing. More detail."
        )
        assert out == ["solo tools", "Does a useful thing"]


class TestNormalizeTimestamp:
    """Tests for _normalize_timestamp."""

    def test_none_returns_none(self):
        assert ard_mapping._normalize_timestamp(None) is None

    def test_appends_z_when_naive(self):
        assert ard_mapping._normalize_timestamp("2026-06-20T12:00:00") == ("2026-06-20T12:00:00Z")

    def test_keeps_existing_z(self):
        assert ard_mapping._normalize_timestamp("2026-06-20T12:00:00Z") == ("2026-06-20T12:00:00Z")

    def test_collapses_utc_offset(self):
        assert ard_mapping._normalize_timestamp("2026-06-20T12:00:00+00:00") == (
            "2026-06-20T12:00:00Z"
        )


class TestMapServer:
    """Tests for map_server."""

    def _record(self):
        return {
            "server_name": "GitHub Tools",
            "description": "GitHub repos and issues.",
            "tags": ["vcs", "github"],
            "tool_list": [{"name": "search_repos"}, {"name": "create_issue"}],
            "version": "1.0.0",
            "updated_at": "2026-06-20T12:00:00",
        }

    def test_happy_path(self):
        entry = ard_mapping.map_server(
            "/servers/github",
            self._record(),
            "registry.example.com",
            "https://registry.example.com/api/public/servers/github/server.json",
        )
        assert isinstance(entry, ArdCatalogEntry)
        assert entry.identifier == "urn:air:registry.example.com:server:github"
        assert entry.type == "application/mcp-server-card+json"
        assert entry.capabilities == ["search_repos", "create_issue"]
        assert entry.url and entry.url.startswith("https://")

    def test_serialized_has_url_never_data(self):
        entry = ard_mapping.map_server(
            "/servers/github", self._record(), "registry.example.com", "https://x/y"
        )
        dumped = entry.model_dump(by_alias=True, exclude_none=True)
        assert "url" in dumped
        assert "data" not in dumped

    def test_invalid_name_skipped(self):
        entry = ard_mapping.map_server(
            "/servers/!!!", self._record(), "registry.example.com", "https://x/y"
        )
        assert entry is None


class TestMapAgent:
    """Tests for map_agent."""

    def test_capabilities_from_skills(self):
        record = {
            "name": "Travel Agent",
            "description": "Books travel.",
            "tags": ["travel"],
            "skills": [{"name": "book_flight"}, {"name": "book_hotel"}],
            "version": "1.0",
            "updated_at": "2026-04-30T04:10:41",
        }
        entry = ard_mapping.map_agent(
            "/agents/travel-agent", record, "registry.example.com", "https://x/y"
        )
        assert entry.type == "application/a2a-agent-card+json"
        assert entry.capabilities == ["book_flight", "book_hotel"]


class TestMapSkill:
    """Tests for map_skill."""

    def test_happy_path(self):
        entry = ard_mapping.map_skill(
            path="/skills/pdf",
            name="pdf",
            description="Handle PDFs.",
            tags=["pdf", "docs"],
            tool_names=["Read", "Bash"],
            version=None,
            updated_at="2026-06-01T00:00:00",
            publisher="registry.example.com",
            record_url="https://registry.example.com/api/public/skills/pdf",
        )
        assert entry.type == "application/ai-skill"
        assert entry.identifier == "urn:air:registry.example.com:skill:pdf"
        assert entry.capabilities == ["Read", "Bash"]
