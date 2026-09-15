"""Virtual server counts must survive the search response projection.

The search repository flattens num_tools / backend_count / backend_paths onto the
result entry and emits no "metadata" key at all. The route used to read all three
only out of ``vs["metadata"]``, so every virtual server in semantic search
reported 0 tools and 0 backends however many it really had.

Found while verifying the issue #1752 step 6 change, which is what first exposed
virtual servers to MCP clients through ``search_registry``. Before that they were
fetched and discarded, so nobody saw the zeros.

These tests call ``_virtual_server_field`` directly, the same helper the route
uses, and then assert the response model accepts what it produces.
"""

from registry.api.search_routes import (
    VirtualServerSearchResult,
    _virtual_server_field,
)


def _read_vs_fields(vs: dict) -> tuple[int, int, list[str]]:
    """Read the three counts through the real helper the route uses."""
    return (
        _virtual_server_field(vs, "num_tools", 0),
        _virtual_server_field(vs, "backend_count", 0),
        _virtual_server_field(vs, "backend_paths", []),
    )


def _repository_virtual_server_entry() -> dict:
    """The shape all three repository branches actually emit."""
    return {
        "entity_type": "virtual_server",
        "path": "/virtual/dev-essentials",
        "server_name": "dev-essentials",
        "description": "developer tooling bundle",
        "tags": ["claude"],
        "num_tools": 7,
        "backend_count": 5,
        "backend_paths": [
            "/aws-kb",
            "/currenttime/",
            "/cloudflare-docs",
            "/ai.exa-exa",
            "/ai.agenticshelf-mcp",
        ],
        "is_enabled": True,
        "relevance_score": 0.0,
        "match_context": "developer tooling bundle",
        "matching_tools": [],
    }


class TestVirtualServerCountProjection:
    """The regression: flattened counts were dropped on the way out."""

    def test_flattened_counts_are_not_lost(self) -> None:
        num_tools, backend_count, backend_paths = _read_vs_fields(
            _repository_virtual_server_entry()
        )

        assert num_tools == 7
        assert backend_count == 5
        assert len(backend_paths) == 5

    def test_metadata_shaped_entry_still_works(self) -> None:
        """A producer that nests them under metadata keeps working."""
        vs = {
            "path": "/virtual/x",
            "metadata": {
                "num_tools": 3,
                "backend_count": 2,
                "backend_paths": ["/a", "/b"],
            },
        }

        assert _read_vs_fields(vs) == (3, 2, ["/a", "/b"])

    def test_flattened_value_wins_over_metadata(self) -> None:
        """The repository's own value is authoritative when both are present."""
        vs = {
            "path": "/virtual/x",
            "num_tools": 7,
            "backend_count": 5,
            "backend_paths": ["/a"],
            "metadata": {"num_tools": 99, "backend_count": 99, "backend_paths": []},
        }

        assert _read_vs_fields(vs) == (7, 5, ["/a"])

    def test_a_genuine_zero_is_preserved(self) -> None:
        """A virtual server with no backends reads 0, not a metadata fallback."""
        vs = {
            "path": "/virtual/empty",
            "num_tools": 0,
            "backend_count": 0,
            "backend_paths": [],
            "metadata": {"num_tools": 99, "backend_count": 99},
        }

        assert _read_vs_fields(vs) == (0, 0, [])

    def test_absent_everywhere_falls_back_to_zero(self) -> None:
        assert _read_vs_fields({"path": "/virtual/empty"}) == (0, 0, [])

    def test_explicit_none_falls_back_rather_than_crashing(self) -> None:
        """A None on either side must not reach the model, which forbids null."""
        vs = {
            "path": "/virtual/x",
            "num_tools": None,
            "backend_count": None,
            "backend_paths": None,
            "metadata": {"num_tools": 4},
        }

        assert _read_vs_fields(vs) == (4, 0, [])

    def test_null_metadata_is_tolerated(self) -> None:
        """metadata: None is what the repository would emit if it emitted the key."""
        assert _read_vs_fields({"path": "/virtual/x", "metadata": None}) == (0, 0, [])


class TestVirtualServerResultModel:
    """The projected values must satisfy the response model."""

    def test_model_accepts_the_flattened_values(self) -> None:
        entry = _repository_virtual_server_entry()
        num_tools, backend_count, backend_paths = _read_vs_fields(entry)

        result = VirtualServerSearchResult(
            path=entry["path"],
            server_name=entry["server_name"],
            relevance_score=entry["relevance_score"],
            num_tools=num_tools,
            backend_count=backend_count,
            backend_paths=backend_paths,
        )

        assert result.num_tools == 7
        assert result.backend_count == 5
        assert result.backend_paths[0] == "/aws-kb"

    def test_a_zero_relevance_virtual_server_is_still_valid(self) -> None:
        """_normalize_scores maps the worst result in a set to exactly 0.0."""
        result = VirtualServerSearchResult(
            path="/virtual/x",
            server_name="x",
            relevance_score=0.0,
            similarity_score=-0.0609,
        )

        assert result.relevance_score == 0.0
        assert result.similarity_score == -0.0609
