"""Unit tests for query-text normalization (issue #1696).

``strip_boilerplate_tokens`` exists because catalog suffixes carry most of
the cosine on a short name: `topic-mcp-server` and `category-mcp-server`
share nothing but `-mcp-server` and still measure 0.76 similar.
"""

from registry.utils.text_normalize import (
    BOILERPLATE_TOKENS,
    distinct_token_count,
    strip_boilerplate_tokens,
)


class TestStripBoilerplateTokens:
    def test_shared_catalog_suffix_is_removed(self) -> None:
        assert strip_boilerplate_tokens("topic-mcp-server") == "topic"
        assert strip_boilerplate_tokens("category-mcp-server") == "category"

    def test_separator_style_does_not_matter(self) -> None:
        """Hyphens, underscores, and spaces must tokenize identically."""
        expected = "topic"
        for name in ("topic-mcp-server", "topic_mcp_server", "topic mcp server"):
            assert strip_boilerplate_tokens(name) == expected

    def test_meaningful_words_survive_with_description(self) -> None:
        assert (
            strip_boilerplate_tokens("payroll-mcp-server Runs payroll for the finance team")
            == "payroll runs payroll for the finance team"
        )

    def test_all_boilerplate_yields_empty_string(self) -> None:
        """The caller must be able to tell "no signal" from "some signal"."""
        assert strip_boilerplate_tokens("mcp-server-tools") == ""

    def test_empty_input(self) -> None:
        assert strip_boilerplate_tokens("") == ""

    def test_plural_forms_are_stripped(self) -> None:
        assert strip_boilerplate_tokens("weather-tools-api") == "weather"

    def test_gateway_and_registry_are_kept(self) -> None:
        """Both carry meaning in this catalog; stripping them lost real duplicates."""
        assert "gateway" not in BOILERPLATE_TOKENS
        assert "registry" not in BOILERPLATE_TOKENS
        assert strip_boilerplate_tokens("gateway-registry-mcp-server") == "gateway registry"

    def test_custom_token_set_is_honoured(self) -> None:
        assert strip_boilerplate_tokens("acme-topic", frozenset({"acme"})) == "topic"

    def test_output_is_lowercased(self) -> None:
        assert strip_boilerplate_tokens("Topic MCP Server") == "topic"


class TestDistinctTokenCount:
    def test_counts_distinct_tokens(self) -> None:
        assert distinct_token_count("topic extractor") == 2

    def test_repetition_is_not_evidence(self) -> None:
        """A name echoed in its own description must not read as two tokens."""
        assert distinct_token_count("topic topic topic") == 1

    def test_empty_string(self) -> None:
        assert distinct_token_count("") == 0

    def test_punctuation_is_not_a_token(self) -> None:
        assert distinct_token_count("topic -- extractor!") == 2
