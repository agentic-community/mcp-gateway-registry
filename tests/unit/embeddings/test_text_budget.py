"""Unit tests for EmbeddingText: byte-identical full text and longest-first trimming (#1917)."""

import pytest

from registry.utils.embedding_text import (
    EmbeddingText,
    _cut_at_word,
)
from registry.utils.embeddings_provider import (
    BEDROCK_COHERE_RETRY_MAX_INPUT_CHARS,
    DEFAULT_RETRY_MAX_INPUT_CHARS,
    default_retry_max_input_chars,
)


def _server_text(
    tool_count: int,
    description_length: int,
) -> EmbeddingText:
    """Build server-shaped text with many tools of the same description length."""
    return EmbeddingText(
        head=["Big Server", "A server with many tools.", "Tags: a, b"],
        items=[
            (f"tool_{i:03d}", ("word " * description_length)[:description_length])
            for i in range(tool_count)
        ],
        tail=["owner: team-x"],
    )


@pytest.mark.unit
class TestFullText:
    """full() rebuilds the pre-#1917 string exactly, so content hashes do not change."""

    def test_matches_the_old_join(self):
        text = EmbeddingText(
            head=["Name", "Desc", "Tags: x"],
            items=[("tool_a", "does a"), ("tool_b", "")],
            tail=["k: v"],
        )
        old = " ".join(
            filter(None, ["Name", "Desc", "Tags: x", "tool_a", "does a", "tool_b", "", "k: v"])
        )
        assert text.full() == old

    def test_empty_and_missing_parts_are_skipped_like_before(self):
        text = EmbeddingText(head=["Name", ""], items=[("", "only a description")])
        assert text.full() == "Name only a description"


@pytest.mark.unit
class TestTrimmed:
    """trimmed() keeps every name and shortens the longest descriptions first."""

    def test_text_under_the_limit_is_unchanged(self):
        text = _server_text(tool_count=3, description_length=50)
        assert text.trimmed(10_000) == text.full()

    def test_result_fits_the_limit(self):
        text = _server_text(tool_count=94, description_length=900)
        assert len(text.full()) > 28_000
        assert len(text.trimmed(28_000)) <= 28_000

    def test_every_tool_name_is_kept(self):
        text = _server_text(tool_count=94, description_length=900)
        trimmed = text.trimmed(28_000)
        for name, _description in text.items:
            assert name in trimmed

    def test_short_descriptions_stay_whole(self):
        long_description = "long " * 400
        text = EmbeddingText(
            head=["Server"],
            items=[("short_tool", "returns the time"), ("long_tool", long_description)],
        )
        trimmed = text.trimmed(400)
        assert "returns the time" in trimmed
        assert len(trimmed) <= 400

    def test_descriptions_are_cut_at_word_boundaries(self):
        text = EmbeddingText(head=["Server"], items=[("tool", "alpha beta gamma delta " * 50)])
        trimmed = text.trimmed(200)
        assert trimmed.split(" ")[-1] in {"alpha", "beta", "gamma", "delta"}

    def test_falls_back_to_a_hard_cut_when_names_alone_do_not_fit(self):
        text = EmbeddingText(head=["Server"], items=[(f"tool_{i}", "x") for i in range(500)])
        trimmed = text.trimmed(300)
        assert len(trimmed) <= 300
        assert trimmed.startswith("Server tool_0 tool_1")

    def test_head_only_text_is_hard_cut(self):
        text = EmbeddingText(head=["word " * 1000])
        assert len(text.trimmed(256)) <= 256


@pytest.mark.unit
class TestCutAtWord:
    """The word-boundary cut never exceeds the limit."""

    def test_cuts_at_the_last_space(self):
        assert _cut_at_word("alpha beta gamma", 12) == "alpha beta"

    def test_single_long_word_is_cut_hard(self):
        assert _cut_at_word("x" * 50, 10) == "x" * 10

    def test_zero_limit_gives_empty(self):
        assert _cut_at_word("alpha", 0) == ""


@pytest.mark.unit
class TestDefaultRetryMaxInputChars:
    """The default trim target depends on the model."""

    def test_cohere_on_bedrock_uses_its_hard_limit(self):
        assert default_retry_max_input_chars("bedrock/cohere.embed-english-v3") == (
            BEDROCK_COHERE_RETRY_MAX_INPUT_CHARS
        )

    @pytest.mark.parametrize(
        "model_name",
        [
            "openai/text-embedding-3-small",
            "bedrock/amazon.titan-embed-text-v2:0",
            "azure/deployment",
            "all-MiniLM-L6-v2",
        ],
    )
    def test_other_models_use_the_general_default(self, model_name):
        assert default_retry_max_input_chars(model_name) == DEFAULT_RETRY_MAX_INPUT_CHARS
