"""Settings tests for the remote embeddings provider after LiteLLM removal (issue #1826)."""

import pytest

from registry.core.config import Settings


@pytest.fixture(autouse=True)
def _clear_embeddings_env(monkeypatch):
    """Keep embeddings settings from the developer's shell out of these tests."""
    for name in (
        "EMBEDDINGS_PROVIDER",
        "EMBEDDINGS_MODEL_NAME",
        "EMBEDDINGS_RESPONSE_FORMAT",
        "EMBEDDINGS_REQUEST_TIMEOUT_SECONDS",
        "EMBEDDINGS_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.unit
@pytest.mark.core
class TestStartupModelCheck:
    """An unsupported remote model name stops the registry at startup."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "openai/text-embedding-3-small",
            "azure/my-deployment",
            "bedrock/amazon.titan-embed-text-v2:0",
            "bedrock/cohere.embed-english-v3",
        ],
    )
    def test_supported_names_accepted(self, model_name):
        s = Settings(embeddings_provider="litellm", embeddings_model_name=model_name)
        assert s.embeddings_model_name == model_name

    def test_removed_prefix_rejected(self):
        with pytest.raises(ValueError, match="removed with LiteLLM"):
            Settings(
                embeddings_provider="litellm", embeddings_model_name="cohere/embed-english-v3.0"
            )

    def test_unknown_prefix_rejected(self):
        with pytest.raises(ValueError, match="EMBEDDINGS_MODEL_NAME='mistral/mistral-embed'"):
            Settings(embeddings_provider="litellm", embeddings_model_name="mistral/mistral-embed")

    def test_raw_array_skips_the_check(self):
        s = Settings(
            embeddings_provider="litellm",
            embeddings_model_name="my-proxy/model-x",
            embeddings_response_format="raw_array",
        )
        assert s.embeddings_model_name == "my-proxy/model-x"

    def test_sentence_transformers_not_checked(self):
        s = Settings(
            embeddings_provider="sentence-transformers", embeddings_model_name="all-MiniLM-L6-v2"
        )
        assert s.embeddings_provider == "sentence-transformers"


@pytest.mark.unit
@pytest.mark.core
class TestTimeoutAndRetries:
    """Defaults and bounds for the two new settings."""

    def test_defaults(self):
        s = Settings()
        assert s.embeddings_request_timeout_seconds == 30
        assert s.embeddings_max_retries == 2

    def test_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("EMBEDDINGS_REQUEST_TIMEOUT_SECONDS", "90")
        monkeypatch.setenv("EMBEDDINGS_MAX_RETRIES", "0")
        s = Settings()
        assert s.embeddings_request_timeout_seconds == 90
        assert s.embeddings_max_retries == 0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("embeddings_request_timeout_seconds", 0),
            ("embeddings_request_timeout_seconds", 601),
            ("embeddings_max_retries", -1),
            ("embeddings_max_retries", 11),
        ],
    )
    def test_out_of_range_rejected(self, field, value):
        with pytest.raises(ValueError):
            Settings(**{field: value})


@pytest.mark.unit
@pytest.mark.core
class TestRetryMaxInputChars:
    """EMBEDDINGS_RETRY_MAX_INPUT_CHARS is optional and bounded (#1917)."""

    def test_unset_by_default(self, monkeypatch):
        monkeypatch.delenv("EMBEDDINGS_RETRY_MAX_INPUT_CHARS", raising=False)
        assert Settings().embeddings_retry_max_input_chars is None

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_value_means_unset(self, monkeypatch, blank):
        # Compose (${VAR:-}) and Terraform (null -> "") pass an empty string.
        monkeypatch.setenv("EMBEDDINGS_RETRY_MAX_INPUT_CHARS", blank)
        assert Settings().embeddings_retry_max_input_chars is None

    def test_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("EMBEDDINGS_RETRY_MAX_INPUT_CHARS", "8000")
        assert Settings().embeddings_retry_max_input_chars == 8000

    @pytest.mark.parametrize("value", [255, 200001])
    def test_out_of_range_rejected(self, value):
        with pytest.raises(ValueError):
            Settings(embeddings_retry_max_input_chars=value)
