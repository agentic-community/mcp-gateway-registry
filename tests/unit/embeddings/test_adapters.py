"""Unit tests for the remote embeddings adapters (issue #1826).

Each adapter must send the request LiteLLM 1.93 sent, so the tests pin the exact
URL, headers and body. httpx.post and the boto3 client are mocked: nothing here
touches the network.
"""

import io
import json
from unittest.mock import (
    MagicMock,
    patch,
)

import httpx
import pytest

from registry.embeddings import adapters
from registry.embeddings.adapters import (
    _build_headers,
    _extract_openai_envelope,
    _extract_raw_array,
    embed_azure,
    embed_bedrock_cohere,
    embed_bedrock_titan,
    embed_openai,
)

API_BASE: str = "https://embeddings.example.com/v1"


def _response(
    status: int,
    body: object,
) -> httpx.Response:
    """Build a real httpx.Response carrying a JSON body."""
    request = httpx.Request("POST", API_BASE + "/embeddings")
    return httpx.Response(status, json=body, request=request)


def _envelope(
    vectors: list[list[float]],
) -> dict:
    """Build an OpenAI embeddings envelope for the given vectors."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)
        ],
    }


def _bedrock_response(
    payload: dict,
) -> dict:
    """Build the shape boto3 invoke_model returns, with a streaming body."""
    return {"body": io.BytesIO(json.dumps(payload).encode())}


@pytest.fixture(autouse=True)
def _no_backoff_sleep():
    """Retries must not slow the suite down."""
    with patch("registry.embeddings.adapters.time.sleep") as sleep:
        yield sleep


@pytest.fixture(autouse=True)
def _fresh_bedrock_client_cache():
    """The boto3 client is cached per region; clear it between tests."""
    adapters._bedrock_client.cache_clear()
    yield
    adapters._bedrock_client.cache_clear()


@pytest.mark.unit
class TestExtractRawArray:
    """The raw_array parser accepts a bare list or one wrapped under a known key."""

    def test_bare_list_of_vectors(self):
        assert _extract_raw_array([[0.1, 0.2], [0.3, 0.4]]) == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.parametrize("key", ["embeddings", "data", "vectors"])
    def test_dict_wrapped_under_known_key(self, key):
        assert _extract_raw_array({key: [[0.1, 0.2]]}) == [[0.1, 0.2]]

    def test_dict_without_known_key_raises(self):
        with pytest.raises(ValueError, match="without an"):
            _extract_raw_array({"result": [[0.1]]})

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            _extract_raw_array([])

    def test_list_of_non_lists_raises(self):
        with pytest.raises(ValueError, match="list of lists"):
            _extract_raw_array([0.1, 0.2])


@pytest.mark.unit
class TestExtractOpenAIEnvelope:
    """Vectors come back in input order even if the endpoint reorders them."""

    def test_orders_by_index(self):
        payload = {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
        assert _extract_openai_envelope(payload) == [[1.0], [2.0]]

    def test_missing_data_raises(self):
        with pytest.raises(ValueError, match="no 'data' list"):
            _extract_openai_envelope({"embeddings": []})


@pytest.mark.unit
class TestBuildHeaders:
    """Operator headers go first; registry-managed headers always win, in any casing."""

    def test_extra_headers_included(self):
        headers = _build_headers({"X-Tenant-Id": "t1"}, "Authorization", "Bearer k")
        assert headers["X-Tenant-Id"] == "t1"
        assert headers["Authorization"] == "Bearer k"

    def test_managed_names_dropped_in_any_casing(self):
        headers = _build_headers(
            {"content-type": "text/plain", "ACCEPT": "x", "authorization": "evil"},
            "Authorization",
            "Bearer k",
        )
        names = [name.lower() for name in headers]
        assert names.count("content-type") == 1
        assert names.count("accept") == 1
        assert names.count("authorization") == 1
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer k"

    def test_no_credential_header_when_value_missing(self):
        headers = _build_headers(None, "Authorization", None)
        assert set(headers) == {"Content-Type", "Accept"}


@pytest.mark.unit
class TestEmbedOpenAI:
    """OpenAI-compatible adapter: URL, Bearer header, body, parsing, retries."""

    def _call(
        self,
        **overrides,
    ) -> list[list[float]]:
        kwargs = {
            "texts": ["a", "b"],
            "model_id": "text-embedding-3-small",
            "api_base": API_BASE,
            "api_key": "sk-test",
            "extra_headers": None,
            "response_format": "openai",
            "timeout_seconds": 30,
            "max_retries": 2,
        }
        kwargs.update(overrides)
        return embed_openai(**kwargs)

    @patch("registry.embeddings.adapters.httpx.post")
    def test_request_shape_matches_litellm(self, mock_post):
        mock_post.return_value = _response(200, _envelope([[0.1], [0.2]]))
        self._call()
        url = mock_post.call_args.args[0]
        kwargs = mock_post.call_args.kwargs
        assert url == API_BASE + "/embeddings"
        assert kwargs["json"] == {"model": "text-embedding-3-small", "input": ["a", "b"]}
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
        assert kwargs["timeout"] == 30

    @patch("registry.embeddings.adapters.httpx.post")
    def test_body_has_only_model_and_input(self, mock_post):
        # LiteLLM also sent encoding_format; strict endpoints reject unknown keys.
        mock_post.return_value = _response(200, _envelope([[0.1], [0.2]]))
        self._call()
        assert set(mock_post.call_args.kwargs["json"]) == {"model", "input"}

    @patch("registry.embeddings.adapters.httpx.post")
    def test_default_api_base_is_openai(self, mock_post):
        mock_post.return_value = _response(200, _envelope([[0.1], [0.2]]))
        self._call(api_base=None)
        assert mock_post.call_args.args[0] == "https://api.openai.com/v1/embeddings"

    @patch("registry.embeddings.adapters.httpx.post")
    def test_trailing_slash_on_api_base(self, mock_post):
        mock_post.return_value = _response(200, _envelope([[0.1], [0.2]]))
        self._call(api_base=API_BASE + "/")
        assert mock_post.call_args.args[0] == API_BASE + "/embeddings"

    @patch("registry.embeddings.adapters.httpx.post")
    def test_returns_vectors_in_order(self, mock_post):
        mock_post.return_value = _response(200, _envelope([[0.1], [0.2]]))
        assert self._call() == [[0.1], [0.2]]

    @patch("registry.embeddings.adapters.httpx.post")
    def test_raw_array_response(self, mock_post):
        mock_post.return_value = _response(200, [[0.1], [0.2]])
        assert self._call(response_format="raw_array") == [[0.1], [0.2]]

    @patch("registry.embeddings.adapters.httpx.post")
    def test_extra_headers_sent(self, mock_post):
        mock_post.return_value = _response(200, _envelope([[0.1], [0.2]]))
        self._call(extra_headers={"X-Module-Name": "registry"})
        assert mock_post.call_args.kwargs["headers"]["X-Module-Name"] == "registry"

    @patch("registry.embeddings.adapters.httpx.post")
    def test_retries_429_then_succeeds(self, mock_post, _no_backoff_sleep):
        mock_post.side_effect = [
            _response(429, {"error": "slow down"}),
            _response(200, _envelope([[0.1], [0.2]])),
        ]
        assert self._call() == [[0.1], [0.2]]
        assert mock_post.call_count == 2
        _no_backoff_sleep.assert_called_once_with(adapters.BACKOFF_BASE_SECONDS)

    @patch("registry.embeddings.adapters.httpx.post")
    def test_retries_connection_error(self, mock_post):
        mock_post.side_effect = [
            httpx.ConnectError("refused"),
            _response(200, _envelope([[0.1], [0.2]])),
        ]
        assert self._call() == [[0.1], [0.2]]

    @patch("registry.embeddings.adapters.httpx.post")
    def test_gives_up_after_max_retries(self, mock_post):
        mock_post.return_value = _response(503, {"error": "down"})
        with pytest.raises(RuntimeError, match="HTTP 503"):
            self._call(max_retries=2)
        assert mock_post.call_count == 3

    @patch("registry.embeddings.adapters.httpx.post")
    def test_does_not_retry_400(self, mock_post):
        mock_post.return_value = _response(400, {"error": {"message": "input too long"}})
        with pytest.raises(RuntimeError, match="HTTP 400.*input too long"):
            self._call()
        assert mock_post.call_count == 1

    @patch("registry.embeddings.adapters.httpx.post")
    def test_does_not_retry_401(self, mock_post):
        mock_post.return_value = _response(401, {"message": "Unauthorized"})
        with pytest.raises(RuntimeError, match="HTTP 401"):
            self._call()
        assert mock_post.call_count == 1

    @patch("registry.embeddings.adapters.httpx.post")
    def test_zero_retries_means_one_attempt(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("refused")
        with pytest.raises(RuntimeError, match="after 1 attempt"):
            self._call(max_retries=0)
        assert mock_post.call_count == 1

    @patch("registry.embeddings.adapters.httpx.post")
    def test_error_excerpt_is_bounded(self, mock_post):
        mock_post.return_value = _response(400, {"error": "x" * 5000})
        with pytest.raises(RuntimeError) as exc_info:
            self._call()
        assert len(str(exc_info.value)) < adapters.ERROR_BODY_EXCERPT_CHARS + 100


@pytest.mark.unit
class TestEmbedAzure:
    """Azure adapter: deployment URL, api-version, api-key header."""

    @patch("registry.embeddings.adapters.httpx.post")
    def test_request_shape_matches_litellm(self, mock_post):
        mock_post.return_value = _response(200, _envelope([[0.1]]))
        embed_azure(
            texts=["a"],
            deployment="my-deployment",
            api_base="https://res.openai.azure.com/",
            api_key="az-key",
            api_version="2025-02-01-preview",
            extra_headers={"X-Tenant-Id": "t1"},
            timeout_seconds=30,
            max_retries=0,
        )
        url = mock_post.call_args.args[0]
        kwargs = mock_post.call_args.kwargs
        assert url == (
            "https://res.openai.azure.com/openai/deployments/my-deployment/embeddings"
            "?api-version=2025-02-01-preview"
        )
        assert kwargs["headers"]["api-key"] == "az-key"
        assert "Authorization" not in kwargs["headers"]
        assert kwargs["headers"]["X-Tenant-Id"] == "t1"
        assert kwargs["json"] == {"model": "my-deployment", "input": ["a"]}


@pytest.mark.unit
class TestEmbedBedrock:
    """Bedrock adapters: request bodies, per-text Titan calls, batched Cohere call."""

    @patch("boto3.client")
    def test_titan_one_call_per_text(self, mock_boto_client):
        client = MagicMock()
        client.invoke_model.side_effect = [
            _bedrock_response({"embedding": [0.1, 0.2], "inputTextTokenCount": 1}),
            _bedrock_response({"embedding": [0.3, 0.4], "inputTextTokenCount": 1}),
        ]
        mock_boto_client.return_value = client
        vectors = embed_bedrock_titan(["a", "b"], "amazon.titan-embed-text-v1", "us-east-1", 30, 2)
        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        bodies = [json.loads(c.kwargs["body"]) for c in client.invoke_model.call_args_list]
        assert bodies == [{"inputText": "a"}, {"inputText": "b"}]
        assert client.invoke_model.call_args.kwargs["modelId"] == "amazon.titan-embed-text-v1"

    @patch("boto3.client")
    def test_titan_v2_prefers_embeddings_by_type(self, mock_boto_client):
        client = MagicMock()
        client.invoke_model.return_value = _bedrock_response(
            {"embedding": [9.9], "embeddingsByType": {"float": [0.5]}, "inputTextTokenCount": 1}
        )
        mock_boto_client.return_value = client
        vectors = embed_bedrock_titan(["a"], "amazon.titan-embed-text-v2:0", "us-east-1", 30, 2)
        assert vectors == [[0.5]]

    @patch("boto3.client")
    def test_cohere_single_batched_call(self, mock_boto_client):
        client = MagicMock()
        client.invoke_model.return_value = _bedrock_response({"embeddings": [[0.1], [0.2]]})
        mock_boto_client.return_value = client
        vectors = embed_bedrock_cohere(["a", "b"], "cohere.embed-english-v3", "us-east-1", 30, 2)
        assert vectors == [[0.1], [0.2]]
        assert client.invoke_model.call_count == 1
        body = json.loads(client.invoke_model.call_args.kwargs["body"])
        assert body == {"texts": ["a", "b"], "input_type": "search_document"}

    @patch("boto3.client")
    def test_cohere_embeddings_by_type_dict(self, mock_boto_client):
        client = MagicMock()
        client.invoke_model.return_value = _bedrock_response({"embeddings": {"float": [[0.7]]}})
        mock_boto_client.return_value = client
        assert embed_bedrock_cohere(["a"], "cohere.embed-english-v3", None, 30, 2) == [[0.7]]

    @patch("boto3.client")
    def test_client_config_maps_retries_and_timeout(self, mock_boto_client):
        mock_boto_client.return_value = MagicMock()
        adapters._bedrock_client("eu-west-1", 45, 3)
        kwargs = mock_boto_client.call_args.kwargs
        assert mock_boto_client.call_args.args[0] == "bedrock-runtime"
        assert kwargs["region_name"] == "eu-west-1"
        assert kwargs["config"].read_timeout == 45
        assert kwargs["config"].retries == {"mode": "standard", "max_attempts": 4}

    @patch("boto3.client")
    def test_client_error_wrapped_with_code_and_message(self, mock_boto_client):
        from botocore.exceptions import ClientError

        client = MagicMock()
        client.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Too many input tokens"}},
            "InvokeModel",
        )
        mock_boto_client.return_value = client
        with pytest.raises(RuntimeError, match="ValidationException: Too many input tokens"):
            embed_bedrock_titan(["a"], "amazon.titan-embed-text-v1", "us-east-1", 30, 2)
