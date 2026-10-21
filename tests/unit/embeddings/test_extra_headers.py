"""Unit tests for operator-configured extra headers on embeddings requests (issue #1819).

Covers both outbound paths in LiteLLMClient: the direct httpx call used for
EMBEDDINGS_RESPONSE_FORMAT=raw_array, and the litellm.embedding() call used for
the default OpenAI envelope. The wire-level test at the bottom guards the LiteLLM
contract, which mock-based tests cannot see.
"""

import importlib
import json
import logging
import sys
import threading
from http.server import (
    BaseHTTPRequestHandler,
    HTTPServer,
)
from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from registry.embeddings.client import (
    LiteLLMClient,
    SentenceTransformersClient,
    create_embeddings_client,
)

API_BASE: str = "https://embeddings.example.com/v1"
TENANT_HEADERS: dict[str, str] = {"X-Tenant-Id": "research"}


def _raw_array_response() -> MagicMock:
    """Return a mocked httpx response carrying one bare-array vector."""
    response = MagicMock()
    response.json.return_value = [[0.1, 0.2]]
    response.raise_for_status.return_value = None
    return response


def _envelope_response() -> dict:
    """Return a minimal OpenAI embeddings envelope."""
    return {"data": [{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}]}


def _raw_array_client(
    extra_headers: dict[str, str] | None,
) -> LiteLLMClient:
    """Build a client on the direct (raw_array) path with a dynamic token."""
    return LiteLLMClient(
        model_name="openai/mock-model",
        api_base=API_BASE,
        embedding_dimension=2,
        token_provider=MagicMock(return_value="dynamic-token"),
        response_format="raw_array",
        extra_headers=extra_headers,
    )


def _envelope_client(
    extra_headers: dict[str, str] | None,
) -> LiteLLMClient:
    """Build a client on the litellm (openai envelope) path with a dynamic token."""
    return LiteLLMClient(
        model_name="openai/mock-model",
        api_base=API_BASE,
        embedding_dimension=2,
        token_provider=MagicMock(return_value="dynamic-token"),
        extra_headers=extra_headers,
    )


@pytest.mark.unit
class TestDirectPath:
    """Headers on the raw_array path, where the registry calls httpx itself."""

    @patch("httpx.post")
    def test_sends_extra_headers(self, mock_post):
        mock_post.return_value = _raw_array_response()
        _raw_array_client(TENANT_HEADERS).encode(["hi"])
        assert mock_post.call_args.kwargs["headers"]["X-Tenant-Id"] == "research"

    @patch("httpx.post")
    def test_keeps_managed_headers(self, mock_post):
        mock_post.return_value = _raw_array_response()
        _raw_array_client(TENANT_HEADERS).encode(["hi"])
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert headers["Authorization"] == "Bearer dynamic-token"

    @patch("httpx.post")
    def test_managed_headers_win_over_extra(self, mock_post):
        # Bypasses Settings validation on purpose: proves the ordering backstop.
        mock_post.return_value = _raw_array_response()
        client = _raw_array_client({"Content-Type": "text/plain", "Authorization": "Bearer evil"})
        client.encode(["hi"])
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer dynamic-token"

    @patch("httpx.post")
    def test_managed_headers_win_in_any_casing(self, mock_post):
        # A lowercase spelling is a different dict key but the same HTTP header,
        # so it must be dropped rather than sent alongside the managed value.
        mock_post.return_value = _raw_array_response()
        client = _raw_array_client({"content-type": "text/plain", "AUTHORIZATION": "Bearer evil"})
        client.encode(["hi"])
        sent_names = [name.lower() for name in mock_post.call_args.kwargs["headers"]]
        assert sent_names.count("content-type") == 1
        assert sent_names.count("authorization") == 1
        assert mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"

    @patch("httpx.post")
    def test_unset_matches_previous_header_set(self, mock_post):
        mock_post.return_value = _raw_array_response()
        _raw_array_client(None).encode(["hi"])
        headers = mock_post.call_args.kwargs["headers"]
        assert set(headers) == {"Content-Type", "Accept", "Authorization"}

    @patch("httpx.post")
    def test_constructor_copies_the_map(self, mock_post):
        mock_post.return_value = _raw_array_response()
        caller_map = {"X-Tenant-Id": "research"}
        client = _raw_array_client(caller_map)
        caller_map["X-Tenant-Id"] = "changed"
        caller_map["X-Added"] = "later"
        client.encode(["hi"])
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Tenant-Id"] == "research"
        assert "X-Added" not in headers


@pytest.mark.unit
class TestLiteLLMPath:
    """Headers on the default path, where litellm.embedding() makes the call."""

    def test_passes_extra_headers_kwarg(self):
        with patch("litellm.embedding", return_value=_envelope_response()) as mock_embedding:
            _envelope_client(TENANT_HEADERS).encode(["hi"])
        assert mock_embedding.call_args.kwargs["extra_headers"] == TENANT_HEADERS

    def test_omits_kwarg_when_empty(self):
        with patch("litellm.embedding", return_value=_envelope_response()) as mock_embedding:
            _envelope_client(None).encode(["hi"])
        assert "extra_headers" not in mock_embedding.call_args.kwargs

    def test_kwarg_is_a_copy(self):
        client = _envelope_client(TENANT_HEADERS)
        with patch("litellm.embedding", return_value=_envelope_response()) as mock_embedding:
            client.encode(["hi"])
        assert mock_embedding.call_args.kwargs["extra_headers"] is not client.extra_headers


@pytest.mark.unit
class TestFactory:
    """create_embeddings_client forwards the map to litellm only."""

    def test_forwards_extra_headers_to_litellm(self):
        client = create_embeddings_client(
            provider="litellm",
            model_name="openai/mock-model",
            api_base=API_BASE,
            extra_headers=TENANT_HEADERS,
        )
        assert isinstance(client, LiteLLMClient)
        assert client.extra_headers == TENANT_HEADERS

    def test_warns_for_sentence_transformers(self, caplog):
        with caplog.at_level(logging.WARNING):
            client = create_embeddings_client(
                provider="sentence-transformers",
                model_name="all-MiniLM-L6-v2",
                extra_headers=TENANT_HEADERS,
            )
        assert isinstance(client, SentenceTransformersClient)
        assert "EMBEDDINGS_EXTRA_HEADER_<N>" in caplog.text

    def test_silent_when_unset_for_sentence_transformers(self, caplog):
        with caplog.at_level(logging.WARNING):
            create_embeddings_client(
                provider="sentence-transformers",
                model_name="all-MiniLM-L6-v2",
            )
        assert "EMBEDDINGS_EXTRA_HEADER" not in caplog.text


@pytest.mark.unit
class TestLogHygiene:
    """Header values never reach logs; INFO carries only the count."""

    def test_header_values_never_logged(self, caplog):
        caplog.set_level(logging.DEBUG)
        client = _envelope_client({"X-Tenant-Id": "very-private-value"})
        with patch("litellm.embedding", return_value=_envelope_response()):
            client.encode(["hi"])
        for record in caplog.records:
            assert "very-private-value" not in record.getMessage()

    def test_info_log_reports_count_only(self, caplog):
        caplog.set_level(logging.INFO)
        _envelope_client({"X-Tenant-Id": "research", "X-Route": "v2"})
        info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        count_lines = [m for m in info_messages if "extra header(s)" in m]
        assert count_lines == ["Embeddings client configured with 2 extra header(s)"]
        assert not any("X-Tenant-Id" in m for m in info_messages)


class _RecordingHandler(BaseHTTPRequestHandler):
    """Loopback stub that records request headers and returns an OpenAI envelope."""

    received_headers: dict[str, str] = {}

    def do_POST(self) -> None:  # noqa: N802 (stdlib callback name)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        _RecordingHandler.received_headers = {k.lower(): v for k, v in self.headers.items()}
        body = json.dumps(
            {
                "data": [{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
                "model": "mock-model",
                "object": "list",
                "usage": {"prompt_tokens": None, "total_tokens": None},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(
        self,
        *args: object,
    ) -> None:
        """Silence the default stderr access log."""


@pytest.fixture
def real_litellm(monkeypatch):
    """Load the real litellm for one test, then restore the conftest mock.

    tests/conftest.py replaces litellm in sys.modules for the whole suite. The
    wire contract test must exercise the real library, so this fixture removes
    every litellm entry, imports the genuine package, and puts the original
    entries back afterwards so later tests still see the mock.
    """
    # Without this, importing litellm downloads its model cost map from GitHub
    # (up to a 5 second wait), which makes the test depend on internet access.
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    saved = {name: mod for name, mod in sys.modules.items() if name.split(".")[0] == "litellm"}
    for name in saved:
        del sys.modules[name]
    try:
        yield importlib.import_module("litellm")
    finally:
        for name in [n for n in sys.modules if n.split(".")[0] == "litellm"]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.mark.unit
class TestLiteLLMWireContract:
    """Guard the unpinned LiteLLM contract, not just our kwarg.

    litellm.embedding() reads extra_headers out of **kwargs rather than a named
    parameter, so a LiteLLM refactor could silently stop forwarding it while
    every mock-based test above stays green. This asserts the header reaches the
    wire through the real litellm.embedding().
    """

    def test_litellm_forwards_extra_headers_to_the_wire(
        self,
        real_litellm,
    ):
        assert not isinstance(real_litellm.embedding, MagicMock)
        _RecordingHandler.received_headers = {}
        server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = LiteLLMClient(
                model_name="openai/mock-model",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
                embedding_dimension=2,
                token_provider=MagicMock(return_value="wire-token"),
                extra_headers=TENANT_HEADERS,
            )
            result = client.encode(["hello"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert result.shape == (1, 2)
        # Compare lower-cased: header names are case-insensitive on the wire.
        assert _RecordingHandler.received_headers.get("x-tenant-id") == "research"
        assert _RecordingHandler.received_headers.get("authorization") == "Bearer wire-token"
