"""Mock OpenAI-compatible embedding HTTP server for IdP auth integration testing.

Returns deterministic hardcoded embeddings. Validates that a Bearer token is
present in the Authorization header (does not verify the token itself -- that's
the IdP's job; this just confirms the registry is sending one).

Usage (standalone):
    uv run python tests/fixtures/mocks/mock_embedding_server.py --port 8900

The server exposes:
    POST /v1/embeddings  (OpenAI-compatible)
    GET  /health

Use with Keycloak as IdP: configure the registry with:
    EMBEDDINGS_PROVIDER=litellm
    EMBEDDINGS_MODEL_NAME=openai/mock-embedding-model
    EMBEDDINGS_API_BASE=http://localhost:8900/v1
    EMBEDDINGS_MODEL_DIMENSIONS=384
    EMBEDDINGS_AUTH_MODE=idp
    EMBEDDINGS_IDP_TOKEN_ENDPOINT=https://localhost:8443/realms/mcp-gateway/protocol/openid-connect/token
    EMBEDDINGS_IDP_CLIENT_ID=embeddings-client
    EMBEDDINGS_IDP_CLIENT_SECRET=<your-client-secret>
"""

import argparse
import hashlib
import json
import logging
import random
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION: int = 384


def _deterministic_embedding(text: str) -> list[float]:
    """Generate a deterministic embedding from text hash."""
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    seed = int(text_hash[:8], 16)
    rng = random.Random(seed)  # nosec B311 - deterministic test embeddings, not cryptographic
    raw = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIMENSION)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


class EmbeddingHandler(BaseHTTPRequestHandler):
    """Handler for OpenAI-compatible /v1/embeddings endpoint."""

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/embeddings":
            self._respond(404, {"error": "not found"})
            return

        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning("Request missing Bearer token")
            self._respond(
                401,
                {
                    "error": {
                        "message": "Missing or invalid Authorization header",
                        "type": "auth_error",
                    }
                },
            )
            return

        token = auth_header[len("Bearer ") :]
        logger.info("Received embedding request with token (length=%d)", len(token))

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            request = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"error": {"message": "Invalid JSON", "type": "invalid_request"}})
            return

        input_texts = request.get("input", [])
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        model = request.get("model", "mock-embedding-model")

        embeddings_data = []
        for i, text in enumerate(input_texts):
            embeddings_data.append(
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": _deterministic_embedding(text),
                }
            )

        response = {
            "object": "list",
            "data": embeddings_data,
            "model": model,
            "usage": {
                "prompt_tokens": sum(len(t.split()) for t in input_texts),
                "total_tokens": sum(len(t.split()) for t in input_texts),
            },
        }

        logger.info("Returning %d embeddings (dim=%d)", len(embeddings_data), EMBEDDING_DIMENSION)
        self._respond(200, response)

    def _respond(self, status: int, body: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        response_bytes = json.dumps(body).encode()
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def log_message(self, format: str, *args) -> None:
        logger.debug(format, *args)


def run_server(port: int = 8900) -> None:
    """Start the mock embedding server."""
    server = HTTPServer(("127.0.0.1", port), EmbeddingHandler)
    logger.info("Mock embedding server started on http://127.0.0.1:%d", port)
    logger.info("Endpoints: POST /v1/embeddings, GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down mock embedding server")
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock OpenAI-compatible embedding server")
    parser.add_argument("--port", type=int, default=8900, help="Port to listen on (default: 8900)")
    args = parser.parse_args()
    run_server(port=args.port)
