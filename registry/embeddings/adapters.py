"""HTTP and Amazon Bedrock adapters for remote embeddings (issue #1826).

These replace the single litellm.embedding() call. Each adapter sends the same
request LiteLLM 1.93 sent, so vectors stay comparable with the ones already
stored in every deployment. The parity notes live beside each adapter.

The adapters are plain functions: they take every value they need as an
argument, read no settings, and return a list of vectors in input order.
"""

import json
import logging
import time
from functools import lru_cache
from typing import Any

import httpx

from ..constants import REGISTRY_MANAGED_EMBEDDINGS_HEADERS

logger = logging.getLogger(__name__)

OPENAI_DEFAULT_API_BASE: str = "https://api.openai.com/v1"
# Same default LiteLLM 1.93 used when AZURE_API_VERSION is unset.
AZURE_DEFAULT_API_VERSION: str = "2025-02-01-preview"
# Cohere on Amazon Bedrock: LiteLLM always sent "search_document", for queries
# too. Changing it would change query vectors against stored document vectors.
BEDROCK_COHERE_INPUT_TYPE: str = "search_document"

# Retried the way the OpenAI SDK under LiteLLM retried them.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})
BACKOFF_BASE_SECONDS: float = 0.5
BACKOFF_MAX_SECONDS: float = 8.0
BEDROCK_CONNECT_TIMEOUT_SECONDS: int = 10
ERROR_BODY_EXCERPT_CHARS: int = 300

_RAW_ARRAY_KEYS: tuple[str, ...] = ("embeddings", "data", "vectors")


def _extract_raw_array(
    payload: Any,
) -> list[list[float]]:
    """Extract a list of embedding vectors from a non-OpenAI "raw array" response.

    Some OpenAI-compatible endpoints return the vectors as a bare JSON array of
    arrays (``[[0.1, ...], [0.2, ...]]``) instead of the standard
    ``{"data": [{"embedding": [...]}]}`` envelope. A few wrap the same bare array
    under a top-level key. This helper accepts:

      - a bare list of vectors: ``[[...], [...]]``
      - a dict with the list of vectors under "embeddings", "data", or "vectors"

    Args:
        payload: Parsed JSON body returned by the embeddings endpoint.

    Returns:
        List of embedding vectors (list of lists of floats).

    Raises:
        ValueError: If no list of numeric vectors can be found.
    """
    vectors = payload
    if isinstance(payload, dict):
        vectors = None
        for key in _RAW_ARRAY_KEYS:
            if isinstance(payload.get(key), list):
                vectors = payload[key]
                break
        if vectors is None:
            raise ValueError(
                "raw_array response is a dict without an "
                "'embeddings'/'data'/'vectors' list; "
                f"got keys {sorted(payload.keys())}"
            )

    if not isinstance(vectors, list) or not vectors:
        raise ValueError("raw_array response did not contain a non-empty list of vectors")
    if not all(isinstance(vector, list) for vector in vectors):
        raise ValueError(
            "raw_array response must be a list of vectors (list of lists); "
            "got a list containing non-list items"
        )
    return vectors


def _extract_openai_envelope(
    payload: Any,
) -> list[list[float]]:
    """Return the vectors from a standard OpenAI ``{"data": [...]}`` response, in input order."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("embeddings response has no 'data' list (expected the OpenAI envelope)")
    items = sorted(payload["data"], key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in items]


def _build_headers(
    extra_headers: dict[str, str] | None,
    auth_header_name: str,
    auth_value: str | None,
) -> dict[str, str]:
    """Build request headers with operator headers first and managed headers last.

    Operator headers (EMBEDDINGS_EXTRA_HEADER_<N>, issue #1819) go in first, minus
    any registry-managed name in any casing: HTTP header names ignore case but
    dict keys do not, so "content-type" would otherwise be sent alongside
    "Content-Type". Settings already rejects managed names, so this is a backstop.
    """
    headers: dict[str, str] = {}
    for name, value in (extra_headers or {}).items():
        if name.lower() not in REGISTRY_MANAGED_EMBEDDINGS_HEADERS:
            headers[name] = value
    headers["Content-Type"] = "application/json"
    headers["Accept"] = "application/json"
    if auth_value:
        headers[auth_header_name] = auth_value
    return headers


def _backoff_seconds(
    attempt: int,
) -> float:
    """Return the exponential backoff delay before retry number ``attempt`` (0-based)."""
    return min(BACKOFF_BASE_SECONDS * (2**attempt), BACKOFF_MAX_SECONDS)


def _status_error(
    response: httpx.Response,
) -> RuntimeError:
    """Build the error for a non-retryable or final HTTP status, with a short body excerpt."""
    excerpt = response.text[:ERROR_BODY_EXCERPT_CHARS]
    return RuntimeError(f"Embeddings endpoint returned HTTP {response.status_code}: {excerpt}")


def _post_with_retries(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_seconds: float,
    max_retries: int,
) -> Any:
    """POST JSON and return the parsed response, retrying transient failures.

    Retries on 408/409/429/5xx and on connection or timeout errors, with
    exponential backoff, up to ``max_retries`` times. Any other status fails at
    once, so a bad key or a too-long input is not retried.

    Raises:
        RuntimeError: On a non-retryable status, or when retries run out.
    """
    attempt = 0
    while True:
        try:
            response = httpx.post(url, json=body, headers=headers, timeout=timeout_seconds)
        except httpx.TransportError as e:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Embeddings request failed after {attempt + 1} attempt(s): {e}"
                ) from e
            logger.warning("Embeddings request error (%s); retrying", type(e).__name__)
        else:
            if response.status_code < 400:
                return response.json()
            if response.status_code not in RETRYABLE_STATUS_CODES or attempt >= max_retries:
                raise _status_error(response)
            logger.warning("Embeddings endpoint returned HTTP %d; retrying", response.status_code)
        time.sleep(_backoff_seconds(attempt))
        attempt += 1


def _parse_bedrock_body(
    response: dict[str, Any],
) -> dict[str, Any]:
    """Read and JSON-decode the streaming body of a Bedrock invoke_model response."""
    return json.loads(response["body"].read())


@lru_cache(maxsize=8)
def _bedrock_client(
    region: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> Any:
    """Return a cached bedrock-runtime client for one region and retry setting.

    Credentials come from the standard AWS chain (task role, instance profile,
    IRSA, environment, ~/.aws), so this code never handles a Bedrock credential.
    botocore's "standard" retry mode retries throttling and transient errors.
    """
    import boto3
    from botocore.config import Config

    config = Config(
        connect_timeout=min(BEDROCK_CONNECT_TIMEOUT_SECONDS, timeout_seconds),
        read_timeout=timeout_seconds,
        retries={"mode": "standard", "max_attempts": max_retries + 1},
    )
    return boto3.client("bedrock-runtime", region_name=region, config=config)


def _invoke_bedrock(
    client: Any,
    model_id: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Call invoke_model and return the decoded JSON, wrapping AWS errors."""
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
    )

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
    except ClientError as e:
        error = e.response.get("Error", {})
        raise RuntimeError(
            f"Amazon Bedrock {model_id} returned {error.get('Code')}: {error.get('Message')}"
        ) from e
    except BotoCoreError as e:
        raise RuntimeError(f"Amazon Bedrock {model_id} request failed: {e}") from e
    return _parse_bedrock_body(response)


def _titan_vector(
    payload: dict[str, Any],
) -> list[float]:
    """Return the float vector from a Titan response (v2 adds embeddingsByType)."""
    by_type = payload.get("embeddingsByType") or {}
    if isinstance(by_type.get("float"), list):
        return by_type["float"]
    if isinstance(payload.get("embedding"), list):
        return payload["embedding"]
    raise ValueError(f"Titan response has no embedding; got keys {sorted(payload)}")


def _cohere_vectors(
    payload: dict[str, Any],
) -> list[list[float]]:
    """Return the float vectors from a Cohere response (a list, or a dict keyed by type)."""
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, dict):
        embeddings = embeddings.get("float")
    if not isinstance(embeddings, list):
        raise ValueError(f"Cohere response has no float embeddings; got keys {sorted(payload)}")
    return embeddings


def embed_openai(
    texts: list[str],
    model_id: str,
    api_base: str | None,
    api_key: str | None,
    extra_headers: dict[str, str] | None,
    response_format: str,
    timeout_seconds: float,
    max_retries: int,
) -> list[list[float]]:
    """Embed texts with an OpenAI-compatible ``/embeddings`` endpoint.

    Parity with LiteLLM's openai provider: POST ``{api_base}/embeddings`` with
    ``Authorization: Bearer <key>`` and body ``{"model": ..., "input": [...]}``.
    LiteLLM also sent ``encoding_format: "float"``, which is the API default; it is
    left out so strict endpoints that accept only model and input work.

    Args:
        texts: Strings to embed.
        model_id: Model name without the 'openai/' prefix.
        api_base: Base URL ending in /v1; defaults to the OpenAI API.
        api_key: Bearer credential (static key or IdP token), or None.
        extra_headers: Operator-configured headers (issue #1819).
        response_format: "openai" for the standard envelope, "raw_array" for a bare
            array of vectors.
        timeout_seconds: Per-request timeout.
        max_retries: Retries for transient failures.

    Returns:
        One vector per input text, in input order.
    """
    url = (api_base or OPENAI_DEFAULT_API_BASE).rstrip("/") + "/embeddings"
    auth = f"Bearer {api_key}" if api_key else None
    headers = _build_headers(extra_headers, "Authorization", auth)
    logger.debug("Calling OpenAI-compatible embeddings endpoint: %s", url)
    payload = _post_with_retries(
        url, headers, {"model": model_id, "input": texts}, timeout_seconds, max_retries
    )
    if response_format == "raw_array":
        return _extract_raw_array(payload)
    return _extract_openai_envelope(payload)


def embed_azure(
    texts: list[str],
    deployment: str,
    api_base: str,
    api_key: str | None,
    api_version: str,
    extra_headers: dict[str, str] | None,
    timeout_seconds: float,
    max_retries: int,
) -> list[list[float]]:
    """Embed texts with an Azure OpenAI deployment.

    Parity with LiteLLM's azure provider: POST
    ``{api_base}/openai/deployments/{deployment}/embeddings?api-version=...`` with
    the credential in an ``api-key`` header, body ``{"model": deployment, "input": [...]}``.

    Returns:
        One vector per input text, in input order.
    """
    url = (
        f"{api_base.rstrip('/')}/openai/deployments/{deployment}/embeddings"
        f"?api-version={api_version}"
    )
    headers = _build_headers(extra_headers, "api-key", api_key)
    logger.debug("Calling Azure OpenAI embeddings deployment: %s", deployment)
    payload = _post_with_retries(
        url, headers, {"model": deployment, "input": texts}, timeout_seconds, max_retries
    )
    return _extract_openai_envelope(payload)


def embed_bedrock_titan(
    texts: list[str],
    model_id: str,
    region: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> list[list[float]]:
    """Embed texts with an Amazon Titan text embeddings model on Amazon Bedrock.

    Parity with LiteLLM: Titan accepts one text per request, so this makes one
    ``invoke_model`` call per text with body ``{"inputText": text}``. Calls run one
    after another: indexing and search embed a single text per call, so a batch is
    almost always size one.

    Returns:
        One vector per input text, in input order.
    """
    client = _bedrock_client(region, timeout_seconds, max_retries)
    vectors: list[list[float]] = []
    for text in texts:
        payload = _invoke_bedrock(client, model_id, {"inputText": text})
        vectors.append(_titan_vector(payload))
    return vectors


def embed_bedrock_cohere(
    texts: list[str],
    model_id: str,
    region: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> list[list[float]]:
    """Embed texts with a Cohere embed model on Amazon Bedrock, in one batched call.

    Parity with LiteLLM: body ``{"texts": [...], "input_type": "search_document"}``,
    the same input_type for documents and queries.

    Returns:
        One vector per input text, in input order.
    """
    client = _bedrock_client(region, timeout_seconds, max_retries)
    body = {"texts": texts, "input_type": BEDROCK_COHERE_INPUT_TYPE}
    return _cohere_vectors(_invoke_bedrock(client, model_id, body))
