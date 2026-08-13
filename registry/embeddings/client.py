"""
Embeddings client abstraction for vendor-agnostic embeddings generation.

This module provides a unified interface for generating embeddings from multiple
providers including local sentence-transformers models and cloud-based APIs via LiteLLM.
"""

import logging
import os
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default timeout for the direct raw_array HTTP call (litellm handles its own).
RAW_ARRAY_REQUEST_TIMEOUT_SECONDS: int = 30
# Top-level keys a raw_array endpoint might wrap its list of vectors under.
_RAW_ARRAY_KEYS: tuple[str, ...] = ("embeddings", "data", "vectors")


def _extract_raw_array(
    payload: Any,
) -> list[list[float]]:
    """Extract a list of embedding vectors from a non-OpenAI "raw array" response.

    Some OpenAI-compatible endpoints return the vectors as a bare JSON array of
    arrays (``[[0.1, ...], [0.2, ...]]``) instead of the standard
    ``{"data": [{"embedding": [...]}]}`` envelope that litellm parses. A few wrap
    the same bare array under a top-level key. This helper accepts:

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
            "got a flat or malformed structure. If your endpoint returns the "
            "OpenAI envelope, set EMBEDDINGS_RESPONSE_FORMAT=openai instead."
        )
    return vectors


class EmbeddingsClient(ABC):
    """Abstract base class for embeddings generation clients."""

    @abstractmethod
    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to encode

        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dimension)

        Raises:
            RuntimeError: If encoding fails
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this client.

        Returns:
            Integer dimension of embedding vectors
        """
        pass


class SentenceTransformersClient(EmbeddingsClient):
    """Client for local sentence-transformers models."""

    def __init__(
        self,
        model_name: str,
        model_dir: Path | None = None,
        cache_dir: Path | None = None,
    ):
        """
        Initialize the SentenceTransformers client.

        Args:
            model_name: Name of the sentence-transformers model
            model_dir: Optional local directory containing the model
            cache_dir: Optional cache directory for downloaded models
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None
        self._load_error: RuntimeError | None = None

    def _load_model(self) -> None:
        """Load the sentence-transformers model."""
        if self._model is not None:
            return

        # If a previous load attempt failed, raise the cached error immediately
        # to avoid repeated download attempts (e.g., hitting HuggingFace on every call)
        if self._load_error is not None:
            raise self._load_error

        try:
            from sentence_transformers import SentenceTransformer

            # Set cache directory if provided
            original_st_home = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(self.cache_dir)

            # Check if local model exists
            model_exists = (
                self.model_dir.exists() and any(self.model_dir.iterdir())
                if self.model_dir and self.model_dir.exists()
                else False
            )

            if model_exists:
                logger.info(f"Loading SentenceTransformer model from local path: {self.model_dir}")
                self._model = SentenceTransformer(str(self.model_dir))
            else:
                logger.info(
                    f"Local model not found, downloading from Hugging Face: {self.model_name}"
                )
                self._model = SentenceTransformer(self.model_name)

            # Restore original environment variable
            if original_st_home:
                os.environ["SENTENCE_TRANSFORMERS_HOME"] = original_st_home
            elif "SENTENCE_TRANSFORMERS_HOME" in os.environ:
                del os.environ["SENTENCE_TRANSFORMERS_HOME"]

            # Get embedding dimension
            self._dimension = self._model.get_sentence_embedding_dimension()

            logger.info(
                f"SentenceTransformer model loaded successfully. Dimension: {self._dimension}"
            )

        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
            self._load_error = RuntimeError(f"Failed to load SentenceTransformer model: {e}")
            raise self._load_error from e

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Generate embeddings using sentence-transformers.

        Args:
            texts: List of text strings to encode

        Returns:
            NumPy array of embeddings

        Raises:
            RuntimeError: If encoding fails
        """
        if self._model is None:
            self._load_model()

        try:
            embeddings = self._model.encode(texts)
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}", exc_info=True)
            raise RuntimeError(f"Failed to encode texts: {e}") from e

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            Integer dimension of embedding vectors

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._dimension is None:
            self._load_model()
        return self._dimension


class LiteLLMClient(EmbeddingsClient):
    """Client for cloud-based embeddings via LiteLLM."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        aws_region: str | None = None,
        embedding_dimension: int | None = None,
        token_provider: Callable[[], str] | None = None,
        response_format: str = "openai",
        request_timeout_seconds: int = RAW_ARRAY_REQUEST_TIMEOUT_SECONDS,
    ):
        """
        Initialize the LiteLLM client.

        Args:
            model_name: LiteLLM model identifier (e.g., 'bedrock/amazon.titan-embed-text-v1',
                       'openai/text-embedding-3-small', 'cohere/embed-english-v3.0')
            api_key: Optional API key for the provider
            api_base: Optional API base URL for the provider
            aws_region: Optional AWS region for Bedrock
            embedding_dimension: Expected embedding dimension (will be validated)
            token_provider: Optional callable that returns a fresh bearer token.
                When provided, the token is injected per-call via api_key instead
                of setting a static environment variable.
            response_format: Response shape of the endpoint. "openai" (default) uses
                litellm to parse the standard envelope; "raw_array" bypasses litellm
                and reads a bare array-of-vectors response directly.
            request_timeout_seconds: Timeout for the direct raw_array HTTP call.

        Note:
            For AWS Bedrock, this client uses the standard AWS credential chain
            (IAM roles, ~/.aws/credentials, environment variables). The api_key
            parameter is not used for Bedrock authentication.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.aws_region = aws_region
        self._embedding_dimension = embedding_dimension
        self._validated_dimension: int | None = None
        self.token_provider = token_provider
        self.response_format = response_format.lower()
        self._request_timeout_seconds = request_timeout_seconds

        # Only set a static API key env var when NOT using a dynamic token provider.
        if self.api_key and self.token_provider is None:
            self._set_api_key_env()
        if self.aws_region:
            os.environ["AWS_REGION_NAME"] = self.aws_region

    def _set_api_key_env(self) -> None:
        """Set the appropriate API key environment variable based on provider."""
        provider = self.model_name.split("/")[0].lower()

        # AWS Bedrock uses standard AWS credential chain (IAM roles, env vars, ~/.aws/credentials)
        # No need to set API key environment variable for Bedrock
        if provider == "bedrock":
            logger.info("Using standard AWS credential chain for Bedrock authentication")
            return

        # Handle other providers with API keys
        env_var_mapping = {
            "openai": "OPENAI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "azure": "AZURE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }

        env_var = env_var_mapping.get(provider)
        if env_var and self.api_key:
            os.environ[env_var] = self.api_key
            logger.debug(f"Set {env_var} environment variable for {provider}")

    def _record_dimension(
        self,
        embeddings_array: np.ndarray,
    ) -> None:
        """Record the embedding dimension on the first call and warn on mismatch."""
        if self._validated_dimension is not None:
            return
        self._validated_dimension = embeddings_array.shape[1]
        if self._embedding_dimension and self._validated_dimension != self._embedding_dimension:
            logger.warning(
                f"Embedding dimension mismatch: expected {self._embedding_dimension}, "
                f"got {self._validated_dimension}"
            )

    def _encode_raw_array(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Call the endpoint directly and read a bare array-of-vectors response.

        Used when response_format == "raw_array". litellm cannot parse a
        non-envelope response, so we POST the OpenAI-style request ourselves and
        extract the vectors via _extract_raw_array. The token (IdP bearer or
        static api_key) is injected exactly as in the litellm path. api_base is
        operator-configured trusted config, not user input.
        """
        import httpx

        if not self.api_base:
            raise RuntimeError(
                "EMBEDDINGS_RESPONSE_FORMAT=raw_array requires EMBEDDINGS_API_BASE "
                "(the OpenAI-compatible endpoint to call directly)."
            )
        url = self.api_base.rstrip("/") + "/embeddings"
        # Endpoints expect the bare model name, not litellm's 'provider/model' form.
        model = self.model_name.split("/", 1)[-1]
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        token = self.token_provider() if self.token_provider is not None else self.api_key
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            logger.debug("Calling raw_array embeddings endpoint: %s", url)
            response = httpx.post(
                url,
                json={"input": texts, "model": model},
                headers=headers,
                timeout=self._request_timeout_seconds,
            )
            response.raise_for_status()
            vectors = _extract_raw_array(response.json())
            embeddings_array = np.array(vectors, dtype=np.float32)
        except httpx.HTTPStatusError as e:
            logger.error(
                "raw_array embeddings endpoint returned status %s",
                e.response.status_code,
                exc_info=True,
            )
            raise RuntimeError(
                f"raw_array embeddings request failed with status {e.response.status_code}"
            ) from e
        except Exception as e:
            logger.error(
                f"Failed to generate embeddings via raw_array endpoint: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to generate embeddings via raw_array endpoint: {e}") from e

        self._record_dimension(embeddings_array)
        return embeddings_array

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Generate embeddings using LiteLLM.

        Args:
            texts: List of text strings to encode

        Returns:
            NumPy array of embeddings

        Raises:
            RuntimeError: If encoding fails or LiteLLM is not installed
        """
        if self.response_format == "raw_array":
            return self._encode_raw_array(texts)

        try:
            from litellm import embedding
        except ImportError as e:
            logger.error("LiteLLM is not installed. Install it with: uv add litellm")
            raise RuntimeError("LiteLLM is not installed. Install it with: uv add litellm") from e

        try:
            # LiteLLM expects 'input' parameter
            kwargs: dict[str, Any] = {"model": self.model_name, "input": texts}

            if self.api_base:
                kwargs["api_base"] = self.api_base

            if self.token_provider is not None:
                kwargs["api_key"] = self.token_provider()

            logger.debug(f"Calling LiteLLM embedding API with model: {self.model_name}")
            response = embedding(**kwargs)

            # Extract embeddings from response
            embeddings_list = [item["embedding"] for item in response["data"]]
            embeddings_array = np.array(embeddings_list, dtype=np.float32)

            self._record_dimension(embeddings_array)

            logger.debug(
                f"Generated {len(embeddings_list)} embeddings with dimension {self._validated_dimension}"
            )
            return embeddings_array

        except Exception as e:
            logger.error(f"Failed to generate embeddings via LiteLLM: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate embeddings via LiteLLM: {e}") from e

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            Integer dimension of embedding vectors

        Raises:
            RuntimeError: If dimension cannot be determined
        """
        # If we have a validated dimension from actual API calls, use that
        if self._validated_dimension is not None:
            return self._validated_dimension

        # Otherwise, use the configured dimension if provided
        if self._embedding_dimension is not None:
            return self._embedding_dimension

        # As a last resort, make a test call with a simple string
        logger.info("Embedding dimension not known, making test call to determine dimension")
        try:
            test_embedding = self.encode(["test"])
            return test_embedding.shape[1]
        except Exception as e:
            logger.error(f"Failed to determine embedding dimension: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to determine embedding dimension: {e}. "
                "Consider setting EMBEDDINGS_DIMENSION in configuration."
            ) from e


def create_embeddings_client(
    provider: str,
    model_name: str,
    model_dir: Path | None = None,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    aws_region: str | None = None,
    embedding_dimension: int | None = None,
    token_provider: Callable[[], str] | None = None,
    response_format: str = "openai",
) -> EmbeddingsClient:
    """
    Factory function to create an embeddings client based on provider.

    Args:
        provider: Provider type ('sentence-transformers' or 'litellm')
        model_name: Model identifier
        model_dir: Optional local model directory (sentence-transformers only)
        cache_dir: Optional cache directory (sentence-transformers only)
        api_key: Optional API key (litellm only)
        api_base: Optional API base URL (litellm only)
        aws_region: Optional AWS region (litellm with Bedrock only)
        embedding_dimension: Optional embedding dimension
        token_provider: Optional callable returning a fresh bearer token (litellm only)
        response_format: Endpoint response shape, "openai" or "raw_array" (litellm only)

    Returns:
        EmbeddingsClient instance

    Raises:
        ValueError: If provider is not supported

    Note:
        For AWS Bedrock, AWS credentials should be configured via standard AWS
        credential chain (IAM roles, environment variables, ~/.aws/credentials).
    """
    provider_lower = provider.lower()

    if provider_lower == "sentence-transformers":
        logger.info(f"Creating SentenceTransformersClient with model: {model_name}")
        return SentenceTransformersClient(
            model_name=model_name,
            model_dir=model_dir,
            cache_dir=cache_dir,
        )

    elif provider_lower == "litellm":
        # Validate that model name has provider prefix
        if "/" not in model_name:
            raise ValueError(
                f"Invalid model name for LiteLLM provider: '{model_name}'. "
                f"LiteLLM requires provider-prefixed model names. "
                f"Examples: 'openai/text-embedding-3-small', 'bedrock/amazon.titan-embed-text-v1', "
                f"'cohere/embed-english-v3.0'. "
                f"If you want to use '{model_name}', set EMBEDDINGS_PROVIDER=sentence-transformers"
            )

        logger.info(f"Creating LiteLLMClient with model: {model_name}")
        return LiteLLMClient(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            aws_region=aws_region,
            embedding_dimension=embedding_dimension,
            token_provider=token_provider,
            response_format=response_format,
        )

    else:
        raise ValueError(
            f"Unsupported embeddings provider: {provider}. "
            "Supported providers: 'sentence-transformers', 'litellm'"
        )
