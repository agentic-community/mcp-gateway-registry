"""
Embeddings client abstraction for vendor-agnostic embeddings generation.

Two providers: local sentence-transformers models, and remote APIs
(OpenAI-compatible endpoints, Azure OpenAI, Amazon Bedrock) called through the
small adapters in adapters.py. The remote provider is still selected with
EMBEDDINGS_PROVIDER=litellm for backward compatibility; LiteLLM itself was
removed in issue #1826.
"""

import logging
import os
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..utils.embeddings_provider import (
    PROVIDER_AZURE,
    PROVIDER_BEDROCK_COHERE,
    PROVIDER_BEDROCK_TITAN,
    PROVIDER_OPENAI,
    RemoteModel,
    resolve_remote_model,
)
from .adapters import (
    AZURE_DEFAULT_API_VERSION,
    _extract_raw_array,
    embed_azure,
    embed_bedrock_cohere,
    embed_bedrock_titan,
    embed_openai,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Defaults for every remote embeddings call (issue #1826). Settings passes the
# operator's EMBEDDINGS_REQUEST_TIMEOUT_SECONDS and EMBEDDINGS_MAX_RETRIES.
DEFAULT_REQUEST_TIMEOUT_SECONDS: int = 30
DEFAULT_MAX_RETRIES: int = 2
# Kept for backward compatibility with callers of the pre-#1826 name.
RAW_ARRAY_REQUEST_TIMEOUT_SECONDS: int = DEFAULT_REQUEST_TIMEOUT_SECONDS

__all__ = [
    "EmbeddingsClient",
    "LiteLLMClient",
    "RemoteEmbeddingsClient",
    "SentenceTransformersClient",
    "_extract_raw_array",
    "create_embeddings_client",
]


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


def _raw_array_model(
    model_name: str,
) -> RemoteModel:
    """Treat any model name as OpenAI-compatible for the raw_array response format.

    Before #1826 the raw_array path POSTed to EMBEDDINGS_API_BASE with the bare
    model name whatever the prefix was, so a custom prefix keeps working there.
    """
    return RemoteModel(PROVIDER_OPENAI, model_name.split("/", 1)[-1])


class RemoteEmbeddingsClient(EmbeddingsClient):
    """Client for remote embeddings APIs: OpenAI-compatible, Azure OpenAI, Amazon Bedrock.

    Selected with EMBEDDINGS_PROVIDER=litellm. The model-name prefix picks the
    adapter: 'openai/', 'azure/', or 'bedrock/'. See utils/embeddings_provider.py.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        aws_region: str | None = None,
        embedding_dimension: int | None = None,
        token_provider: Callable[[], str] | None = None,
        response_format: str = "openai",
        request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        extra_headers: dict[str, str] | None = None,
    ):
        """
        Initialize the remote embeddings client.

        Args:
            model_name: Provider-prefixed model name, for example
                'openai/text-embedding-3-small', 'azure/<deployment>',
                'bedrock/amazon.titan-embed-text-v2:0', 'bedrock/cohere.embed-english-v3'.
            api_key: Static API key (OpenAI-compatible and Azure). Falls back to
                OPENAI_API_KEY or AZURE_API_KEY when unset, as LiteLLM did.
            api_base: Base URL. OpenAI defaults to api.openai.com; Azure requires it
                (or AZURE_API_BASE).
            aws_region: AWS region for Amazon Bedrock.
            embedding_dimension: Expected embedding dimension (validated on first call).
            token_provider: Optional callable returning a fresh bearer token per call
                (EMBEDDINGS_AUTH_MODE=idp). Takes precedence over api_key.
            response_format: "openai" (standard envelope) or "raw_array" (bare array
                of vectors from an OpenAI-compatible endpoint).
            request_timeout_seconds: Timeout for each remote request.
            max_retries: Retries for transient failures (429, 5xx, connection errors).
            extra_headers: Operator-configured headers sent on every OpenAI-compatible
                and Azure request (EMBEDDINGS_EXTRA_HEADER_<N>). The registry's own
                Content-Type, Accept, and credential header always take precedence.

        Raises:
            ValueError: If no adapter supports the model name.

        Note:
            Amazon Bedrock uses the standard AWS credential chain (IAM roles,
            ~/.aws/credentials, environment variables). api_key is not used for it.
        """
        self.model_name = model_name
        self.response_format = response_format.lower()
        if self.response_format == "raw_array":
            self.remote_model = _raw_array_model(model_name)
        else:
            self.remote_model = resolve_remote_model(model_name)
        self.api_key = api_key
        self.api_base = api_base
        self.aws_region = aws_region
        self._embedding_dimension = embedding_dimension
        self._validated_dimension: int | None = None
        self.token_provider = token_provider
        self._request_timeout_seconds = request_timeout_seconds
        self._max_retries = max_retries
        # Copied so a later change to the caller's dict cannot change what this
        # client sends mid-process.
        self.extra_headers: dict[str, str] = dict(extra_headers or {})
        if self.extra_headers:
            # Count only: header names and values stay out of INFO logs.
            logger.info(
                "Embeddings client configured with %d extra header(s)",
                len(self.extra_headers),
            )
        logger.info(
            "Remote embeddings client: provider=%s model=%s",
            self.remote_model.provider,
            self.remote_model.model_id,
        )

    def _credential(
        self,
        env_fallback: str,
    ) -> str | None:
        """Return the IdP token, the static key, or the provider's env var, in that order."""
        if self.token_provider is not None:
            return self.token_provider()
        return self.api_key or os.environ.get(env_fallback) or None

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

    def _embed_openai(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Call an OpenAI-compatible endpoint (also used for raw_array)."""
        api_base = (
            self.api_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        )
        if self.response_format == "raw_array" and not api_base:
            raise RuntimeError(
                "EMBEDDINGS_RESPONSE_FORMAT=raw_array requires EMBEDDINGS_API_BASE "
                "(the OpenAI-compatible endpoint to call directly)."
            )
        return embed_openai(
            texts=texts,
            model_id=self.remote_model.model_id,
            api_base=api_base,
            api_key=self._credential("OPENAI_API_KEY"),
            extra_headers=self.extra_headers,
            response_format=self.response_format,
            timeout_seconds=self._request_timeout_seconds,
            max_retries=self._max_retries,
        )

    def _embed_azure(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Call an Azure OpenAI deployment."""
        api_base = self.api_base or os.environ.get("AZURE_API_BASE")
        if not api_base:
            raise RuntimeError(
                "Azure OpenAI embeddings require EMBEDDINGS_API_BASE (or AZURE_API_BASE), "
                "for example https://<resource>.openai.azure.com"
            )
        return embed_azure(
            texts=texts,
            deployment=self.remote_model.model_id,
            api_base=api_base,
            api_key=self._credential("AZURE_API_KEY"),
            api_version=os.environ.get("AZURE_API_VERSION") or AZURE_DEFAULT_API_VERSION,
            extra_headers=self.extra_headers,
            timeout_seconds=self._request_timeout_seconds,
            max_retries=self._max_retries,
        )

    def _embed_bedrock(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Call Amazon Bedrock with the request shape the model family expects."""
        adapter = (
            embed_bedrock_titan
            if self.remote_model.provider == PROVIDER_BEDROCK_TITAN
            else embed_bedrock_cohere
        )
        return adapter(
            texts=texts,
            model_id=self.remote_model.model_id,
            region=self.aws_region,
            timeout_seconds=self._request_timeout_seconds,
            max_retries=self._max_retries,
        )

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Generate embeddings with the remote API for this model.

        Args:
            texts: List of text strings to encode

        Returns:
            NumPy array of embeddings, one row per input text

        Raises:
            RuntimeError: If the request fails or the response cannot be parsed
        """
        provider = self.remote_model.provider
        try:
            if provider == PROVIDER_OPENAI:
                vectors = self._embed_openai(texts)
            elif provider == PROVIDER_AZURE:
                vectors = self._embed_azure(texts)
            elif provider in (PROVIDER_BEDROCK_TITAN, PROVIDER_BEDROCK_COHERE):
                vectors = self._embed_bedrock(texts)
            else:
                raise RuntimeError(f"No embeddings adapter for provider {provider!r}")
            embeddings_array = np.array(vectors, dtype=np.float32)
        except Exception as e:
            logger.error(
                "Failed to generate embeddings via %s (%s): %s",
                provider,
                self.remote_model.model_id,
                e,
            )
            raise RuntimeError(f"Failed to generate embeddings via {provider}: {e}") from e

        self._record_dimension(embeddings_array)
        logger.debug(
            "Generated %d embeddings with dimension %s",
            len(vectors),
            self._validated_dimension,
        )
        return embeddings_array

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            Integer dimension of embedding vectors

        Raises:
            RuntimeError: If dimension cannot be determined
        """
        if self._validated_dimension is not None:
            return self._validated_dimension

        if self._embedding_dimension is not None:
            return self._embedding_dimension

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


# Pre-#1826 name. Kept so existing imports keep working.
LiteLLMClient = RemoteEmbeddingsClient


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
    extra_headers: dict[str, str] | None = None,
    request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> EmbeddingsClient:
    """
    Factory function to create an embeddings client based on provider.

    Args:
        provider: Provider type: 'sentence-transformers' (local) or 'litellm' (remote
            API; the value predates the removal of LiteLLM in issue #1826)
        model_name: Model identifier
        model_dir: Optional local model directory (sentence-transformers only)
        cache_dir: Optional cache directory (sentence-transformers only)
        api_key: Optional API key (remote only)
        api_base: Optional API base URL (remote only)
        aws_region: Optional AWS region (remote, Amazon Bedrock only)
        embedding_dimension: Optional embedding dimension
        token_provider: Optional callable returning a fresh bearer token (litellm only)
        response_format: Endpoint response shape, "openai" or "raw_array" (litellm only)
        extra_headers: Optional extra request headers (remote only). Ignored with a
            warning for sentence-transformers, which makes no HTTP call.
        request_timeout_seconds: Timeout for each remote request
        max_retries: Retries for transient remote failures

    Returns:
        EmbeddingsClient instance

    Raises:
        ValueError: If the provider or the remote model name is not supported

    Note:
        For Amazon Bedrock, AWS credentials come from the standard AWS credential
        chain (IAM roles, environment variables, ~/.aws/credentials).
    """
    provider_lower = provider.lower()

    if provider_lower == "sentence-transformers":
        if extra_headers:
            logger.warning(
                "EMBEDDINGS_EXTRA_HEADER_<N> is set (%d header(s)) but is ignored for "
                "provider 'sentence-transformers', which makes no HTTP call. Set "
                "EMBEDDINGS_PROVIDER=litellm to use them.",
                len(extra_headers),
            )
        logger.info(f"Creating SentenceTransformersClient with model: {model_name}")
        return SentenceTransformersClient(
            model_name=model_name,
            model_dir=model_dir,
            cache_dir=cache_dir,
        )

    elif provider_lower == "litellm":
        logger.info(f"Creating RemoteEmbeddingsClient with model: {model_name}")
        return RemoteEmbeddingsClient(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            aws_region=aws_region,
            embedding_dimension=embedding_dimension,
            token_provider=token_provider,
            response_format=response_format,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            extra_headers=extra_headers,
        )

    else:
        raise ValueError(
            f"Unsupported embeddings provider: {provider}. "
            "Supported providers: 'sentence-transformers', 'litellm'"
        )
