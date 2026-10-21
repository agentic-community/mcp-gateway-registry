"""Map an EMBEDDINGS_MODEL_NAME to the remote API that serves it.

Kept free of heavy imports (no numpy, httpx, or boto3) so Settings can call
resolve_remote_model() at startup and reject an unsupported model name before
the first embedding call. Issue #1826 replaced LiteLLM with our own adapters;
this module is the single list of model-name prefixes those adapters accept.
"""

from typing import NamedTuple

OPENAI_PREFIX: str = "openai/"
AZURE_PREFIX: str = "azure/"
BEDROCK_PREFIX: str = "bedrock/"

PROVIDER_OPENAI: str = "openai"
PROVIDER_AZURE: str = "azure"
PROVIDER_BEDROCK_TITAN: str = "bedrock-titan"
PROVIDER_BEDROCK_COHERE: str = "bedrock-cohere"

# Bedrock model IDs the adapter knows how to call. Titan takes one text per
# request ({"inputText": ...}); Cohere takes a batch ({"texts": [...]}).
BEDROCK_TITAN_MODEL_PREFIXES: tuple[str, ...] = (
    "amazon.titan-embed-text-",
    "amazon.titan-embed-g1-text-",
)
BEDROCK_COHERE_MODEL_PREFIX: str = "cohere.embed-"

# Prefixes LiteLLM accepted that the registry no longer calls directly.
# Each maps to the replacement named in the startup error.
REMOVED_PREFIX_REPLACEMENTS: dict[str, str] = {
    "cohere/": "bedrock/cohere.embed-english-v3 (Cohere on Amazon Bedrock)",
    "voyage/": "an OpenAI-compatible endpoint: openai/<model> with EMBEDDINGS_API_BASE",
    "voyage-": "an OpenAI-compatible endpoint: openai/<model> with EMBEDDINGS_API_BASE",
}

SUPPORTED_MODEL_EXAMPLES: str = (
    "openai/text-embedding-3-small, azure/<deployment-name>, "
    "bedrock/amazon.titan-embed-text-v2:0, bedrock/cohere.embed-english-v3"
)


class RemoteModel(NamedTuple):
    """A model name split into the provider that serves it and its native ID."""

    provider: str
    model_id: str


def _unsupported_error(
    model_name: str,
    detail: str,
) -> ValueError:
    """Build the error raised for a model name no adapter can serve."""
    return ValueError(
        f"EMBEDDINGS_MODEL_NAME={model_name!r} is not supported: {detail} "
        f"Supported forms: {SUPPORTED_MODEL_EXAMPLES}. Any OpenAI-compatible "
        "embeddings endpoint (vLLM, Ollama, Mistral, a gateway) works as "
        "openai/<model> with EMBEDDINGS_API_BASE set to its base URL."
    )


def _resolve_bedrock(
    model_name: str,
    model_id: str,
) -> RemoteModel:
    """Pick the Bedrock request shape for a bedrock/<model-id> name."""
    if model_id.startswith(BEDROCK_TITAN_MODEL_PREFIXES):
        return RemoteModel(PROVIDER_BEDROCK_TITAN, model_id)
    if model_id.startswith(BEDROCK_COHERE_MODEL_PREFIX):
        return RemoteModel(PROVIDER_BEDROCK_COHERE, model_id)
    raise _unsupported_error(
        model_name,
        "only Amazon Titan text embeddings and Cohere embed models are supported on "
        "Amazon Bedrock.",
    )


def resolve_remote_model(
    model_name: str,
) -> RemoteModel:
    """Return the provider and native model ID for a remote embeddings model.

    Args:
        model_name: EMBEDDINGS_MODEL_NAME, for example 'openai/text-embedding-3-small'.

    Returns:
        RemoteModel with the provider constant and the model ID the provider expects
        (the prefix removed).

    Raises:
        ValueError: If no adapter serves this model name. The message names the
            replacement for prefixes LiteLLM used to accept.
    """
    name = (model_name or "").strip()
    lowered = name.lower()
    for prefix, replacement in REMOVED_PREFIX_REPLACEMENTS.items():
        if lowered.startswith(prefix):
            detail = f"the {prefix!r} provider was removed with LiteLLM; use {replacement}."
            raise _unsupported_error(name, detail)

    for prefix, provider in ((OPENAI_PREFIX, PROVIDER_OPENAI), (AZURE_PREFIX, PROVIDER_AZURE)):
        if lowered.startswith(prefix):
            model_id = name[len(prefix) :]
            if not model_id:
                raise _unsupported_error(name, f"nothing follows {prefix!r}.")
            return RemoteModel(provider, model_id)

    if lowered.startswith(BEDROCK_PREFIX):
        return _resolve_bedrock(name, name[len(BEDROCK_PREFIX) :])

    if "/" not in name:
        raise _unsupported_error(
            name,
            "the model name has no provider prefix. If you meant a local "
            "sentence-transformers model, set EMBEDDINGS_PROVIDER=sentence-transformers.",
        )
    raise _unsupported_error(name, "the model name has no supported provider prefix.")
