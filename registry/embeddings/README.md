# Embeddings Module

Vendor-agnostic embeddings generation for MCP Gateway Registry's semantic search functionality.

## Overview

This module provides a unified interface for generating text embeddings from multiple providers, supporting both local models (sentence-transformers) and remote APIs: OpenAI-compatible endpoints, Azure OpenAI, and Amazon Bedrock. The remote adapters live in `adapters.py`; LiteLLM is no longer used (issue #1826).

## Features

- **Vendor-agnostic**: Switch between embeddings providers with configuration changes
- **Local & Cloud Support**: Use a local model or a remote API (OpenAI-compatible, Azure OpenAI, Amazon Bedrock)
- **Backward Compatible**: Works seamlessly with existing DocumentDB vector search
- **Easy Configuration**: Simple environment variable setup
- **Extensible**: Easy to add new providers

## Architecture

```
EmbeddingsClient (Abstract Base Class)
├── SentenceTransformersClient (Local models)
└── RemoteEmbeddingsClient (remote APIs; LiteLLMClient is an alias)
      └── adapters.py: embed_openai, embed_azure, embed_bedrock_titan, embed_bedrock_cohere
```

## Quick Start

### Using Sentence Transformers (Default)

```bash
# In .env
EMBEDDINGS_PROVIDER=sentence-transformers
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDINGS_MODEL_DIMENSIONS=384
```

```python
from registry.embeddings import create_embeddings_client

client = create_embeddings_client(
    provider="sentence-transformers",
    model_name="all-MiniLM-L6-v2",
    embedding_dimension=384,
)

embeddings = client.encode(["Hello world", "This is a test"])
print(embeddings.shape)  # (2, 384)
```

### Using OpenAI

```bash
# In .env
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=openai/text-embedding-3-small
EMBEDDINGS_MODEL_DIMENSIONS=1536
EMBEDDINGS_API_KEY=your_openai_api_key
```

```python
from registry.embeddings import create_embeddings_client

client = create_embeddings_client(
    provider="litellm",
    model_name="openai/text-embedding-3-small",
    api_key="your_openai_api_key",
    embedding_dimension=1536,
)

embeddings = client.encode(["Hello world", "This is a test"])
print(embeddings.shape)  # (2, 1536)
```

### Using Amazon Bedrock

Amazon Bedrock uses the standard AWS credential chain for authentication.

```bash
# In .env
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=bedrock/amazon.titan-embed-text-v1
EMBEDDINGS_MODEL_DIMENSIONS=1536
EMBEDDINGS_AWS_REGION=us-east-1
```

**Configure AWS credentials via standard methods:**

**Option 1: IAM Roles (Recommended for EC2/EKS)**
```bash
# No additional configuration needed
# EC2 instance or EKS pod automatically uses attached IAM role
```

**Option 2: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export AWS_REGION=us-east-1
```

**Option 3: AWS Credentials File**
```bash
# ~/.aws/credentials
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# ~/.aws/config
[default]
region = us-east-1
```

**Python Usage:**
```python
from registry.embeddings import create_embeddings_client

# Uses standard AWS credential chain
client = create_embeddings_client(
    provider="litellm",
    model_name="bedrock/amazon.titan-embed-text-v1",
    aws_region="us-east-1",
    embedding_dimension=1536,
)

embeddings = client.encode(["Hello world", "This is a test"])
print(embeddings.shape)  # (2, 1536)
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `EMBEDDINGS_PROVIDER` | Provider type: `sentence-transformers` or `litellm` | `sentence-transformers` | No |
| `EMBEDDINGS_MODEL_NAME` | Model identifier | `all-MiniLM-L6-v2` | Yes |
| `EMBEDDINGS_MODEL_DIMENSIONS` | Embedding dimension | `384` | Yes |
| `EMBEDDINGS_API_KEY` | API key for an OpenAI-compatible endpoint or Azure OpenAI | - | For cloud* |
| `EMBEDDINGS_API_BASE` | Endpoint base URL (OpenAI defaults to api.openai.com; Azure requires it) | - | For Azure and custom endpoints |
| `EMBEDDINGS_AWS_REGION` | AWS region for Amazon Bedrock | `us-east-1` | For Bedrock |
| `EMBEDDINGS_REQUEST_TIMEOUT_SECONDS` | Timeout for each remote request | `30` | No |
| `EMBEDDINGS_MAX_RETRIES` | Retries on 429, 5xx and connection errors | `2` | No |

*Not required for Amazon Bedrock, which uses the standard AWS credential chain (IAM roles, environment variables, ~/.aws/credentials). The full settings table is in [docs/embeddings.md](../../docs/embeddings.md#environment-variables).

### Supported Models

#### Sentence Transformers (Local)

- `all-MiniLM-L6-v2` (384 dimensions) - Fast, lightweight
- `all-mpnet-base-v2` (768 dimensions) - High quality
- `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions) - Multilingual
- Any model from [Hugging Face sentence-transformers](https://huggingface.co/models?library=sentence-transformers)

#### Remote APIs

The model-name prefix picks the adapter: `openai/<model>` (OpenAI or any OpenAI-compatible endpoint), `azure/<deployment>`, `bedrock/amazon.titan-embed-text-*`, or `bedrock/cohere.embed-*`. The full list with dimensions is in [docs/embeddings.md](../../docs/embeddings.md#remote-embeddings-apis). The `cohere/` and `voyage/` prefixes were removed with LiteLLM.

## API Reference

### EmbeddingsClient (Abstract)

Base class for all embeddings clients.

**Methods:**
- `encode(texts: List[str]) -> np.ndarray`: Generate embeddings for texts
- `get_embedding_dimension() -> int`: Get embedding dimension

### SentenceTransformersClient

Local embeddings using sentence-transformers library.

**Constructor:**
```python
SentenceTransformersClient(
    model_name: str,
    model_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
)
```

**Parameters:**
- `model_name`: Hugging Face model identifier
- `model_dir`: Local directory with pre-downloaded model (optional)
- `cache_dir`: Cache directory for models (optional)

### RemoteEmbeddingsClient

Remote embeddings through the adapters in `adapters.py`. `LiteLLMClient` is kept as an alias of this class.

**Constructor:**
```python
RemoteEmbeddingsClient(
    model_name: str,
    api_key: str | None = None,
    api_base: str | None = None,
    aws_region: str | None = None,
    embedding_dimension: int | None = None,
    token_provider: Callable[[], str] | None = None,
    response_format: str = "openai",
    request_timeout_seconds: int = 30,
    max_retries: int = 2,
    extra_headers: dict[str, str] | None = None,
)
```

**Parameters:**
- `model_name`: Provider-prefixed model (for example `openai/text-embedding-3-small`, `bedrock/amazon.titan-embed-text-v2:0`). An unsupported prefix raises `ValueError`.
- `api_key`: Static key for OpenAI-compatible endpoints and Azure. Falls back to `OPENAI_API_KEY` or `AZURE_API_KEY`. Not used for Amazon Bedrock.
- `api_base`: Endpoint base URL. Falls back to `OPENAI_BASE_URL`/`OPENAI_API_BASE`, or `AZURE_API_BASE` for Azure.
- `aws_region`: AWS region for Amazon Bedrock.
- `token_provider`: Callable returning a fresh bearer token per call (`EMBEDDINGS_AUTH_MODE=idp`). Takes precedence over `api_key`.
- `response_format`: `openai` or `raw_array`. With `raw_array` any prefix is accepted and the request goes to `api_base`.
- `request_timeout_seconds`, `max_retries`: Timeout and transient-failure retries for every request.
- `extra_headers`: Operator headers (`EMBEDDINGS_EXTRA_HEADER_<N>`); the registry's own headers always win.
- `embedding_dimension`: Expected dimension for validation (optional).

**Amazon Bedrock Notes:**
- Uses the standard AWS credential chain (IAM roles, environment variables, ~/.aws/credentials).
- Titan takes one text per request, so the adapter makes one `invoke_model` call per text. Cohere takes a batch in one call.

### Factory Function

```python
create_embeddings_client(
    provider: str,
    model_name: str,
    model_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    api_base: Optional[str] = None,
    aws_region: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
) -> EmbeddingsClient
```

Creates an embeddings client based on the provider type.

**Parameters:**
- `provider`: "sentence-transformers" or "litellm" (remote API)
- `model_name`: Model identifier
- `model_dir`: Local model directory (sentence-transformers only)
- `cache_dir`: Cache directory (sentence-transformers only)
- `api_key`: API key (remote only; not used for Amazon Bedrock)
- `api_base`: Endpoint base URL (remote only)
- `aws_region`: AWS region (Amazon Bedrock only)
- `request_timeout_seconds`, `max_retries`: Timeout and retries for remote requests
- `embedding_dimension`: Expected dimension

## Integration with Search Service

The embeddings module integrates seamlessly with the search service, which uses DocumentDB vector search:

```python
# In registry/search/service.py
from registry.embeddings import create_embeddings_client

class SearchService:
    async def _load_embedding_model(self):
        self.embedding_model = create_embeddings_client(
            provider=settings.embeddings_provider,
            model_name=settings.embeddings_model_name,
            # ... other parameters from settings
        )
```

## Migration Guide

### From Direct SentenceTransformer Usage

**Before:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)
```

**After:**
```python
from registry.embeddings import create_embeddings_client

client = create_embeddings_client(
    provider="sentence-transformers",
    model_name="all-MiniLM-L6-v2",
)
embeddings = client.encode(texts)
```

### Switching to Cloud Provider

Just update your `.env` file:

```bash
# From
EMBEDDINGS_PROVIDER=sentence-transformers
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDINGS_MODEL_DIMENSIONS=384

# To
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=openai/text-embedding-3-small
EMBEDDINGS_MODEL_DIMENSIONS=1536
EMBEDDINGS_API_KEY=your_openai_api_key
```

No code changes required!

## Performance Considerations

### Local Models (Sentence Transformers)
- **Pros**: No API costs, privacy, no network latency
- **Cons**: CPU/GPU requirements, model download size
- **Best for**: High-volume usage, sensitive data, offline operation

### Remote APIs
- **Pros**: No local resources, higher quality models, instant availability
- **Cons**: API costs, network dependency, data leaves premises
- **Best for**: Low-volume usage, rapid prototyping, maximum quality

## Troubleshooting

### Unsupported Model Name

```
ValueError: EMBEDDINGS_MODEL_NAME='cohere/embed-english-v3.0' is not supported: ...
```

**Solution:** Use a supported prefix (see [Remote APIs](#remote-apis)). For another provider with an OpenAI-compatible endpoint, use `openai/<model>` with `EMBEDDINGS_API_BASE`.

### Dimension Mismatch

```
WARNING: Embedding dimension mismatch: expected 384, got 1536
```

**Solution:** Update `EMBEDDINGS_MODEL_DIMENSIONS` to match your model's actual output.

### API Authentication Errors

For cloud providers, ensure your API key is correctly set:
- OpenAI: Set `EMBEDDINGS_API_KEY`
- Cohere: Set `EMBEDDINGS_API_KEY`
- Bedrock: Configure AWS credentials via standard AWS methods

## Testing

Run the test suite to verify the integration:

```bash
# Create a test file
cat > test_embeddings.py << 'EOF'
from registry.embeddings import create_embeddings_client

# Test sentence-transformers
client = create_embeddings_client(
    provider="sentence-transformers",
    model_name="all-MiniLM-L6-v2",
)
embeddings = client.encode(["test"])
print(f"✓ Embeddings shape: {embeddings.shape}")
EOF

# Run test
uv run python test_embeddings.py
```

## Contributing

To add a new embeddings provider:

1. Create a new client class inheriting from `EmbeddingsClient`
2. Implement `encode()` and `get_embedding_dimension()` methods
3. Update `create_embeddings_client()` factory function
4. Add configuration options to `registry/core/config.py`
5. Document in this README

## License

Apache 2.0 - See LICENSE file for details
