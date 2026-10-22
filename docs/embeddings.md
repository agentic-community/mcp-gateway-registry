# Embeddings Configuration

Flexible, vendor-agnostic embeddings generation for MCP Gateway Registry's semantic search functionality.

## Overview

The MCP Gateway Registry provides semantic search capabilities across MCP servers, tools, and AI agents. Embeddings come from one of two providers:

1. **Sentence Transformers** (default, `EMBEDDINGS_PROVIDER=sentence-transformers`): a local model, no network calls.
2. **A remote embeddings API** (`EMBEDDINGS_PROVIDER=litellm`): any OpenAI-compatible endpoint, Azure OpenAI, or Amazon Bedrock (Titan and Cohere embed models).

The remote provider keeps the setting value `litellm` for backward compatibility. The registry now calls these APIs with its own small adapters (`registry/embeddings/adapters.py`) instead of the LiteLLM library, and LiteLLM is no longer installed (issue #1826).

Switch between providers with simple configuration changes - no code modifications required.

## Features

- **Vendor-agnostic**: Switch between embeddings providers with configuration changes
- **Local & Cloud Support**: Use a local model or a cloud API (OpenAI-compatible endpoints, Azure OpenAI, Amazon Bedrock)
- **Backward Compatible**: Works seamlessly with existing DocumentDB vector search
- **Easy Configuration**: Simple environment variable setup
- **Extensible**: Easy to add new providers
- **AWS Deployable**: Terraform support for AWS deployments

## Quick Start

### Option 1: Sentence Transformers (Default)

Local embedding models that run on your infrastructure.

```bash
# In .env
EMBEDDINGS_PROVIDER=sentence-transformers
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDINGS_MODEL_DIMENSIONS=384
```

**Characteristics:**
- Runs locally on your infrastructure
- No API costs
- No external network calls required
- Requires CPU/GPU resources
- Model files stored locally
- Data stays within your infrastructure

### Option 2: OpenAI

Cloud-based embedding service via OpenAI API.

```bash
# In .env
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=openai/text-embedding-ada-002
EMBEDDINGS_MODEL_DIMENSIONS=1536
EMBEDDINGS_API_KEY=sk-your-openai-api-key
```

**Characteristics:**
- Cloud-based service
- Requires API key
- API costs per 1K tokens
- No local compute resources needed
- Network dependency
- Data sent to OpenAI

### Option 3: Amazon Bedrock Titan

Cloud-based embedding service via Amazon Bedrock.

```bash
# In .env
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=bedrock/amazon.titan-embed-text-v1
EMBEDDINGS_MODEL_DIMENSIONS=1536
EMBEDDINGS_AWS_REGION=us-east-1
# No API key needed - uses IAM
```

**Characteristics:**
- Cloud-based service
- Uses IAM authentication (no API key required)
- Integrates with AWS security model
- API costs apply
- Requires AWS credentials
- Available in select AWS regions

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `EMBEDDINGS_PROVIDER` | Provider type: `sentence-transformers` or `litellm` | `sentence-transformers` | No |
| `EMBEDDINGS_MODEL_NAME` | Model identifier | `all-MiniLM-L6-v2` | Yes |
| `EMBEDDINGS_MODEL_DIMENSIONS` | Embedding dimension | `384` | Yes |
| `EMBEDDINGS_API_KEY` | API key for an OpenAI-compatible endpoint or Azure OpenAI. Falls back to `OPENAI_API_KEY` or `AZURE_API_KEY` when unset | - | For cloud* |
| `EMBEDDINGS_API_BASE` | Base URL of the endpoint. OpenAI defaults to `https://api.openai.com/v1`; Azure requires it (or `AZURE_API_BASE`) | - | For Azure and custom endpoints |
| `EMBEDDINGS_AWS_REGION` | AWS region for Amazon Bedrock | `us-east-1` | For Bedrock |
| `EMBEDDINGS_RESPONSE_FORMAT` | `openai` (standard envelope) or `raw_array` (bare array of vectors) | `openai` | No |
| `EMBEDDINGS_REQUEST_TIMEOUT_SECONDS` | Timeout for each remote request | `30` | No |
| `EMBEDDINGS_MAX_RETRIES` | Retries on HTTP 429, 5xx and connection errors, with exponential backoff. On Amazon Bedrock this maps to botocore's standard retry mode | `2` | No |
| `EMBEDDINGS_RETRY_MAX_INPUT_CHARS` | Trim target for the one retry after a failed embedding. See [Long entries](#long-entries) | per model | No |
| `EMBEDDINGS_EXTRA_HEADER_1` to `_5` | Extra request headers as `Name: value` (see the [IdP embeddings FAQ](faq/idp-authenticated-embeddings-endpoint.md#extra-request-headers)) | - | No |

*Not required for Amazon Bedrock, which uses the standard AWS credential chain (IAM roles, environment variables, ~/.aws/credentials).

Azure OpenAI uses `AZURE_API_VERSION` when set and `2025-02-01-preview` otherwise.

The registry checks the model name at startup. An unsupported name stops the registry with an error that names the fix, rather than leaving search to fall back to keywords on the first query.

### Terraform Configuration

For AWS ECS deployments, configure embeddings in your `terraform.tfvars`:

#### Using Sentence Transformers (Default)

```hcl
# Local embeddings - no additional configuration needed
# Uses defaults: sentence-transformers with all-MiniLM-L6-v2
```

#### Using OpenAI

```hcl
embeddings_provider         = "litellm"
embeddings_model_name       = "openai/text-embedding-ada-002"
embeddings_model_dimensions = 1536
embeddings_api_key          = "sk-proj-YOUR-OPENAI-API-KEY"
```

#### Using Amazon Bedrock

```hcl
embeddings_provider         = "litellm"
embeddings_model_name       = "bedrock/amazon.titan-embed-text-v1"
embeddings_model_dimensions = 1536
embeddings_aws_region       = "us-east-1"
embeddings_api_key          = ""  # Empty for Bedrock (uses IAM)
```

See [terraform/aws-ecs/terraform.tfvars.example](../terraform/aws-ecs/terraform.tfvars.example) for complete examples.

## Supported Models

### Sentence Transformers (Local)

| Model | Dimensions | Description |
|-------|------------|-------------|
| `all-MiniLM-L6-v2` | 384 | Fast, lightweight (default) |
| `all-mpnet-base-v2` | 768 | High quality |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual |

Any model from [Hugging Face sentence-transformers](https://huggingface.co/models?library=sentence-transformers) is supported.

### Remote embeddings APIs

The model-name prefix picks the API:

| Prefix | API | Example | Dimensions |
|--------|-----|---------|-----------:|
| `openai/` | OpenAI, or any OpenAI-compatible `/embeddings` endpoint via `EMBEDDINGS_API_BASE` | `openai/text-embedding-3-small` | 1536 |
| `openai/` | | `openai/text-embedding-3-large` | 3072 |
| `openai/` | | `openai/text-embedding-ada-002` | 1536 |
| `azure/` | Azure OpenAI; the name after the prefix is the deployment name | `azure/my-embeddings-deployment` | model dependent |
| `bedrock/` | Amazon Bedrock Titan text embeddings | `bedrock/amazon.titan-embed-text-v1` | 1536 |
| `bedrock/` | | `bedrock/amazon.titan-embed-text-v2:0` | 1024 |
| `bedrock/` | Cohere on Amazon Bedrock | `bedrock/cohere.embed-english-v3` | 1024 |
| `bedrock/` | | `bedrock/cohere.embed-multilingual-v3` | 1024 |

Self-hosted servers (vLLM, Ollama, Text Embeddings Inference) and most hosted APIs (Mistral, Together, a corporate AI gateway) expose an OpenAI-compatible endpoint. Use `openai/<model>` with `EMBEDDINGS_API_BASE` set to its base URL.

**Removed with LiteLLM (issue #1826):** `cohere/` (Cohere's own API; use Cohere on Amazon Bedrock), `voyage/`, and any other LiteLLM-only prefix. A registry configured with one refuses to start and names the replacement.

## Long entries

Some entries produce more text than a model accepts: a server with dozens of long tool descriptions can exceed OpenAI's and Titan's 8,192-token limit, and Cohere on Amazon Bedrock rejects any text over 2,048 characters. The provider then rejects the request and the entry gets no vector.

The registry always sends the full text first, so every entry that fits is unchanged. If the call fails and the text is longer than the trim target, it retries once with the text trimmed: every tool or skill name is kept, and the longest descriptions are shortened first, at word boundaries. A failure on shorter text is not retried, because length cannot be the cause. The retry logs one WARNING with the original and trimmed length.

The trim target is `EMBEDDINGS_RETRY_MAX_INPUT_CHARS`. Unset, it is 2,048 characters for Cohere on Amazon Bedrock and 28,000 for other models (about 6,100 to 6,700 tokens of English text). Lower it for a model with a smaller input limit, or for text that uses more tokens per character, such as code or non-English text.

An entry that needs trimming costs one rejected request and one trimmed request each time it is indexed: when it is registered or updated, when a reindex runs, and, for the built-in AI Registry tools server, at each startup. Startup does not re-index the rest of the catalog. The stored content hash is that of the full text, the same as before.

## Migration Between Providers

### Switching Providers

When you switch embedding providers or models with different dimensions, the registry automatically:

1. Detects dimension mismatch
2. Rebuilds the search index
3. Regenerates embeddings for all registered items

Example logs when switching from sentence-transformers (384) to OpenAI (1536):

```
WARNING: Embedding dimension mismatch detected
  Expected: 384 (from existing index)
  Got: 1536 (from current model)
Rebuilding search index with new dimensions...
Regenerating embeddings for all items...
Index rebuild complete
```

### No Code Changes Required

Just update your environment variables or Terraform configuration:

```bash
# From
EMBEDDINGS_PROVIDER=sentence-transformers
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDINGS_MODEL_DIMENSIONS=384

# To
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=openai/text-embedding-ada-002
EMBEDDINGS_MODEL_DIMENSIONS=1536
EMBEDDINGS_API_KEY=sk-your-key
```

Restart the service and the index will be automatically rebuilt.

## Amazon Bedrock Setup

### IAM Permissions

For Amazon Bedrock embeddings, ensure your ECS task role has the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1",
        "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0",
        "arn:aws:bedrock:*::foundation-model/cohere.embed-english-v3"
      ]
    }
  ]
}
```

### Authentication Methods

**IAM Roles (Recommended for ECS/EC2/EKS)**
```bash
# No additional configuration needed
# ECS task, EC2 instance, or EKS pod automatically uses attached IAM role
```

## Architecture

### Embeddings Module Design

```
EmbeddingsClient (Abstract Base Class)
├── SentenceTransformersClient (Local models)
└── RemoteEmbeddingsClient (remote APIs; LiteLLMClient is an alias)
      └── adapters.py: embed_openai, embed_azure, embed_bedrock_titan, embed_bedrock_cohere
```

### Integration with Search Service

The embeddings module integrates seamlessly with the search service, which uses DocumentDB vector search:

```python
# In registry/search/service.py
from registry.embeddings import create_embeddings_client

class SearchService:
    async def _load_embedding_model(self):
        self.embedding_model = create_embeddings_client(
            provider=settings.embeddings_provider,
            model_name=settings.embeddings_model_name,
            api_key=settings.embeddings_api_key,
            aws_region=settings.embeddings_aws_region,
            embedding_dimension=settings.embeddings_model_dimensions,
        )
```

## Performance Considerations

### Local Models (Sentence Transformers)
- Runs on your infrastructure (CPU/GPU)
- No external API calls
- No per-request costs
- Model files stored locally
- Network-independent operation

### Remote APIs
- Runs on provider infrastructure
- Requires network connectivity
- API costs apply (varies by provider)
- No local compute requirements
- Data transmitted to provider

## Graceful Degradation

### Lexical Fallback When Model Unavailable

If the embedding model fails to load or is unreachable (e.g., invalid model name, expired API key, network failure), the search system automatically falls back to **lexical-only search** instead of returning errors.

**What happens:**

1. The embeddings client caches the load error (`_load_error`) to avoid repeated download/API attempts
2. The search repository detects the failure and sets `_embedding_unavailable = True`
3. All subsequent searches use keyword matching (regex on path, name, description, tags, tools) instead of vector similarity
4. Servers and agents are still indexed, but without embeddings (stored with empty vectors)
5. The API response includes `"search_mode": "lexical-only"` to indicate reduced search quality

**How to detect:**

- Check the API response `search_mode` field: `"hybrid"` (normal) vs. `"lexical-only"` (fallback)
- Look for log warnings: `"Embedding model unavailable, falling back to lexical-only search"`
- During indexing: `"Embedding model unavailable, indexing '<name>' without embeddings"`

**How to recover:**

Fix the embedding configuration and restart the service. On restart, the error cache is cleared and the system will attempt to load the model again. If successful, search returns to full hybrid mode automatically.

See [Hybrid Search Architecture](design/hybrid-search-architecture.md) for details on lexical-only scoring.

## Troubleshooting

### Embedding Model Not Found

```
Failed to load SentenceTransformer model: sentence-transformers/my-model is not a local folder
and is not a valid model identifier listed on 'https://huggingface.co/models'
```

**Solution:** Verify the model name in `EMBEDDINGS_MODEL_NAME` is correct. Check the [Hugging Face model hub](https://huggingface.co/models?library=sentence-transformers) for valid names. The system will continue operating with lexical-only search until the model is fixed.

### Unsupported Model Name

```
ValueError: EMBEDDINGS_MODEL_NAME='cohere/embed-english-v3.0' is not supported: the 'cohere/' provider was removed with LiteLLM; use bedrock/cohere.embed-english-v3 (Cohere on Amazon Bedrock).
```

**Solution:** Use one of the prefixes in [Remote embeddings APIs](#remote-embeddings-apis). For any other provider with an OpenAI-compatible endpoint, set `EMBEDDINGS_MODEL_NAME=openai/<model>` and `EMBEDDINGS_API_BASE` to its base URL.

### Dimension Mismatch

```
WARNING: Embedding dimension mismatch: expected 384, got 1536
```

**Solution:** Update `EMBEDDINGS_MODEL_DIMENSIONS` to match your model's actual output dimension. The system will automatically rebuild the index.

### API Authentication Errors

**OpenAI:**
```bash
# Verify API key is set correctly
echo $EMBEDDINGS_API_KEY
# Should start with sk-
```

**Bedrock:**
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check Bedrock access
aws bedrock list-foundation-models --region us-east-1
```

### Missing IAM Permissions

If using AWS ECS and Bedrock, ensure the task execution role has access to the embeddings API key secret:

```bash
# Check IAM policy in terraform/aws-ecs/modules/mcp-gateway/iam.tf
# Should include: aws_secretsmanager_secret.embeddings_api_key.arn
```

## API Reference

### Factory Function

```python
from registry.embeddings import create_embeddings_client

client = create_embeddings_client(
    provider: str,                    # "sentence-transformers" or "litellm" (remote API)
    model_name: str,                  # Model identifier, prefixed for remote APIs
    api_key: str | None = None,       # API key (remote only)
    api_base: str | None = None,      # Endpoint base URL (remote only)
    aws_region: str | None = None,    # AWS region (Amazon Bedrock only)
    embedding_dimension: int | None = None,
    request_timeout_seconds: int = 30,
    max_retries: int = 2,
)
```

### Client Methods

**Generate Embeddings:**
```python
embeddings = client.encode(["text1", "text2"])
# Returns: numpy array of shape (n_texts, embedding_dim)
```

**Get Dimension:**
```python
dim = client.get_embedding_dimension()
# Returns: int (e.g., 384, 1536)
```

## Best Practices

1. Choose the provider that matches your deployment requirements
2. Consider IAM authentication if deploying on AWS
3. Monitor costs when using cloud APIs - implement caching if needed
4. Keep dimension consistent - changing models requires index rebuild
5. Test search results after switching providers to ensure they meet your requirements

## Further Reading

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Amazon Bedrock Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/embeddings.html)
- [Sentence Transformers Models](https://www.sbert.net/docs/pretrained_models.html)
- Search Service Implementation

## Contributing

To add a new embeddings provider:

1. Create a new client class inheriting from `EmbeddingsClient`
2. Implement `encode()` and `get_embedding_dimension()` methods
3. Update `create_embeddings_client()` factory function
4. Add configuration options to `registry/core/config.py`
5. Update this documentation

## License

Apache 2.0 - See [LICENSE](../LICENSE) file for details
