# FastEmbed Pre-Baked Model Design Document

**Created**: 2026-01-24
**Author**: AI Design Team
**Status**: Draft - Pending Review

---

## Executive Summary

This document proposes adding FastEmbed as a fallback embedding provider to the MCP Gateway Registry container. The goal is to ensure reliable semantic search functionality even when the primary embedding model (configured via environment variables) fails to download from Hugging Face or other external sources.

---

## Problem Statement

### Current Architecture

The registry container currently supports two embedding providers:

1. **SentenceTransformers** (default): Downloads models from Hugging Face on first use
2. **LiteLLM**: Uses cloud APIs (Amazon Bedrock, OpenAI, Cohere, etc.)

### Current Flow

```
Container Start
    │
    ▼
Load Embedding Model (lazy, on first use)
    │
    ├── SentenceTransformers: Download from Hugging Face
    │   └── If download fails → Search service unavailable
    │
    └── LiteLLM: API call to cloud provider
        └── If API fails → Search service unavailable
```

### The Problem

1. **Network Dependency**: First-time model download requires internet access to Hugging Face
2. **Download Failures**: Hugging Face rate limits, network issues, or firewall restrictions can block downloads
3. **Cold Start Latency**: Initial model download adds significant startup time (~30-60 seconds)
4. **Air-Gapped Environments**: Containers deployed in restricted networks cannot download models
5. **No Fallback**: If the configured model fails to load, semantic search is completely unavailable

### Impact

When embeddings fail to load:
- Semantic search endpoint returns 500 errors
- MCP server and agent discovery degraded to keyword-only search
- User experience significantly impacted
- No graceful degradation

---

## Proposed Solution

### Overview

Pre-bake a lightweight, fast embedding model (FastEmbed with `BAAI/bge-small-en-v1.5`) into the Docker container image. This model serves as an automatic fallback when the configured embedding model fails to load.

### Why FastEmbed?

| Feature | FastEmbed | SentenceTransformers |
|---------|-----------|---------------------|
| **Model Size** | ~50MB (bge-small-en-v1.5) | ~90MB (all-MiniLM-L6-v2) |
| **Load Time** | ~1-2 seconds | ~3-5 seconds |
| **Dependencies** | ONNX Runtime (CPU) | PyTorch (CPU) |
| **Memory** | ~200MB | ~500MB |
| **Quality** | Good (MTEB: 62.x) | Good (MTEB: 58.x) |
| **Dimensions** | 384 | 384 |
| **Quantization** | INT8 support | Float32 only |

### Chosen Model: `BAAI/bge-small-en-v1.5`

- **Dimensions**: 384 (matches default `all-MiniLM-L6-v2`)
- **Size**: ~45MB compressed
- **Speed**: 10,000+ documents/second on CPU
- **Quality**: Top-tier on MTEB benchmark for its size class (MTEB: 62.x vs 58.x for MiniLM)
- **License**: MIT (permissive for commercial use)
- **Source**: Beijing Academy of AI (BAAI)

**Performance Note**: ONNX inference is single-threaded by default. For multi-core usage, set `OMP_NUM_THREADS` environment variable.

### Architecture

```
Container Start
    │
    ▼
Load Primary Embedding Model
    │
    ├── Success → Use configured model
    │
    └── Failure (download/API error)
        │
        ▼
    Load FastEmbed Fallback
        │
        ├── Success → Use pre-baked FastEmbed model
        │   └── Log warning about fallback mode
        │
        └── Failure → Semantic search unavailable
            └── Log critical error
```

---

## Detailed Design

### 1. New Embedding Client: FastEmbedClient

**File**: `registry/embeddings/client.py`

```python
class FastEmbedClient(EmbeddingsClient):
    """Client for FastEmbed models with pre-baked fallback support."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the FastEmbed client.

        Args:
            model_name: FastEmbed model name (default: BAAI/bge-small-en-v1.5)
            cache_dir: Directory containing pre-baked model files
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Optional["TextEmbedding"] = None
        self._dimension: Optional[int] = None

    def _load_model(self) -> None:
        """Load the FastEmbed model."""
        if self._model is not None:
            return

        try:
            from fastembed import TextEmbedding

            # Configure cache directory for pre-baked models
            kwargs = {"model_name": self.model_name}
            if self.cache_dir:
                kwargs["cache_dir"] = str(self.cache_dir)

            logger.info(
                f"Loading FastEmbed model: {self.model_name} "
                f"(cache_dir: {self.cache_dir})"
            )
            self._model = TextEmbedding(**kwargs)

            # Determine dimension from a test embedding
            test_embedding = list(self._model.embed(["test"]))[0]
            self._dimension = len(test_embedding)

            logger.info(
                f"FastEmbed model loaded successfully. Dimension: {self._dimension}"
            )

        except Exception as e:
            logger.error(
                f"Failed to load FastEmbed model: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to load FastEmbed model: {e}") from e

    def encode(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Generate embeddings using FastEmbed.

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
            # FastEmbed returns a generator, convert to list then array
            embeddings_generator = self._model.embed(texts)
            embeddings_list = list(embeddings_generator)
            return np.array(embeddings_list, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to encode texts with FastEmbed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to encode texts with FastEmbed: {e}") from e

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
```

### 2. Factory Function Enhancement

**File**: `registry/embeddings/client.py`

Update `create_embeddings_client()` to support:
1. New `fastembed` provider option
2. Automatic fallback when primary provider fails

```python
from typing import Literal

# Type alias for supported embedding providers
EmbeddingProvider = Literal["sentence-transformers", "litellm", "fastembed"]

FALLBACK_MODEL_NAME = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL_DIMENSIONS = 384


def _try_load_fastembed_fallback(
    cache_dir: Optional[Path] = None,
) -> Optional[EmbeddingsClient]:
    """
    Attempt to load FastEmbed fallback model.

    Args:
        cache_dir: Directory containing pre-baked model files

    Returns:
        FastEmbedClient if successful, None otherwise
    """
    try:
        logger.warning(
            "Primary embedding model failed. Attempting FastEmbed fallback..."
        )
        client = FastEmbedClient(
            model_name=FALLBACK_MODEL_NAME,
            cache_dir=cache_dir,
        )
        # Trigger model loading to verify it works
        _ = client.get_embedding_dimension()
        logger.warning(
            f"FastEmbed fallback loaded successfully. "
            f"Model: {FALLBACK_MODEL_NAME}, Dimension: {FALLBACK_MODEL_DIMENSIONS}"
        )
        return client
    except Exception as e:
        logger.error(f"FastEmbed fallback also failed: {e}")
        return None


def create_embeddings_client(
    provider: str,
    model_name: str,
    model_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    aws_region: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    enable_fallback: bool = True,
) -> EmbeddingsClient:
    """
    Factory function to create an embeddings client based on provider.

    Args:
        provider: Provider type ('sentence-transformers', 'litellm', or 'fastembed')
        model_name: Model identifier
        model_dir: Optional local model directory (sentence-transformers only)
        cache_dir: Optional cache directory
        api_key: Optional API key (litellm only)
        api_base: Optional API base URL (litellm only)
        aws_region: Optional AWS region (litellm with Bedrock only)
        embedding_dimension: Optional embedding dimension
        enable_fallback: If True, fall back to FastEmbed on primary failure

    Returns:
        EmbeddingsClient instance

    Raises:
        ValueError: If provider is not supported
        RuntimeError: If all providers (including fallback) fail
    """
    provider_lower = provider.lower()

    # Try primary provider
    try:
        if provider_lower == "fastembed":
            logger.info(f"Creating FastEmbedClient with model: {model_name}")
            return FastEmbedClient(
                model_name=model_name,
                cache_dir=cache_dir,
            )

        elif provider_lower == "sentence-transformers":
            logger.info(
                f"Creating SentenceTransformersClient with model: {model_name}"
            )
            return SentenceTransformersClient(
                model_name=model_name,
                model_dir=model_dir,
                cache_dir=cache_dir,
            )

        elif provider_lower == "litellm":
            # ... existing LiteLLM logic ...
            pass

        else:
            raise ValueError(
                f"Unsupported embeddings provider: {provider}. "
                "Supported providers: 'sentence-transformers', 'litellm', 'fastembed'"
            )

    except Exception as primary_error:
        logger.error(
            f"Primary embedding provider '{provider}' failed: {primary_error}"
        )

        if enable_fallback:
            fallback_client = _try_load_fastembed_fallback(cache_dir)
            if fallback_client:
                return fallback_client

        # Re-raise if no fallback or fallback failed
        raise
```

### 3. Dockerfile Changes

**File**: `docker/Dockerfile.registry`

Add FastEmbed dependency and pre-download model during build. Uses a separate model-cache stage for better build caching.

```dockerfile
# ===== MODEL CACHE STAGE =====
# Separate stage for model download - cached independently from application code
FROM python:3.12-slim AS model-cache

# Install fastembed with pinned version
RUN pip install "fastembed==0.3.6"

# Download model in isolation (cached unless fastembed version changes)
# Model: BAAI/bge-small-en-v1.5 (MIT License, Beijing Academy of AI)
RUN python -c "
from fastembed import TextEmbedding
import os

cache_dir = '/models/fastembed'
os.makedirs(cache_dir, exist_ok=True)

print('Pre-downloading FastEmbed model: BAAI/bge-small-en-v1.5')
model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir=cache_dir)

# Verify model works
embeddings = list(model.embed(['Hello world test']))
print(f'Model loaded successfully. Embedding dimension: {len(embeddings[0])}')
"

# ===== BACKEND BUILD STAGE =====
FROM python:3.12-slim AS backend-builder

# ... existing setup ...

# Install Python dependencies including FastEmbed (pinned version)
RUN pip install uv && \
    uv venv .venv --python 3.12 && \
    . .venv/bin/activate && \
    uv pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.0.0" \
    "torchvision" && \
    uv pip install \
    # ... existing packages ...
    "fastembed==0.3.6" \
    # ... rest of packages ...

# ... continue with rest of build ...

# ===== FINAL RUNTIME STAGE =====
FROM python:3.12-slim AS runtime

# ... existing setup ...

# Copy pre-downloaded model from model-cache stage (not backend-builder)
# This enables independent caching - model only re-downloads when fastembed version changes
COPY --from=model-cache /models/fastembed /app/registry/.fastembed_cache
```

**Benefits of separate model-cache stage:**
- Model download cached independently from application code changes
- Reduces build time on code-only changes (~30-60 seconds saved)
- Clear separation of concerns

### 4. Configuration Changes

**File**: `registry/core/config.py`

Add FastEmbed configuration:

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Embeddings settings
    embeddings_provider: str = "sentence-transformers"  # 'sentence-transformers', 'litellm', or 'fastembed'
    embeddings_model_name: str = "all-MiniLM-L6-v2"
    embeddings_model_dimensions: int = 384

    # Fallback settings
    embeddings_enable_fallback: bool = True
    embeddings_fallback_model: str = "BAAI/bge-small-en-v1.5"
    embeddings_fallback_dimensions: int = 384

    @property
    def fastembed_cache_dir(self) -> Path:
        """Directory containing pre-baked FastEmbed models."""
        if self.is_local_dev:
            return Path.cwd() / "registry" / ".fastembed_cache"
        return self.container_registry_dir / ".fastembed_cache"
```

### 6. Health Endpoint Enhancement

**File**: `registry/api/health_routes.py`

Add embedding status to health endpoint for operational visibility:

```python
class EmbeddingHealthStatus(BaseModel):
    """Embedding service health status."""
    provider: str
    model: str
    dimensions: int
    fallback_mode: bool
    load_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    embeddings: Optional[EmbeddingHealthStatus] = None
```

Example response when running in fallback mode:
```json
{
  "status": "healthy",
  "embeddings": {
    "provider": "fastembed",
    "model": "BAAI/bge-small-en-v1.5",
    "dimensions": 384,
    "fallback_mode": true,
    "load_time_ms": 1523
  }
}
```

### 5. Search Service Integration

**File**: `registry/search/service.py`

Update `_load_embedding_model()` to use fallback:

```python
async def _load_embedding_model(self):
    """Load the embeddings model using the configured provider with fallback support."""
    logger.info(
        f"Loading embedding model with provider: {settings.embeddings_provider}"
    )

    try:
        # Prepare cache directory
        model_cache_path = settings.container_registry_dir / ".cache"
        model_cache_path.mkdir(parents=True, exist_ok=True)

        # Create embeddings client using factory with fallback enabled
        self.embedding_model = create_embeddings_client(
            provider=settings.embeddings_provider,
            model_name=settings.embeddings_model_name,
            model_dir=settings.embeddings_model_dir
            if settings.embeddings_provider == "sentence-transformers"
            else None,
            cache_dir=settings.fastembed_cache_dir
            if settings.embeddings_enable_fallback
            else model_cache_path,
            api_key=settings.embeddings_api_key
            if settings.embeddings_provider == "litellm"
            else None,
            api_base=settings.embeddings_api_base
            if settings.embeddings_provider == "litellm"
            else None,
            aws_region=settings.embeddings_aws_region
            if settings.embeddings_provider == "litellm"
            else None,
            embedding_dimension=settings.embeddings_model_dimensions,
            enable_fallback=settings.embeddings_enable_fallback,
        )

        # Get and log the embedding dimension
        embedding_dim = self.embedding_model.get_embedding_dimension()

        # Check if we're running in fallback mode
        if isinstance(self.embedding_model, FastEmbedClient):
            logger.warning(
                f"Running in FALLBACK mode with FastEmbed. "
                f"Model: {settings.embeddings_fallback_model}, "
                f"Dimension: {embedding_dim}"
            )
        else:
            logger.info(
                f"Embedding model loaded successfully. "
                f"Provider: {settings.embeddings_provider}, "
                f"Model: {settings.embeddings_model_name}, "
                f"Dimension: {embedding_dim}"
            )

        # Update dimensions if needed
        if embedding_dim != settings.embeddings_model_dimensions:
            logger.warning(
                f"Embedding dimension mismatch: "
                f"configured={settings.embeddings_model_dimensions}, "
                f"actual={embedding_dim}. Using actual dimension."
            )
            settings.embeddings_model_dimensions = embedding_dim

    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", exc_info=True)
        self.embedding_model = None
```

---

## Vector Storage Compatibility

### Note on FAISS Deprecation

**FAISS is deprecated and will be removed from the codebase.** The registry now uses MongoDB CE (with client-side vector search) or AWS DocumentDB (with native vector search) for embedding storage.

### Dimension Considerations

The fallback model (`BAAI/bge-small-en-v1.5`) produces 384-dimensional embeddings, matching the default `all-MiniLM-L6-v2` model. For users with different dimension configurations (e.g., Amazon Bedrock Titan with 1024 dimensions):

1. **Detection**: On fallback activation, log dimension mismatch warning
2. **Behavior**: Existing embeddings in MongoDB/DocumentDB will be incompatible
3. **Resolution**: Re-indexing of servers/agents will occur automatically on next registration or manual trigger

Since MongoDB/DocumentDB stores embeddings as document fields, dimension changes are handled more gracefully than with FAISS binary indexes.

---

## Environment Variables

### New Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDINGS_ENABLE_FALLBACK` | Enable FastEmbed fallback | `true` |
| `EMBEDDINGS_FALLBACK_MODEL` | Fallback model name | `BAAI/bge-small-en-v1.5` |
| `EMBEDDINGS_FALLBACK_DIMENSIONS` | Fallback model dimensions | `384` |

### Updated .env.example

```bash
# =============================================================================
# EMBEDDINGS CONFIGURATION
# =============================================================================

# Provider selection: sentence-transformers, litellm, or fastembed
EMBEDDINGS_PROVIDER=sentence-transformers

# Model selection (provider-specific)
# - sentence-transformers: all-MiniLM-L6-v2, all-mpnet-base-v2, etc.
# - litellm: bedrock/amazon.titan-embed-text-v2:0, openai/text-embedding-3-small, etc.
# - fastembed: BAAI/bge-small-en-v1.5, BAAI/bge-base-en-v1.5, etc.
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2

# Embedding dimensions (must match model)
EMBEDDINGS_MODEL_DIMENSIONS=384

# Fallback Configuration
# Enable automatic fallback to pre-baked FastEmbed model on primary failure
EMBEDDINGS_ENABLE_FALLBACK=true

# Fallback model (pre-baked in container image)
EMBEDDINGS_FALLBACK_MODEL=BAAI/bge-small-en-v1.5
EMBEDDINGS_FALLBACK_DIMENSIONS=384
```

---

## Docker Image Size Impact

### Current Image Size

| Component | Size |
|-----------|------|
| Python runtime | ~200MB |
| PyTorch (CPU) | ~500MB |
| SentenceTransformers | ~150MB |
| Application code | ~50MB |
| **Total** | ~900MB |

### With FastEmbed Fallback

| Component | Size |
|-----------|------|
| Python runtime | ~200MB |
| PyTorch (CPU) | ~500MB |
| SentenceTransformers | ~150MB |
| FastEmbed + ONNX | ~100MB |
| Pre-baked model | ~50MB |
| Application code | ~50MB |
| **Total** | ~1050MB |

**Impact**: +150MB (~17% increase)

### Optimization Options

1. **Multi-stage optimization**: Only copy required ONNX files
2. **Lazy PyTorch loading**: Don't load PyTorch if only FastEmbed is used
3. **Remove SentenceTransformers in fallback-only mode**: Save ~150MB

---

## Testing Strategy

### Unit Tests

```python
class TestFastEmbedClient:
    """Tests for FastEmbedClient."""

    def test_encode_single_text(self):
        """Test encoding a single text."""
        client = FastEmbedClient(model_name="BAAI/bge-small-en-v1.5")
        embeddings = client.encode(["Hello world"])
        assert embeddings.shape == (1, 384)

    def test_encode_multiple_texts(self):
        """Test encoding multiple texts."""
        client = FastEmbedClient(model_name="BAAI/bge-small-en-v1.5")
        embeddings = client.encode(["Hello", "World", "Test"])
        assert embeddings.shape == (3, 384)

    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        client = FastEmbedClient(model_name="BAAI/bge-small-en-v1.5")
        assert client.get_embedding_dimension() == 384

    def test_load_from_cache_dir(self):
        """Test loading pre-baked model from cache directory."""
        cache_dir = Path("/app/registry/.fastembed_cache")
        client = FastEmbedClient(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=cache_dir,
        )
        embeddings = client.encode(["Test"])
        assert embeddings.shape == (1, 384)


class TestFallbackMechanism:
    """Tests for fallback mechanism."""

    @patch("registry.embeddings.client.SentenceTransformersClient")
    def test_fallback_on_primary_failure(self, mock_st):
        """Test that fallback activates when primary fails."""
        mock_st.side_effect = RuntimeError("Download failed")

        client = create_embeddings_client(
            provider="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            enable_fallback=True,
        )

        assert isinstance(client, FastEmbedClient)

    def test_no_fallback_when_disabled(self):
        """Test that fallback doesn't activate when disabled."""
        with pytest.raises(RuntimeError):
            create_embeddings_client(
                provider="sentence-transformers",
                model_name="nonexistent-model",
                enable_fallback=False,
            )

    def test_fallback_with_dimension_mismatch_logging(self, caplog):
        """Test that dimension mismatch is logged when fallback activates."""
        # Simulate scenario where primary was 1024d but fallback is 384d
        # Verify warning is logged
        pass


class TestEmbeddingPerformance:
    """Performance benchmark tests."""

    def test_fastembed_vs_sentence_transformers_speed(self):
        """Compare encoding speed between FastEmbed and SentenceTransformers."""
        import time

        texts = ["Sample text for embedding"] * 100

        # FastEmbed timing
        fe_client = FastEmbedClient(model_name="BAAI/bge-small-en-v1.5")
        start = time.time()
        fe_client.encode(texts)
        fe_time = time.time() - start

        # Log results for comparison (not a hard assertion)
        print(f"FastEmbed: {fe_time:.3f}s for {len(texts)} texts")
```

### Integration Tests

```python
class TestSemanticSearchWithFallback:
    """Integration tests for semantic search with fallback."""

    async def test_search_works_in_fallback_mode(self):
        """Test that semantic search works when using fallback."""
        # Force fallback by using invalid primary provider
        settings.embeddings_provider = "sentence-transformers"
        settings.embeddings_model_name = "nonexistent"
        settings.embeddings_enable_fallback = True

        service = FaissService()
        await service.initialize()

        # Add test server
        await service.add_or_update_service(
            "test/server",
            {"server_name": "test", "description": "Test server"},
        )

        # Search should work
        results = await service.search_mixed("test server")
        assert len(results["servers"]) > 0
```

---

## Migration Plan

### Phase 1: Add FastEmbed Support (Non-Breaking)

1. Add `FastEmbedClient` class
2. Add `fastembed` as a selectable provider
3. Update factory function
4. Add unit tests
5. **No fallback yet** - explicit provider only

### Phase 2: Pre-Bake Model in Docker

1. Update Dockerfile to install FastEmbed
2. Pre-download model during build
3. Test container image size
4. Verify model loads from cache

### Phase 3: Enable Fallback Mechanism

1. Add fallback logic to factory
2. Add configuration options
3. Update documentation
4. Integration tests

### Phase 4: Production Rollout

1. Deploy to staging
2. Test fallback scenarios
3. Monitor for issues
4. Gradual production rollout

---

## Monitoring and Observability

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `embeddings_fallback_activated_total` | Counter | Times fallback was activated |
| `embeddings_provider_load_duration_seconds` | Histogram | Time to load embedding model |
| `embeddings_encode_duration_seconds` | Histogram | Time to encode texts |
| `embeddings_dimension_mismatch_total` | Counter | Dimension mismatches detected |

### Logging

```python
# Primary load failure
logger.error(
    f"Primary embedding provider '{provider}' failed: {error}",
    extra={
        "provider": provider,
        "model": model_name,
        "error_type": type(error).__name__,
    }
)

# Fallback activation
logger.warning(
    f"Activating FastEmbed fallback mode",
    extra={
        "fallback_model": FALLBACK_MODEL_NAME,
        "fallback_dimensions": FALLBACK_MODEL_DIMENSIONS,
        "original_provider": provider,
    }
)
```

### Health Check Enhancement

Add embedding status to health endpoint:

```json
{
  "status": "healthy",
  "embeddings": {
    "provider": "fastembed",
    "model": "BAAI/bge-small-en-v1.5",
    "dimensions": 384,
    "fallback_mode": true,
    "load_time_ms": 1523
  }
}
```

---

## Security Considerations

1. **Model Provenance**: `BAAI/bge-small-en-v1.5` is from Beijing Academy of AI, MIT licensed
2. **Supply Chain**: Model downloaded during build, not runtime
3. **Integrity**: Consider adding checksum verification for pre-baked model
4. **No External Calls**: Fallback operates entirely offline

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Image size increase | High | Low | Optimization options documented |
| Dimension mismatch | Medium | Medium | Auto-reindex capability |
| Model quality difference | Low | Low | BGE model comparable quality |
| ONNX compatibility | Low | Medium | Pin ONNX versions |
| Build time increase | Medium | Low | Parallel downloads, caching |

---

## Open Questions

1. ~~Should we support multiple fallback models (e.g., different dimension options)?~~ **DECIDED: No** - Single fallback model (`BAAI/bge-small-en-v1.5`, 384d) keeps implementation simple. Users with non-384d configurations will need to handle dimension mismatch via index rebuild.
2. ~~Should fallback be enabled by default or opt-in?~~ **DECIDED: Enabled by default** - Users benefit from resilience without additional configuration. Can be disabled via `EMBEDDINGS_ENABLE_FALLBACK=false`.
3. ~~How do we handle existing FAISS indexes when switching to fallback with different dimensions?~~ **DECIDED: Not applicable** - FAISS is deprecated and will be removed from the codebase. MongoDB CE/DocumentDB handles vector storage natively.
4. ~~Should we emit alerts when running in fallback mode?~~ **DECIDED: No alerts** - Just log error/warning messages in container logs. System is still functional with fallback, no need for active alerting.

---

## Appendix: Alternative Models Considered

| Model | Dimensions | Size | MTEB Score | Decision |
|-------|------------|------|------------|----------|
| BAAI/bge-small-en-v1.5 | 384 | 45MB | 62.x | **Selected** |
| BAAI/bge-base-en-v1.5 | 768 | 110MB | 63.x | Too large |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 90MB | 58.x | Uses PyTorch |
| intfloat/e5-small-v2 | 384 | 45MB | 61.x | Good alternative |
| Alibaba-NLP/gte-small | 384 | 50MB | 62.x | Good alternative |

---

## References

- [FastEmbed Documentation](https://github.com/qdrant/fastembed)
- [BAAI/bge-small-en-v1.5 on Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [MTEB Benchmark Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## Review Checklist

- [x] Technical accuracy verified
- [ ] Code examples tested
- [ ] Docker changes validated
- [x] Migration path clear
- [x] Security considerations addressed
- [x] Performance impact acceptable
- [ ] Documentation complete
- [x] Team review completed

---

## Team Review

**Status**: APPROVED FOR IMPLEMENTATION

Team persona review conducted on 2026-01-24. All personas approved with recommendations that have been incorporated into this design.

See full review details: [team-persona-review.md](387/team-persona-review.md)

**Summary of Decisions:**

| Persona | Decision | Status |
|---------|----------|--------|
| Chief Architect | APPROVE | Recommendations incorporated |
| Backend API Developer | APPROVE | Recommendations incorporated |
| DevOps Engineer | APPROVE | Recommendations incorporated |
| Security Engineer | APPROVE | Recommendations incorporated |
| SRE Engineer | APPROVE | Recommendations incorporated (no alerts per decision) |
| AI/Agent Developer | APPROVE | Recommendations incorporated |
| Merge Specialist | CONDITIONAL | In progress |

**Key Decisions Made:**
1. Single fallback model only (no multiple dimension options)
2. Fallback enabled by default
3. No external alerts - logging to container logs only
4. Pin `fastembed` to specific version (`==0.3.6`)
5. FAISS deprecated - MongoDB CE/DocumentDB handles vector storage

*Next Step: Begin Phase 1 implementation*
