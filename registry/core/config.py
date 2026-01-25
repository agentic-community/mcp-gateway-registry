import os
import secrets
from pathlib import Path
from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"  # Ignore extra environment variables
    )
    
    # Auth settings
    secret_key: str = ""
    admin_user: str = "admin"
    admin_password: str = "password"
    session_cookie_name: str = "mcp_gateway_session"
    session_max_age_seconds: int = 60 * 60 * 8  # 8 hours
    session_cookie_secure: bool = False  # Set to True in production with HTTPS
    session_cookie_domain: Optional[str] = None  # e.g., ".example.com" for cross-subdomain sharing
    auth_server_url: str = "http://localhost:8888"
    auth_server_external_url: str = "http://localhost:8888"  # External URL for OAuth redirects
    
    # Embeddings settings [Default]
    # Provider options: 'sentence-transformers', 'litellm', or 'fastembed'
    embeddings_provider: str = "sentence-transformers"
    embeddings_model_name: str = "all-MiniLM-L6-v2"
    embeddings_model_dimensions: int = 384  # 384 for default and 1024 for bedrock titan v2

    # Fallback settings (FastEmbed pre-baked model)
    embeddings_enable_fallback: bool = True
    embeddings_fallback_model: str = "BAAI/bge-small-en-v1.5"
    embeddings_fallback_dimensions: int = 384

    # LiteLLM-specific settings (only used when embeddings_provider='litellm')
    # For Bedrock: Set to None and configure AWS credentials via standard methods
    # (IAM roles, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env vars, or ~/.aws/credentials)
    embeddings_api_key: Optional[str] = None
    embeddings_secret_key: Optional[str] = None
    embeddings_api_base: Optional[str] = None
    embeddings_aws_region: Optional[str] = "us-east-1"
    
    # Health check settings
    health_check_interval_seconds: int = 300  # 5 minutes for automatic background checks (configurable via env var)
    health_check_timeout_seconds: int = 2  # Very fast timeout for user-driven actions
    
    # WebSocket performance settings
    max_websocket_connections: int = 100  # Reasonable limit for development/testing
    websocket_send_timeout_seconds: float = 2.0  # Allow slightly more time per connection
    websocket_broadcast_interval_ms: int = 10  # Very responsive - 10ms minimum between broadcasts
    websocket_max_batch_size: int = 20  # Smaller batches for faster updates
    websocket_cache_ttl_seconds: int = 1  # 1 second cache for near real-time user feedback

    # Well-known discovery settings
    enable_wellknown_discovery: bool = True
    wellknown_cache_ttl: int = 300  # 5 minutes

    # Security scanning settings (MCP Servers)
    security_scan_enabled: bool = True
    security_scan_on_registration: bool = True
    security_block_unsafe_servers: bool = True
    security_analyzers: str = "yara"  # Comma-separated: yara, llm, or yara,llm
    security_scan_timeout: int = 60  # 1 minutes
    security_add_pending_tag: bool = True
    mcp_scanner_llm_api_key: str = ""  # Optional LLM API key for advanced analysis

    # Agent security scanning settings (A2A Agents)
    agent_security_scan_enabled: bool = True
    agent_security_scan_on_registration: bool = True
    agent_security_block_unsafe_agents: bool = True
    agent_security_analyzers: str = "yara,spec"  # Comma-separated: yara, spec, heuristic, llm, endpoint
    agent_security_scan_timeout: int = 60  # 1 minute
    agent_security_add_pending_tag: bool = True
    a2a_scanner_llm_api_key: str = ""  # Optional Azure OpenAI API key for LLM-based analysis
    
    # Storage Backend Configuration
    storage_backend: str = "file"  # Options: "file", "documentdb"

    # DocumentDB Configuration (only used when storage_backend="documentdb")
    documentdb_host: str = "localhost"
    documentdb_port: int = 27017
    documentdb_database: str = "mcp_registry"
    documentdb_username: Optional[str] = None
    documentdb_password: Optional[str] = None
    documentdb_use_tls: bool = True
    documentdb_tls_ca_file: str = "global-bundle.pem"
    documentdb_use_iam: bool = False
    documentdb_replica_set: Optional[str] = None
    documentdb_read_preference: str = "secondaryPreferred"
    documentdb_direct_connection: bool = False  # Set to True only for single-node MongoDB (tests)

    # DocumentDB Namespace (for multi-tenancy support)
    documentdb_namespace: str = "default"

    # Container paths - adjust for local development
    container_app_dir: Path = Path("/app")
    container_registry_dir: Path = Path("/app/registry")
    container_log_dir: Path = Path("/app/logs")
    
    # Local development mode detection
    @property
    def is_local_dev(self) -> bool:
        """Check if running in local development mode."""
        return not Path("/app").exists()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Generate secret key if not provided
        if not self.secret_key:
            self.secret_key = secrets.token_hex(32)

    @property
    def embeddings_model_dir(self) -> Path:
        if self.is_local_dev:
            return Path.cwd() / "registry" / "models" / self.embeddings_model_name
        return self.container_registry_dir / "models" / self.embeddings_model_name

    @property
    def fastembed_cache_dir(self) -> Path:
        """Directory containing pre-baked FastEmbed models."""
        if self.is_local_dev:
            return Path.cwd() / "registry" / ".fastembed_cache"
        return self.container_registry_dir / ".fastembed_cache"

    @property
    def servers_dir(self) -> Path:
        if self.is_local_dev:
            return Path.cwd() / "registry" / "servers"
        return self.container_registry_dir / "servers"

    @property
    def static_dir(self) -> Path:
        if self.is_local_dev:
            return Path.cwd() / "registry" / "static"
        return self.container_registry_dir / "static"

    @property
    def templates_dir(self) -> Path:
        if self.is_local_dev:
            return Path.cwd() / "registry" / "templates"
        return self.container_registry_dir / "templates"

    @property
    def nginx_config_path(self) -> Path:
        return Path("/etc/nginx/conf.d/nginx_rev_proxy.conf")

    @property
    def state_file_path(self) -> Path:
        return self.servers_dir / "server_state.json"

    @property
    def log_dir(self) -> Path:
        """Get log directory based on environment."""
        if self.is_local_dev:
            return Path.cwd() / "logs"
        return self.container_log_dir

    @property
    def log_file_path(self) -> Path:
        if self.is_local_dev:
            return Path.cwd() / "logs" / "registry.log"
        return self.container_log_dir / "registry.log"

    @property
    def faiss_index_path(self) -> Path:
        return self.servers_dir / "service_index.faiss"

    @property
    def faiss_metadata_path(self) -> Path:
        return self.servers_dir / "service_index_metadata.json"

    @property
    def dotenv_path(self) -> Path:
        if self.is_local_dev:
            return Path.cwd() / ".env"
        return self.container_registry_dir / ".env"

    @property
    def agents_dir(self) -> Path:
        """Directory for agent card storage."""
        if self.is_local_dev:
            return Path.cwd() / "registry" / "agents"
        return self.container_registry_dir / "agents"

    @property
    def agent_state_file_path(self) -> Path:
        """Path to agent state file (enabled/disabled tracking)."""
        return self.agents_dir / "agent_state.json"


class EmbeddingConfig:
    """Helper class for embedding configuration and metadata generation."""

    def __init__(
        self,
        settings_instance: Settings
    ):
        self.settings = settings_instance


    @property
    def model_family(self) -> str:
        """Extract model family from model name.

        Examples:
            - "openai/text-embedding-ada-002" -> "openai"
            - "all-MiniLM-L6-v2" -> "sentence-transformers"
            - "amazon.titan-embed-text-v2:0" -> "amazon-bedrock"
        """
        model_name = self.settings.embeddings_model_name

        if "/" in model_name:
            # Format: "provider/model-name"
            return model_name.split("/")[0]
        elif "amazon." in model_name or "titan" in model_name.lower():
            return "amazon-bedrock"
        elif self.settings.embeddings_provider == "litellm":
            return "litellm"
        else:
            return self.settings.embeddings_provider


    @property
    def index_name(self) -> str:
        """Generate dimension-specific collection/index name.

        Returns index name in format: mcp-embeddings-{dimensions}-{namespace}
        Example: mcp-embeddings-1536-default
        """
        base_name = "mcp-embeddings"
        dimensions = self.settings.embeddings_model_dimensions
        namespace = self.settings.documentdb_namespace

        # Replace base name with dimension-specific name
        return f"{base_name}-{dimensions}-{namespace}"


    def get_embedding_metadata(self) -> dict:
        """Generate embedding metadata for document storage.

        Returns:
            Dictionary with embedding metadata including:
            - provider: Embedding provider (e.g., "litellm", "sentence-transformers")
            - model: Full model name
            - model_family: Extracted model family
            - dimensions: Embedding dimension count
            - version: Model version (extracted if available, else "v1")
            - created_at: Current timestamp in ISO format
            - indexing_strategy: Search strategy (currently "hybrid")
        """
        from datetime import datetime, timezone

        model_name = self.settings.embeddings_model_name

        # Extract version if present in model name
        version = "v1"
        if "v2" in model_name.lower():
            version = "v2"
        elif "v3" in model_name.lower():
            version = "v3"
        elif "ada-002" in model_name:
            version = "ada-002"

        return {
            "provider": self.settings.embeddings_provider,
            "model": model_name,
            "model_family": self.model_family,
            "dimensions": self.settings.embeddings_model_dimensions,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "indexing_strategy": "hybrid"
        }


# Global settings instance
settings = Settings()

# Global embedding config instance
embedding_config = EmbeddingConfig(settings) 