"""Settings for the openapi-mcp container.

``ContainerSettings`` holds the container's own configuration. ``GuardSettings``
holds only the fields ``registry/utils/url_guard.py`` reads, so the guard runs
here without the registry's ``Settings`` class (which requires ``SECRET_KEY``).
"""

from typing import Literal

from pydantic import (
    Field,
    SecretStr,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

DEFAULT_PORT: int = 9876
LogFormat = Literal["json", "text"]
# Inside a container the server must listen on every interface so nginx and the
# registry can reach it. Compose publishes the port on loopback only, and ECS and
# Kubernetes restrict ingress with a security group and a NetworkPolicy.
DEFAULT_HOST: str = "0.0.0.0"  # nosec B104 - container bind, ingress restricted per surface


class ContainerSettings(BaseSettings):
    """Configuration of the openapi-mcp container, read from the environment."""

    model_config = SettingsConfigDict(extra="ignore")

    openapi_mcp_sync_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Verifies registry sync tokens. Same value as the registry's.",
    )
    openapi_mcp_backend_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
        description="Timeout for each backend REST call.",
    )
    openapi_mcp_max_response_bytes: int = Field(
        default=2_097_152,
        ge=1024,
        le=104_857_600,
        description="Largest backend response passed to the agent.",
    )
    openapi_mcp_conversion_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        le=120,
        description="Longest one spec conversion may take.",
    )
    host: str = Field(default=DEFAULT_HOST)
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    build_version: str = Field(default="dev")
    # Application logs (docs/logging-standard.md). The same APP_LOG_* knobs as
    # the registry, auth server and mcpgw, read in logging_setup.py.
    app_log_dir: str = Field(
        default="",
        description="Directory for openapi-mcp.log. Empty means /var/log/containers/ai-registry.",
    )
    app_log_file_format: LogFormat = Field(default="json")
    app_log_console_format: LogFormat = Field(default="json")
    app_log_level: str = Field(default="INFO")
    app_log_max_bytes: int = Field(default=52_428_800, ge=1024)
    app_log_backup_count: int = Field(default=5, ge=0)


class GuardSettings(BaseSettings):
    """The settings ``url_guard`` reads, with the registry's defaults.

    ``tests/test_guard_settings.py`` fails when ``url_guard.py`` starts reading a
    field this class lacks, so a new guard setting cannot silently fall back to
    a missing attribute in the container.
    """

    model_config = SettingsConfigDict(extra="ignore")

    ssrf_allowed_hosts: str = ""
    ssrf_allowed_cidrs: str = ""
    gateway_proxy_allow_private_targets: bool = False
    github_extra_hosts: str = ""
    egress_oauth_trusted_idp_hosts: str = ""
    egress_http_pool_max_connections: int = 100
    egress_http_pool_max_keepalive: int = 20
    egress_http_pool_keepalive_expiry_seconds: float = 30.0
    egress_http_pool_connect_retries: int = 1
    # The registry-side OpenAPI identity is never needed here: the container
    # only calls third-party APIs, under the ordinary proxy profile.
    openapi_mcp_enabled: bool = False
    openapi_mcp_internal_url: str = "http://openapi-mcp:9876"
