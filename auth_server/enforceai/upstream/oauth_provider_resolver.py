from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Mapping,
    Optional,
)

from ..config import (
    EnforceAISettings,
    UpstreamOAuthProviderConfig,
)
from ..db.data_layer import (
    EnforceAIStores,
)
from ..secrets.upstream_oauth_client import (
    load_upstream_oauth_client_secret,
)


UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED: str = "UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED"


@dataclass(frozen=True)
class ResolvedUpstreamOAuthProvider:
    provider_id: str
    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    default_scopes: list[str]
    extra_authorize_params: dict[str, str]
    client_secret: Optional[str]
    source: str


def resolve_upstream_oauth_provider(
    *,
    provider_id: str,
    stores: EnforceAIStores,
    settings: Optional[EnforceAISettings] = None,
    env_providers: Optional[Mapping[str, UpstreamOAuthProviderConfig]] = None,
    require_client_secret: bool,
) -> Optional[ResolvedUpstreamOAuthProvider]:
    """Resolve upstream OAuth provider config for runtime usage.

    Resolution order:
    1) EnforceAI provider registry store (DB) when available
    2) Environment-based provider config (settings)

    Args:
        provider_id: Provider id to resolve.
        stores: EnforceAI stores bundle.
        settings: EnforceAI settings (env providers live here).
        env_providers: Optional explicit env provider mapping override.
        require_client_secret: Whether to also resolve the client secret.

    Returns:
        Resolved provider config, or None if not configured in either source.

    Raises:
        ValueError: If a configured provider is missing required secret material.
    """
    normalized_provider_id = provider_id.strip()
    if not normalized_provider_id:
        return None

    provider_store = getattr(stores, "upstream_oauth_provider_store", None)
    if provider_store is not None:
        public = provider_store.get_provider(provider_id=normalized_provider_id)
        if public is not None:
            client_secret: Optional[str] = None
            if require_client_secret:
                client_secret = provider_store.get_provider_secret_for_runtime(
                    provider_id=normalized_provider_id,
                )
                if client_secret is None or not client_secret.strip():
                    raise ValueError("Upstream OAuth provider secret is not configured")

            record = public.provider
            return ResolvedUpstreamOAuthProvider(
                provider_id=record.provider_id,
                authorization_endpoint=record.authorization_endpoint,
                token_endpoint=record.token_endpoint,
                client_id=record.client_id,
                default_scopes=list(record.default_scopes),
                extra_authorize_params=dict(record.extra_authorize_params),
                client_secret=client_secret,
                source="db",
            )

    candidates: Mapping[str, UpstreamOAuthProviderConfig]
    if env_providers is not None:
        candidates = env_providers
    elif settings is not None:
        candidates = settings.upstream_oauth_providers
    else:
        candidates = {}
    env_provider = candidates.get(normalized_provider_id)
    if env_provider is None:
        return None

    client_secret = None
    if require_client_secret:
        client_secret = load_upstream_oauth_client_secret(provider=env_provider)

    return ResolvedUpstreamOAuthProvider(
        provider_id=normalized_provider_id,
        authorization_endpoint=env_provider.authorization_endpoint,
        token_endpoint=env_provider.token_endpoint,
        client_id=env_provider.client_id,
        default_scopes=list(env_provider.default_scopes),
        extra_authorize_params=dict(env_provider.extra_authorize_params),
        client_secret=client_secret,
        source="env",
    )
