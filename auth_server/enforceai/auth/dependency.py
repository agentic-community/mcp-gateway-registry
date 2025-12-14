from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from fastapi import (
    Depends,
    HTTPException,
    Request,
)
from pydantic import ValidationError

from .resolver import (
    IdentityResolver,
)
from ..config import (
    EnforceAISettings,
)
from ..db.data_layer import (
    EnforceAIDataLayer,
    EnforceAIStores,
)
from ..errors import (
    DependencyUnavailableError,
    EnforceAIError,
)
from ..fgac.catalog import (
    load_scope_catalog,
)
from ..fgac.models import (
    ScopeCatalog,
)
from ..identity import (
    IdentityContext,
)
from ..oidc.jwks import (
    JWKSCache,
)
from ..oidc.verify import (
    OIDCVerifier,
)
from ..providers.api_key import (
    ApiKeyProvider,
)
from ..providers.gateway_token import (
    GatewayTokenProvider,
)
from ..providers.oidc import (
    OidcProvider,
)

logger = logging.getLogger(__name__)

IDENTITY_STATE_KEY: str = "enforceai_identity"
CATALOG_STATE_KEY: str = "enforceai_scope_catalog"


@dataclass(frozen=True)
class EnforceAIRequestContext:
    identity: IdentityContext
    catalog: ScopeCatalog


def _map_to_http_exception(
    exc: EnforceAIError,
) -> HTTPException:
    return exc.as_http_exception()


@lru_cache(maxsize=1)
def get_enforceai_settings() -> EnforceAISettings:
    try:
        return EnforceAISettings()
    except ValidationError as exc:
        raise DependencyUnavailableError(
            "EnforceAI settings validation failed",
            public_message="Enforcement misconfigured",
        ) from exc


@lru_cache(maxsize=1)
def get_enforceai_stores() -> EnforceAIStores:
    settings = get_enforceai_settings()

    data_layer = EnforceAIDataLayer(db_path=settings.db_path)
    data_layer.initialize()
    return data_layer.build_stores()


@lru_cache(maxsize=1)
def get_jwks_cache() -> JWKSCache:
    return JWKSCache()


@lru_cache(maxsize=1)
def get_identity_resolver() -> IdentityResolver:
    settings = get_enforceai_settings()
    stores = get_enforceai_stores()

    oidc_provider: Optional[OidcProvider] = None
    api_key_provider: Optional[ApiKeyProvider] = None
    gateway_token_provider: Optional[GatewayTokenProvider] = None

    if settings.auth_provider in {"oidc", "mixed"}:
        verifier = OIDCVerifier(
            issuers=settings.oidc_issuers,
            jwks_cache=get_jwks_cache(),
        )
        oidc_provider = OidcProvider(
            verifier=verifier,
            agent_store=stores.agent_store,
        )

    if settings.auth_provider in {"api-key", "mixed"}:
        if settings.api_key_pepper_path is None:
            raise DependencyUnavailableError(
                "API key pepper path missing",
                public_message="Enforcement misconfigured",
            )
        api_key_provider = ApiKeyProvider(
            api_key_store=stores.api_key_store,
            agent_store=stores.agent_store,
            pepper_path=settings.api_key_pepper_path,
        )

    if settings.auth_provider in {"gateway-token", "mixed"}:
        if (
            settings.gateway_private_key_path is None
            or settings.gateway_public_keys_dir is None
            or settings.gateway_active_kid is None
            or settings.gateway_issuer is None
        ):
            raise DependencyUnavailableError(
                "Gateway token configuration missing",
                public_message="Enforcement misconfigured",
            )

        gateway_token_provider = GatewayTokenProvider(
            agent_store=stores.agent_store,
            revocation_store=stores.revocation_store,
            private_key_path=settings.gateway_private_key_path,
            public_keys_dir=settings.gateway_public_keys_dir,
            active_kid=settings.gateway_active_kid,
            expected_issuer=settings.gateway_issuer,
        )

    return IdentityResolver(
        auth_provider=settings.auth_provider,
        oidc_provider=oidc_provider,
        api_key_provider=api_key_provider,
        gateway_token_provider=gateway_token_provider,
        gateway_issuer=settings.gateway_issuer,
        oidc_issuers=set(settings.oidc_issuers.keys()),
    )


def get_scope_catalog(
    settings: EnforceAISettings = Depends(get_enforceai_settings),
) -> ScopeCatalog:
    try:
        return load_scope_catalog(path=settings.scopes_catalog_path)
    except DependencyUnavailableError:
        raise
    except ValueError as exc:
        raise DependencyUnavailableError(
            "Invalid scope catalog",
            public_message="Scope catalog unavailable",
        ) from exc


async def get_enforceai_request_context(
    request: Request,
    resolver: IdentityResolver = Depends(get_identity_resolver),
    catalog: ScopeCatalog = Depends(get_scope_catalog),
) -> EnforceAIRequestContext:
    """FastAPI dependency: resolve identity and load FGAC catalog once per request."""

    try:
        identity = await resolver.resolve_identity(headers=request.headers)
    except EnforceAIError as exc:
        raise _map_to_http_exception(exc) from exc
    except Exception as exc:  # noqa: BLE001 - fail closed, signal retry
        logger.exception("Unexpected enforcement dependency failure")
        raise HTTPException(
            status_code=503,
            detail="Enforcement dependency unavailable",
        ) from exc

    setattr(
        request.state,
        IDENTITY_STATE_KEY,
        identity,
    )
    setattr(
        request.state,
        CATALOG_STATE_KEY,
        catalog,
    )

    return EnforceAIRequestContext(
        identity=identity,
        catalog=catalog,
    )


def clear_enforceai_dependency_caches() -> None:
    get_enforceai_settings.cache_clear()
    get_enforceai_stores.cache_clear()
    get_jwks_cache.cache_clear()
    get_identity_resolver.cache_clear()
