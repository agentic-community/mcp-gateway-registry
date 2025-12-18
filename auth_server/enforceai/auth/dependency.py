from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from datetime import (
    datetime,
    timezone,
)
from fastapi import (
    Depends,
    HTTPException,
    Request,
)
from pydantic import ValidationError
from itsdangerous import (
    BadSignature,
    SignatureExpired,
    URLSafeTimedSerializer,
)

from .credentials import (
    extract_credential_input,
)
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
    UnauthorizedError,
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
from ..secrets.upstream_kek import (
    load_upstream_kek,
)
from ..upstream.oauth_client import (
    OAuthTokenClient,
)
from gateway_session import (
    normalize_session_data,
)

logger = logging.getLogger(__name__)

IDENTITY_STATE_KEY: str = "enforceai_identity"
CATALOG_STATE_KEY: str = "enforceai_scope_catalog"
SESSION_COOKIE_NAME: str = "mcp_gateway_session"
SESSION_MAX_AGE_SECONDS: int = 60 * 60 * 8
ENFORCEAI_ADMIN_GROUP: str = "enforceai-admin"


@dataclass(frozen=True)
class EnforceAIRequestContext:
    identity: IdentityContext
    catalog: ScopeCatalog


@dataclass(frozen=True)
class EnforceAIManagementContext:
    user_id: str
    actor_agent_id: str
    provider: str
    groups: list[str]
    is_admin: bool
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
    upstream_kek: Optional[bytes] = None
    if settings.upstream_kek_path is not None:
        try:
            upstream_kek = load_upstream_kek(settings.upstream_kek_path)
        except ValueError as exc:
            raise DependencyUnavailableError(
                "Invalid upstream KEK",
                public_message="Enforcement misconfigured",
            ) from exc

    return data_layer.build_stores(upstream_kek=upstream_kek)


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


@lru_cache(maxsize=1)
def get_upstream_oauth_token_client() -> OAuthTokenClient:
    return OAuthTokenClient()


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


def _has_header_credentials(
    request: Request,
) -> bool:
    try:
        extract_credential_input(request.headers)
        return True
    except EnforceAIError as exc:
        if str(exc) == "No credentials provided":
            return False
        raise


def _get_cookie_signer(
    request: Request,
) -> URLSafeTimedSerializer:
    signer = getattr(request.app.state, "session_signer", None)
    if isinstance(signer, URLSafeTimedSerializer):
        return signer

    secret_key = getattr(request.app.state, "session_secret_key", None)
    if isinstance(secret_key, str) and secret_key.strip():
        return URLSafeTimedSerializer(secret_key.strip())

    raise DependencyUnavailableError(
        "Session signer unavailable",
        public_message="Enforcement misconfigured",
    )


def _resolve_management_context_from_cookie(
    *,
    request: Request,
    stores: EnforceAIStores,
    catalog: ScopeCatalog,
) -> EnforceAIManagementContext:
    cookie_value = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie_value is None or not cookie_value.strip():
        raise UnauthorizedError("No credentials provided")

    signer = _get_cookie_signer(request)
    try:
        payload = signer.loads(
            cookie_value,
            max_age=SESSION_MAX_AGE_SECONDS,
        )
    except SignatureExpired as exc:
        raise UnauthorizedError("Session expired") from exc
    except BadSignature as exc:
        raise UnauthorizedError("Invalid session") from exc
    except Exception as exc:  # noqa: BLE001 - treat as invalid session
        raise UnauthorizedError("Invalid session") from exc

    if not isinstance(payload, dict):
        raise UnauthorizedError("Invalid session")

    normalized = normalize_session_data(
        payload,
        default_provider="local",
        max_age_seconds=SESSION_MAX_AGE_SECONDS,
    )

    record = stores.session_store.get_session_by_id(session_id=normalized.session_id)
    if record is None or record.revoked_at is not None:
        raise UnauthorizedError("Session invalidated")

    stores.session_store.touch_session(
        session_id=normalized.session_id,
        now=datetime.now(timezone.utc).replace(microsecond=0),
    )

    groups: list[str] = normalized.groups or []
    is_admin = ENFORCEAI_ADMIN_GROUP in groups

    if normalized.auth_method == "oidc" and normalized.email:
        stores.user_store.upsert_oidc_user(
            user_id=normalized.user_id,
            email=normalized.email,
            role="admin" if is_admin else "user",
        )

    return EnforceAIManagementContext(
        user_id=normalized.user_id,
        actor_agent_id=normalized.session_id,
        provider="session-cookie",
        groups=groups,
        is_admin=is_admin,
        catalog=catalog,
    )


async def get_enforceai_management_context(
    request: Request,
    resolver: IdentityResolver = Depends(get_identity_resolver),
    catalog: ScopeCatalog = Depends(get_scope_catalog),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> EnforceAIManagementContext:
    """FastAPI dependency: allow management auth via cookie-session or existing credentials."""

    try:
        if _has_header_credentials(request):
            identity = await resolver.resolve_identity(headers=request.headers)
            groups = identity.user_roles or []
            is_admin = ENFORCEAI_ADMIN_GROUP in groups
            return EnforceAIManagementContext(
                user_id=identity.user_id,
                actor_agent_id=identity.agent_id,
                provider=identity.provider,
                groups=list(groups),
                is_admin=is_admin,
                catalog=catalog,
            )

        return _resolve_management_context_from_cookie(
            request=request,
            stores=stores,
            catalog=catalog,
        )
    except EnforceAIError as exc:
        raise _map_to_http_exception(exc) from exc
    except Exception as exc:  # noqa: BLE001 - fail closed, signal retry
        logger.exception("Unexpected enforcement dependency failure")
        raise HTTPException(
            status_code=503,
            detail="Enforcement dependency unavailable",
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
    get_upstream_oauth_token_client.cache_clear()
