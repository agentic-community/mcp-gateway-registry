from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import (
    Any,
    Mapping,
    Optional,
)

from ..db.data_layer import (
    EnforceAIStores,
)
from ..identity import (
    IdentityContext,
)
from ..models.upstream_auth import (
    UpstreamAuthConfig,
    UpstreamAuthInjection,
)
from ..models.upstream_credentials import (
    UpstreamCredentialRecord,
)
from .oauth_client import (
    OAuthTokenClient,
    OAuthTokenClientError,
)
from ..config import (
    UpstreamOAuthProviderConfig,
)
from .oauth_provider_resolver import (
    UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED,
    resolve_upstream_oauth_provider,
)

_CLAIMS_MAX_BYTES: int = 4096
_OAUTH_DEFAULT_ACCESS_TOKEN_TTL_SECONDS: int = 3600


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _canonicalize_server_path(
    value: Optional[str],
) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if any(ch in stripped for ch in ("\r", "\n")):
        raise ValueError("server_path must not contain newline characters")
    if not stripped.startswith("/"):
        stripped = f"/{stripped}"
    return stripped.rstrip("/") or "/"


def _normalize_optional_header_value(
    value: Optional[str],
) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if any(ch in stripped for ch in ("\r", "\n")):
        raise ValueError("Header values must not contain newline characters")
    return stripped


def _build_mcp_claims(
    *,
    identity: IdentityContext,
) -> str:
    payload = {
        "user_id": identity.user_id,
        "agent_id": identity.agent_id,
        "provider": identity.provider,
    }
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    if len(encoded) > _CLAIMS_MAX_BYTES:
        raise ValueError("X-MCP-Claims payload too large")
    return encoded.decode("utf-8")


def _select_active_credential(
    *,
    store,
    server_path: str,
    credential_type: str,
    credential_binding: str,
    user_id: Optional[str],
    agent_id: Optional[str],
    provider: Optional[str],
) -> tuple[Optional[UpstreamCredentialRecord], str]:
    candidates = [
        record
        for record in store.list_credentials(
            server_path=server_path,
            user_id=user_id,
            agent_id=agent_id,
            include_revoked=False,
        )
        if record.credential_type == credential_type
        and record.credential_binding == credential_binding
        and (provider is None or record.provider == provider)
    ]

    now = _utc_now()
    active = []
    expired = []
    for record in candidates:
        if record.expires_at is not None and record.expires_at <= now:
            expired.append(record)
            continue
        active.append(record)

    if len(active) == 1:
        return active[0], "active"
    if len(active) > 1:
        raise UpstreamInjectionError(
            status_code=409,
            error_code="UPSTREAM_AUTH_MISCONFIGURED",
            public_message="Multiple active upstream credentials configured",
        )

    if len(expired) == 1:
        return expired[0], "expired"
    if len(expired) > 1:
        raise UpstreamInjectionError(
            status_code=409,
            error_code="UPSTREAM_AUTH_MISCONFIGURED",
            public_message="Multiple expired upstream credentials configured",
        )
    return None, "missing"


def _extract_api_key(
    payload: Mapping[str, Any],
) -> str:
    raw = payload.get("api_key")
    if not isinstance(raw, str):
        raise UpstreamInjectionError(
            status_code=409,
            error_code="UPSTREAM_AUTH_MISCONFIGURED",
            public_message="Upstream api-key credential payload must include api_key",
        )
    normalized = _normalize_optional_header_value(raw)
    if normalized is None:
        raise UpstreamInjectionError(
            status_code=409,
            error_code="UPSTREAM_AUTH_MISCONFIGURED",
            public_message="Upstream api-key credential payload api_key must be non-empty",
        )
    return normalized


def _extract_bearer_token(
    payload: Mapping[str, Any],
) -> str:
    raw = payload.get("token")
    if not isinstance(raw, str):
        raw = payload.get("jwt")
    if not isinstance(raw, str):
        raise UpstreamInjectionError(
            status_code=409,
            error_code="UPSTREAM_AUTH_MISCONFIGURED",
            public_message="Upstream jwt credential payload must include token",
        )
    normalized = _normalize_optional_header_value(raw)
    if normalized is None:
        raise UpstreamInjectionError(
            status_code=409,
            error_code="UPSTREAM_AUTH_MISCONFIGURED",
            public_message="Upstream jwt credential payload token must be non-empty",
        )
    return normalized


def _extract_oauth_access_token(
    payload: Mapping[str, Any],
) -> str:
    raw = payload.get("access_token")
    if not isinstance(raw, str):
        raw = payload.get("token")
    return _extract_bearer_token({"token": raw} if raw is not None else payload)


def _extract_refresh_token(
    payload: Mapping[str, Any],
) -> Optional[str]:
    raw = payload.get("refresh_token")
    if not isinstance(raw, str):
        return None
    normalized = _normalize_optional_header_value(raw)
    return normalized


def _compose_authorization_value(
    *,
    scheme: Optional[str],
    token: str,
) -> str:
    if scheme is None:
        return token
    normalized_scheme = _normalize_optional_header_value(scheme)
    if normalized_scheme is None:
        return token
    return f"{normalized_scheme} {token}"


@dataclass(frozen=True)
class UpstreamInjectionResult:
    mcp_principal: str
    mcp_auth_type: str
    mcp_scopes: str
    mcp_provider: str
    mcp_claims: str

    mode: str
    upstream_authorization: str
    upstream_api_key: str
    upstream_api_key_header: str


class UpstreamInjectionError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        error_code: str,
        public_message: str,
    ) -> None:
        super().__init__(public_message)
        self.status_code = status_code
        self.error_code = error_code
        self.public_message = public_message


async def resolve_upstream_injection(
    *,
    server_path: Optional[str],
    upstream_auth: UpstreamAuthConfig,
    identity: IdentityContext,
    stores: EnforceAIStores,
    oauth_providers: Optional[Mapping[str, UpstreamOAuthProviderConfig]] = None,
    oauth_token_client: Optional[OAuthTokenClient] = None,
    oauth_refresh_skew_seconds: int = 60,
    allow_missing_credential: bool = False,
) -> UpstreamInjectionResult:
    canonical_server_path = _canonicalize_server_path(server_path)

    mcp_principal = f"user:{identity.user_id}"
    mcp_auth_type = identity.provider
    mcp_scopes = " ".join(identity.scopes)
    mcp_provider = upstream_auth.provider or ""
    mcp_claims = _build_mcp_claims(identity=identity)

    if upstream_auth.type == "none" or canonical_server_path is None:
        return UpstreamInjectionResult(
            mcp_principal=mcp_principal,
            mcp_auth_type=mcp_auth_type,
            mcp_scopes=mcp_scopes,
            mcp_provider=mcp_provider,
            mcp_claims=mcp_claims,
            mode="none",
            upstream_authorization="",
            upstream_api_key="",
            upstream_api_key_header="",
        )

    if upstream_auth.type == "header-trust":
        return UpstreamInjectionResult(
            mcp_principal=mcp_principal,
            mcp_auth_type=mcp_auth_type,
            mcp_scopes=mcp_scopes,
            mcp_provider=mcp_provider,
            mcp_claims=mcp_claims,
            mode="header-trust",
            upstream_authorization="",
            upstream_api_key="",
            upstream_api_key_header="",
        )

    injection: Optional[UpstreamAuthInjection] = upstream_auth.injection
    if injection is None:
        raise UpstreamInjectionError(
            status_code=409,
            error_code="UPSTREAM_AUTH_MISCONFIGURED",
            public_message="Upstream auth injection is missing for this server",
        )

    store = stores.upstream_credential_store
    if store is None:
        raise UpstreamInjectionError(
            status_code=503,
            error_code="UPSTREAM_AUTH_MISCONFIGURED",
            public_message="Upstream credential store unavailable",
        )

    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    if upstream_auth.credential_binding == "user":
        user_id = identity.user_id
    elif upstream_auth.credential_binding == "agent":
        agent_id = identity.agent_id
    elif upstream_auth.credential_binding == "user+agent":
        user_id = identity.user_id
        agent_id = identity.agent_id
    elif upstream_auth.credential_binding == "service":
        user_id = None
        agent_id = None
    else:
        raise UpstreamInjectionError(
            status_code=409,
            error_code="UPSTREAM_AUTH_MISCONFIGURED",
            public_message="Unsupported upstream credential binding",
        )

    record, state = _select_active_credential(
        store=store,
        server_path=canonical_server_path,
        credential_type=upstream_auth.type,
        credential_binding=upstream_auth.credential_binding,
        user_id=user_id,
        agent_id=agent_id,
        provider=upstream_auth.provider,
    )
    if record is None:
        if allow_missing_credential:
            return UpstreamInjectionResult(
                mcp_principal=mcp_principal,
                mcp_auth_type=mcp_auth_type,
                mcp_scopes=mcp_scopes,
                mcp_provider=mcp_provider,
                mcp_claims=mcp_claims,
                mode="none",
                upstream_authorization="",
                upstream_api_key="",
                upstream_api_key_header="",
            )
        if state == "expired":
            raise UpstreamInjectionError(
                status_code=424,
                error_code="UPSTREAM_CREDENTIALS_EXPIRED",
                public_message="Upstream credential expired",
            )
        raise UpstreamInjectionError(
            status_code=424,
            error_code="UPSTREAM_CREDENTIALS_REQUIRED",
            public_message="Upstream credential required",
        )

    secret = store.get_credential_secret(credential_id=record.credential_id)
    payload: Mapping[str, Any] = secret.payload if secret is not None else {}

    if upstream_auth.type == "api-key":
        api_key = _extract_api_key(payload)
        store.update_last_used_at(credential_id=record.credential_id, last_used_at=_utc_now())
        return UpstreamInjectionResult(
            mcp_principal=mcp_principal,
            mcp_auth_type=mcp_auth_type,
            mcp_scopes=mcp_scopes,
            mcp_provider=mcp_provider,
            mcp_claims=mcp_claims,
            mode="api-key",
            upstream_authorization="",
            upstream_api_key=api_key,
            upstream_api_key_header=injection.header_name,
        )

    if upstream_auth.type == "jwt":
        token = _extract_bearer_token(payload)
        authorization = _compose_authorization_value(scheme=injection.scheme, token=token)
        store.update_last_used_at(credential_id=record.credential_id, last_used_at=_utc_now())
        return UpstreamInjectionResult(
            mcp_principal=mcp_principal,
            mcp_auth_type=mcp_auth_type,
            mcp_scopes=mcp_scopes,
            mcp_provider=mcp_provider,
            mcp_claims=mcp_claims,
            mode="bearer",
            upstream_authorization=authorization,
            upstream_api_key="",
            upstream_api_key_header="",
        )

    if upstream_auth.type in {"oauth2", "oidc", "provider-oauth"}:
        if upstream_auth.provider is None or not upstream_auth.provider.strip():
            raise UpstreamInjectionError(
                status_code=409,
                error_code="UPSTREAM_AUTH_MISCONFIGURED",
                public_message="Upstream OAuth provider is required",
            )

        access_token = _extract_oauth_access_token(payload)
        now = _utc_now()
        needs_refresh = False
        if record.expires_at is not None:
            refresh_at = record.expires_at - timedelta(seconds=oauth_refresh_skew_seconds)
            if refresh_at <= now:
                needs_refresh = True

        if needs_refresh:
            refresh_token = _extract_refresh_token(payload)
            if refresh_token is None:
                raise UpstreamInjectionError(
                    status_code=424,
                    error_code="UPSTREAM_CREDENTIALS_EXPIRED",
                    public_message="Upstream credential expired",
                )
            if oauth_token_client is None:
                raise UpstreamInjectionError(
                    status_code=503,
                    error_code="UPSTREAM_AUTH_MISCONFIGURED",
                    public_message="Upstream OAuth token client unavailable",
                )

            try:
                resolved_provider = resolve_upstream_oauth_provider(
                    provider_id=upstream_auth.provider,
                    stores=stores,
                    env_providers=oauth_providers,
                    require_client_secret=True,
                )
            except ValueError as exc:
                raise UpstreamInjectionError(
                    status_code=424,
                    error_code=UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED,
                    public_message="Upstream OAuth provider not configured",
                ) from exc

            if resolved_provider is None or resolved_provider.client_secret is None:
                raise UpstreamInjectionError(
                    status_code=424,
                    error_code=UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED,
                    public_message="Upstream OAuth provider not configured",
                )

            try:
                refreshed = await oauth_token_client.refresh_token(
                    token_endpoint=resolved_provider.token_endpoint,
                    client_id=resolved_provider.client_id,
                    client_secret=resolved_provider.client_secret,
                    refresh_token=refresh_token,
                )
            except OAuthTokenClientError as exc:
                raise UpstreamInjectionError(
                    status_code=424,
                    error_code="UPSTREAM_OAUTH_REFRESH_FAILED",
                    public_message="Upstream OAuth refresh failed",
                ) from exc

            new_secret_payload: dict[str, object] = dict(payload)
            new_secret_payload["access_token"] = refreshed.access_token
            if refreshed.refresh_token is not None:
                new_secret_payload["refresh_token"] = refreshed.refresh_token
            if refreshed.id_token is not None:
                new_secret_payload["id_token"] = refreshed.id_token

            updated_expires_at = refreshed.expires_at
            if updated_expires_at is None:
                updated_expires_at = now + timedelta(seconds=_OAUTH_DEFAULT_ACCESS_TOKEN_TTL_SECONDS)

            store.update_credential(
                credential_id=record.credential_id,
                token_type=refreshed.token_type,
                scopes=refreshed.scopes,
                expires_at=updated_expires_at,
                secret_payload=new_secret_payload,
            )
            access_token = refreshed.access_token

        try:
            resolved_provider = resolve_upstream_oauth_provider(
                provider_id=upstream_auth.provider,
                stores=stores,
                env_providers=oauth_providers,
                require_client_secret=False,
            )
        except ValueError as exc:
            raise UpstreamInjectionError(
                status_code=424,
                error_code=UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED,
                public_message="Upstream OAuth provider not configured",
            ) from exc

        if resolved_provider is None:
            raise UpstreamInjectionError(
                status_code=424,
                error_code=UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED,
                public_message="Upstream OAuth provider not configured",
            )

        authorization = _compose_authorization_value(scheme=injection.scheme, token=access_token)
        store.update_last_used_at(credential_id=record.credential_id, last_used_at=_utc_now())
        return UpstreamInjectionResult(
            mcp_principal=mcp_principal,
            mcp_auth_type=mcp_auth_type,
            mcp_scopes=mcp_scopes,
            mcp_provider=mcp_provider,
            mcp_claims=mcp_claims,
            mode="bearer",
            upstream_authorization=authorization,
            upstream_api_key="",
            upstream_api_key_header="",
        )

    raise UpstreamInjectionError(
        status_code=409,
        error_code="UPSTREAM_AUTH_MISCONFIGURED",
        public_message=f"Unsupported upstream auth type for Phase 5: {upstream_auth.type}",
    )
