from __future__ import annotations

from typing import Mapping, Optional

import jwt

from auth_server.enforceai.auth.credentials import (
    CredentialInput,
    extract_credential_input,
)
from auth_server.enforceai.config import (
    AuthProviderMode,
)
from auth_server.enforceai.errors import (
    UnauthorizedError,
)
from auth_server.enforceai.identity import (
    IdentityContext,
)
from auth_server.enforceai.providers.api_key import (
    ApiKeyProvider,
)
from auth_server.enforceai.providers.gateway_token import (
    GatewayTokenProvider,
)
from auth_server.enforceai.providers.oidc import (
    OidcProvider,
)


def _peek_unverified_issuer(
    token: str,
) -> str:
    try:
        claims = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_aud": False,
            },
        )
    except Exception as exc:  # noqa: BLE001 - map to 401
        raise UnauthorizedError("Invalid bearer token") from exc

    issuer = claims.get("iss")
    if not isinstance(issuer, str) or not issuer.strip():
        raise UnauthorizedError("Invalid bearer token")
    return issuer.strip()


class IdentityResolver:
    def __init__(
        self,
        *,
        auth_provider: AuthProviderMode,
        oidc_provider: Optional[OidcProvider] = None,
        api_key_provider: Optional[ApiKeyProvider] = None,
        gateway_token_provider: Optional[GatewayTokenProvider] = None,
        gateway_issuer: Optional[str] = None,
        oidc_issuers: Optional[set[str]] = None,
    ) -> None:
        self._auth_provider = auth_provider
        self._oidc_provider = oidc_provider
        self._api_key_provider = api_key_provider
        self._gateway_token_provider = gateway_token_provider
        self._gateway_issuer = gateway_issuer
        self._oidc_issuers = oidc_issuers or set()

    async def resolve_identity(
        self,
        *,
        headers: Mapping[str, str],
    ) -> IdentityContext:
        credential = extract_credential_input(headers)

        if self._auth_provider == "api-key":
            return self._resolve_api_key_only(credential=credential)

        if self._auth_provider == "gateway-token":
            return self._resolve_gateway_token_only(credential=credential)

        if self._auth_provider == "oidc":
            return await self._resolve_oidc_only(credential=credential)

        if self._auth_provider == "mixed":
            return await self._resolve_mixed(credential=credential)

        raise UnauthorizedError("Unsupported auth mode")

    def _resolve_api_key_only(
        self,
        *,
        credential: CredentialInput,
    ) -> IdentityContext:
        if credential.kind != "api-key":
            raise UnauthorizedError("Unauthorized")
        if self._api_key_provider is None:
            raise UnauthorizedError("API key provider not configured")
        return self._api_key_provider.resolve_identity(
            api_key_value=credential.value,
        )

    def _resolve_gateway_token_only(
        self,
        *,
        credential: CredentialInput,
    ) -> IdentityContext:
        if credential.kind not in {"gateway-token", "bearer"}:
            raise UnauthorizedError("Unauthorized")
        if self._gateway_token_provider is None:
            raise UnauthorizedError("Gateway token provider not configured")
        return self._gateway_token_provider.resolve_identity(
            token=credential.value,
        )

    async def _resolve_oidc_only(
        self,
        *,
        credential: CredentialInput,
    ) -> IdentityContext:
        if credential.kind != "bearer":
            raise UnauthorizedError("Unauthorized")
        if self._oidc_provider is None:
            raise UnauthorizedError("OIDC provider not configured")
        return await self._oidc_provider.resolve_identity(
            bearer_token=credential.value,
            agent_id_header=credential.agent_id_header,
        )

    async def _resolve_mixed(
        self,
        *,
        credential: CredentialInput,
    ) -> IdentityContext:
        if credential.kind == "api-key":
            if self._api_key_provider is None:
                raise UnauthorizedError("API key provider not configured")
            return self._api_key_provider.resolve_identity(
                api_key_value=credential.value,
            )

        if credential.kind == "gateway-token":
            if self._gateway_token_provider is None:
                raise UnauthorizedError("Gateway token provider not configured")
            return self._gateway_token_provider.resolve_identity(
                token=credential.value,
            )

        if credential.kind == "bearer":
            issuer = _peek_unverified_issuer(credential.value)

            if self._gateway_issuer is not None and issuer == self._gateway_issuer:
                if self._gateway_token_provider is None:
                    raise UnauthorizedError("Gateway token provider not configured")
                return self._gateway_token_provider.resolve_identity(
                    token=credential.value,
                )

            if issuer in self._oidc_issuers:
                if self._oidc_provider is None:
                    raise UnauthorizedError("OIDC provider not configured")
                return await self._oidc_provider.resolve_identity(
                    bearer_token=credential.value,
                    agent_id_header=credential.agent_id_header,
                )

            raise UnauthorizedError("Unauthorized")

        raise UnauthorizedError("Unauthorized")
