from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Optional,
)
from urllib.parse import urlencode

from ..config import (
    UpstreamOAuthProviderConfig,
)
from ..models.upstream_oauth import (
    UpstreamOAuthCredentialType,
)
from ..stores.sqlite.upstream_oauth_state_store import (
    SqliteUpstreamOAuthStateStore,
)
from .pkce import (
    compute_code_challenge,
    generate_code_verifier,
)


def _normalize_scopes(
    *,
    requested: Optional[list[str]],
    defaults: list[str],
    credential_type: UpstreamOAuthCredentialType,
) -> list[str]:
    scopes = requested if requested is not None else defaults
    normalized = [item.strip() for item in scopes if item.strip()]
    deduped = sorted(set(normalized))
    if credential_type == "oidc" and "openid" not in deduped:
        deduped.insert(0, "openid")
    return deduped


@dataclass(frozen=True)
class OAuthStartResult:
    authorization_url: str
    state_id: str
    expires_at: datetime


@dataclass(frozen=True)
class OAuthConsumeResult:
    state_id: str
    server_path: str
    provider: str
    credential_type: UpstreamOAuthCredentialType
    credential_binding: str
    user_id: str
    agent_id: Optional[str]
    redirect_uri: str
    code_verifier: str


def start_oauth_flow(
    *,
    state_store: SqliteUpstreamOAuthStateStore,
    provider: UpstreamOAuthProviderConfig,
    provider_id: str,
    server_path: str,
    credential_type: UpstreamOAuthCredentialType,
    credential_binding: str,
    user_id: str,
    agent_id: Optional[str],
    redirect_uri: str,
    scopes: Optional[list[str]],
    ttl_seconds: int,
) -> OAuthStartResult:
    code_verifier = generate_code_verifier()
    code_challenge = compute_code_challenge(code_verifier=code_verifier)
    nonce = secrets.token_urlsafe(16)

    normalized_scopes = _normalize_scopes(
        requested=scopes,
        defaults=provider.default_scopes,
        credential_type=credential_type,
    )

    state = state_store.create_state(
        server_path=server_path,
        credential_type=credential_type,
        credential_binding=credential_binding,
        user_id=user_id,
        agent_id=agent_id,
        provider=provider_id,
        redirect_uri=redirect_uri,
        ttl_seconds=ttl_seconds,
        secret_payload={
            "code_verifier": code_verifier,
            "nonce": nonce,
        },
    )

    query: dict[str, str] = {
        "response_type": "code",
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(normalized_scopes),
        "state": state.state_id,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if credential_type == "oidc":
        query["nonce"] = nonce

    for key, value in provider.extra_authorize_params.items():
        if key not in query:
            query[key] = value

    return OAuthStartResult(
        authorization_url=f"{provider.authorization_endpoint}?{urlencode(query)}",
        state_id=state.state_id,
        expires_at=state.expires_at,
    )


def consume_oauth_state(
    *,
    state_store: SqliteUpstreamOAuthStateStore,
    state_id: str,
    actor_user_id: str,
) -> OAuthConsumeResult:
    consumed = state_store.consume_state(state_id=state_id)
    if consumed is None:
        raise ValueError("OAuth state invalid or expired")

    record, secret = consumed
    if record.user_id != actor_user_id:
        raise ValueError("OAuth state does not match current user")

    code_verifier_raw = secret.payload.get("code_verifier")
    if not isinstance(code_verifier_raw, str) or not code_verifier_raw.strip():
        raise ValueError("OAuth state missing code_verifier")

    return OAuthConsumeResult(
        state_id=record.state_id,
        server_path=record.server_path,
        provider=record.provider,
        credential_type=record.credential_type,
        credential_binding=record.credential_binding,
        user_id=record.user_id,
        agent_id=record.agent_id,
        redirect_uri=record.redirect_uri,
        code_verifier=code_verifier_raw,
    )

