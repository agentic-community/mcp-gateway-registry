from __future__ import annotations

from typing import Literal, Mapping, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from auth_server.enforceai.errors import UnauthorizedError

CredentialKind = Literal[
    "bearer",
    "gateway-token",
    "api-key",
]


def _get_header_value(
    headers: Mapping[str, str],
    header_name: str,
) -> Optional[str]:
    raw = headers.get(header_name)
    if raw is None:
        return None

    stripped = raw.strip()
    if not stripped:
        return None

    return stripped


def _extract_bearer_token(
    raw_value: str,
    *,
    header_name: str,
) -> str:
    parts = raw_value.strip().split(None, 1)
    if len(parts) != 2:
        raise UnauthorizedError(
            f"{header_name} must be in 'Bearer <token>' format",
        )

    scheme, token = parts
    if scheme.lower() != "bearer":
        raise UnauthorizedError(
            f"{header_name} must use the Bearer scheme",
        )

    normalized_token = token.strip()
    if not normalized_token:
        raise UnauthorizedError(
            f"{header_name} must be in 'Bearer <token>' format",
        )

    return normalized_token


def _normalize_headers(
    headers: Mapping[str, str],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in headers.items():
        normalized[key.lower()] = value
    return normalized


class CredentialInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    kind: CredentialKind
    value: str = Field(
        ...,
        min_length=1,
    )
    agent_id_header: Optional[str] = None


def extract_credential_input(
    headers: Mapping[str, str],
) -> CredentialInput:
    """Extract exactly one credential input from request headers.

    Args:
        headers: Request headers mapping. Header names are treated
            case-insensitively.

    Returns:
        Parsed CredentialInput.

    Raises:
        UnauthorizedError: If no credentials are present, credentials are
            ambiguous, or a bearer authorization header is malformed.
    """

    normalized_headers = _normalize_headers(headers)

    authorization = _get_header_value(
        normalized_headers,
        "authorization",
    )
    x_authorization = _get_header_value(
        normalized_headers,
        "x-authorization",
    )
    x_gateway_token = _get_header_value(
        normalized_headers,
        "x-gateway-token",
    )
    x_api_key = _get_header_value(
        normalized_headers,
        "x-api-key",
    )
    x_agent_id = _get_header_value(
        normalized_headers,
        "x-agent-id",
    )

    credential_candidates: list[tuple[CredentialKind, str, str]] = []
    if authorization is not None:
        credential_candidates.append(
            (
                "bearer",
                "Authorization",
                authorization,
            )
        )
    if x_authorization is not None:
        credential_candidates.append(
            (
                "bearer",
                "X-Authorization",
                x_authorization,
            )
        )
    if x_gateway_token is not None:
        credential_candidates.append(
            (
                "gateway-token",
                "X-Gateway-Token",
                x_gateway_token,
            )
        )
    if x_api_key is not None:
        credential_candidates.append(
            (
                "api-key",
                "X-API-Key",
                x_api_key,
            )
        )

    if not credential_candidates:
        raise UnauthorizedError("No credentials provided")

    if len(credential_candidates) != 1:
        provided_headers = ", ".join(
            header_name
            for _kind, header_name, _value in credential_candidates
        )
        raise UnauthorizedError(
            f"Multiple credentials provided: {provided_headers}",
        )

    kind, header_name, raw_value = credential_candidates[0]

    if kind == "bearer":
        value = _extract_bearer_token(
            raw_value,
            header_name=header_name,
        )
    else:
        value = raw_value.strip()

    return CredentialInput(
        kind=kind,
        value=value,
        agent_id_header=x_agent_id,
    )
