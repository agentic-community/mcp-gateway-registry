"""Pydantic models for the registration gate (admission control webhook)."""

from enum import Enum

from pydantic import (
    BaseModel,
    Field,
)


class RegistrationGateAuthType(str, Enum):
    """Authentication type for calling the gate endpoint."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"


class RegistrationGateRequest(BaseModel):
    """Payload sent to the registration gate endpoint.

    The registration_payload is sanitized before sending:
    all credential fields (auth_credential, auth_credential_encrypted,
    tokens, secrets, API keys) are stripped to prevent leaking
    sensitive data to the external gate endpoint.
    """

    asset_type: str = Field(
        ...,
        description="Type of asset: 'agent', 'server', or 'skill'",
    )
    operation: str = Field(
        ...,
        description="Operation type: 'register' or 'update'",
    )
    source_api: str = Field(
        ...,
        description="Source API path that triggered the request",
    )
    registration_payload: dict = Field(
        ...,
        description="Sanitized registration request payload (credential fields removed)",
    )
    request_headers: dict[str, str] = Field(
        default_factory=dict,
        description="HTTP request headers (sensitive headers excluded)",
    )


class RegistrationGateResponse(BaseModel):
    """Expected response from the registration gate endpoint."""

    status: str = Field(
        ...,
        description="Gate decision: 'allowed' or 'denied'",
    )
    error: str | None = Field(
        default=None,
        description="Reason for denial (only present when status='denied')",
    )


class RegistrationGateResult(BaseModel):
    """Internal result from the gate service check."""

    allowed: bool = Field(
        ...,
        description="Whether the registration is allowed to proceed",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message to return to the caller if denied",
    )
    gate_status_code: int | None = Field(
        default=None,
        description="HTTP status code returned by the gate endpoint",
    )
    attempts: int = Field(
        default=0,
        description="Number of HTTP attempts made to the gate endpoint",
    )
