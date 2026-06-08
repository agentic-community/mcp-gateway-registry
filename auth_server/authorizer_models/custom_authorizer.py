"""
Pydantic models for the custom authorizer webhook contract.

They are shared between the service layer (services/custom_authorizer.py)
and any code that constructs or validates authorizer payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AuthorizerMode(str, Enum):
    """Controls how authorization decisions are made for every /validate call.

    - NATIVE:  Built-in JWT / OAuth2 / session validation only.  Default.
    - CUSTOM:  External authorizer only — ALL native validation is skipped.
    - BOTH:    Full native validation first; external authorizer consulted last.
               The custom authorizer receives the complete native result and
               can base its decision on it (username, scopes, groups, etc.).
    """

    NATIVE = "native"
    CUSTOM = "custom"
    BOTH = "both"


class CustomAuthRequest(BaseModel):
    """Details of the original HTTP request being authorized.

    Sensitive headers (Authorization, Cookie, …) are masked by
    ``mask_sensitive_headers`` before this object is populated.
    """

    method: str = Field(..., description="HTTP method (GET, POST, PUT, DELETE, …)")
    path: str = Field(..., description="Request path without query string")
    original_url: str = Field(
        ..., description="Full original URL including scheme, host, path, and query string"
    )
    query_params: dict[str, str] = Field(
        default_factory=dict,
        description="Query parameters as key-value pairs",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Request headers (sensitive values are masked)",
    )
    body: Optional[str] = Field(
        default=None,
        description="Request body if present",
    )
    client_ip: str = Field(..., description="Client IP address")


class NativeAuthResult(BaseModel):
    """Result from the auth server's native JWT / OAuth2 validation pipeline.

    Populated only when ``AUTHORIZER_MODE=both``; set to ``None`` when
    ``AUTHORIZER_MODE=custom`` (native validation is skipped entirely).
    """

    valid: bool = Field(..., description="Whether native validation succeeded")
    username: Optional[str] = Field(
        default=None,
        description="Authenticated username from the JWT (preferred_username claim)",
    )
    scopes: list[str] = Field(
        default_factory=list,
        description="OAuth2 scopes granted to the user",
    )
    groups: list[str] = Field(
        default_factory=list,
        description="User groups from the JWT token",
    )
    auth_method: Optional[str] = Field(
        default=None,
        description="Authentication method / provider used (keycloak, cognito, …)",
    )
    client_id: Optional[str] = Field(
        default=None,
        description="OAuth2 client_id that requested the token",
    )


class CustomAuthContext(BaseModel):
    """Context metadata attached to every outbound authorization request."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of the authorization call",
    )
    request_id: str = Field(..., description="Unique request ID for correlation and debugging")
    gateway_version: str = Field(
        default="1.0.0",
        description="Version of the MCP Gateway Registry",
    )


class CustomAuthorizerPayload(BaseModel):
    """Top-level payload POSTed to the custom authorizer endpoint.

    Schema reference: custom-authorizer-openapi.json → CustomAuthorizerPayload
    """

    request: CustomAuthRequest
    native_auth_result: Optional[NativeAuthResult] = Field(
        default=None,
        description=(
            "Native JWT/OAuth2 result.  Null when AUTHORIZER_MODE=custom "
            "(native validation is skipped)."
        ),
    )
    context: CustomAuthContext


class CustomAuthErrorDetail(BaseModel):
    """Error details returned when the custom authorizer denies a request."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional error context",
    )


class CustomAuthorizerResponse(BaseModel):
    """Response from the custom authorizer endpoint.

    ``authorized`` is the only required field; ``metadata`` and ``error``
    are optional and used for auditing / debugging.
    """

    authorized: bool = Field(..., description="Whether access is authorized")
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional authorizer-specific audit data",
    )
    error: Optional[CustomAuthErrorDetail] = Field(
        default=None,
        description="Error details if not authorized",
    )
