#!/usr/bin/env python3
"""
Sample Custom Authorizer for MCP Gateway Registry.

A self-contained FastAPI application that implements the custom authorizer
contract defined in custom-authorizer-openapi.json.  Use this for local
testing and as a reference when building your own authorizer.

Usage:
    pip install fastapi uvicorn pydantic
    uvicorn sample-custom-authorizer:app --host 0.0.0.0 --port 8090

Configuration (environment variables):
    ALLOWED_USERNAMES   Comma-separated substrings; a request whose username
                        contains any of them is allowed.
                        Default: "admin,service-account"
    BLOCKED_USERNAMES   Comma-separated substrings; checked before the allow
                        list.  A match results in an immediate deny.
                        Default: "blocked,suspended"
    DEFAULT_ALLOW       What to do when no pattern matches.
                        Default: "true"  (fail-open for the sample app)
    API_KEY             If set, every request must carry
                        "Authorization: Bearer <API_KEY>".
                        Default: "" (authentication disabled)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration

ALLOWED_USERNAMES: list[str] = [
    p.strip()
    for p in os.environ.get("ALLOWED_USERNAMES", "admin,service-account").split(",")
    if p.strip()
]
BLOCKED_USERNAMES: list[str] = [
    p.strip()
    for p in os.environ.get("BLOCKED_USERNAMES", "blocked,suspended").split(",")
    if p.strip()
]
DEFAULT_ALLOW: bool = os.environ.get("DEFAULT_ALLOW", "true").lower() in ("true", "1", "yes")
API_KEY: str = os.environ.get("API_KEY", "").strip()

# Models — mirror of auth_server/models/custom_authorizer.py

class CustomAuthRequest(BaseModel):
    method: str
    path: str
    original_url: str
    query_params: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    body: Optional[str] = None
    client_ip: str


class NativeAuthResult(BaseModel):
    valid: bool
    username: Optional[str] = None
    scopes: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    auth_method: Optional[str] = None
    client_id: Optional[str] = None


class CustomAuthContext(BaseModel):
    timestamp: str
    request_id: str
    gateway_version: str = "1.0.0"


class CustomAuthorizerPayload(BaseModel):
    request: CustomAuthRequest
    native_auth_result: Optional[NativeAuthResult] = None
    context: CustomAuthContext


class CustomAuthErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class CustomAuthorizerResponse(BaseModel):
    authorized: bool
    metadata: Optional[dict[str, Any]] = None
    error: Optional[CustomAuthErrorDetail] = None

# Application

app = FastAPI(
    title="Sample Custom Authorizer",
    description=(
        "Reference implementation of the MCP Gateway custom authorizer webhook. "
        "Decision logic: blocked list → allow list → DEFAULT_ALLOW."
    ),
    version="1.0.0",
)


def _verify_api_key(authorization: str | None) -> None:
    """Raise HTTP 401 when API_KEY is configured and the header is wrong."""
    if not API_KEY:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization scheme — expected Bearer")
    if authorization[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.post("/authorize", response_model=CustomAuthorizerResponse)
async def authorize(
    payload: CustomAuthorizerPayload,
    authorization: str | None = Header(default=None),
) -> CustomAuthorizerResponse:
    """Evaluate an incoming request and return an authorization decision.

    Decision order:
    1. BLOCKED list  — deny immediately if the username matches any pattern
    2. ALLOWED list  — allow immediately if the username matches any pattern
    3. DEFAULT_ALLOW — fallback when no pattern matches
    """
    _verify_api_key(authorization)

    username: str = ""
    if payload.native_auth_result and payload.native_auth_result.username:
        username = payload.native_auth_result.username

    logger.info(
        "Evaluating: user=%r path=%s method=%s request_id=%s",
        username or "(anonymous)",
        payload.request.path,
        payload.request.method,
        payload.context.request_id,
    )

    # 1 — Blocked list (deny takes precedence over everything)
    for pattern in BLOCKED_USERNAMES:
        if pattern and pattern in username:
            logger.info("DENIED — username %r matches blocked pattern %r", username, pattern)
            return CustomAuthorizerResponse(
                authorized=False,
                error=CustomAuthErrorDetail(
                    code="USER_BLOCKED",
                    message="Access denied: user matches a blocked pattern",
                    details={
                        "username": username,
                        "matched_pattern": pattern,
                        "policy": "blocked-username-list",
                    },
                ),
                metadata={
                    "policy_name": "username-blocklist",
                    "evaluated_at": _now_iso(),
                },
            )

    # 2 — Allowed list
    for pattern in ALLOWED_USERNAMES:
        if pattern and pattern in username:
            logger.info("ALLOWED — username %r matches allowed pattern %r", username, pattern)
            return CustomAuthorizerResponse(
                authorized=True,
                metadata={
                    "policy_name": "username-allowlist",
                    "matched_pattern": pattern,
                    "evaluated_at": _now_iso(),
                },
            )

    # 3 — Default decision
    decision = "ALLOWED" if DEFAULT_ALLOW else "DENIED"
    logger.info(
        "%s — username %r matched no pattern; default_allow=%s",
        decision,
        username,
        DEFAULT_ALLOW,
    )

    if DEFAULT_ALLOW:
        return CustomAuthorizerResponse(
            authorized=True,
            metadata={
                "policy_name": "default-allow",
                "evaluated_at": _now_iso(),
            },
        )

    return CustomAuthorizerResponse(
        authorized=False,
        error=CustomAuthErrorDetail(
            code="DEFAULT_DENY",
            message="Access denied: no matching allow pattern and default policy is deny",
            details={"username": username, "policy": "default-deny"},
        ),
        metadata={
            "policy_name": "default-deny",
            "evaluated_at": _now_iso(),
        },
    )


@app.get("/health")
async def health() -> dict:
    """Health check that also surfaces the current configuration."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "config": {
            "allowed_usernames": ALLOWED_USERNAMES,
            "blocked_usernames": BLOCKED_USERNAMES,
            "default_allow": DEFAULT_ALLOW,
            "api_key_configured": bool(API_KEY),
        },
    }
