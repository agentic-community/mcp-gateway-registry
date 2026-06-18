#!/usr/bin/env python3
"""
Cedar Policy-based Custom Authorizer for MCP Gateway Registry.

Implements the custom authorizer webhook contract using the **cedarpy** library,

Policies are loaded from a `.cedar` file at startup and hot-reloaded
when the file changes on disk.

Usage:
    pip install fastapi uvicorn pydantic cedarpy
    uvicorn app:app --host 0.0.0.0 --port 8090

Configuration (environment variables):
    POLICIES_FILE   Path to the Cedar policy file. Default: policies.cedar
    API_KEY         If set, requests must carry Authorization: Bearer <API_KEY>.
"""

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cedarpy import AuthzResult, Decision, is_authorized
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

POLICIES_FILE: str = os.environ.get("POLICIES_FILE", "policies.cedar")
API_KEY: str = os.environ.get("API_KEY", "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# Webhook contract models (mirror of auth_server/authorizer_models/custom_authorizer.py)
# ─────────────────────────────────────────────────────────────────────────────


class CustomAuthRequest(BaseModel):
    method: str
    path: str
    original_url: str
    query_params: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = None
    client_ip: str


class NativeAuthResult(BaseModel):
    valid: bool
    username: str | None = None
    scopes: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    auth_method: str | None = None
    client_id: str | None = None


class CustomAuthContext(BaseModel):
    timestamp: str
    request_id: str
    gateway_version: str = "1.0.0"


class CustomAuthorizerPayload(BaseModel):
    request: CustomAuthRequest
    native_auth_result: NativeAuthResult | None = None
    context: CustomAuthContext


class CustomAuthErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class CustomAuthorizerResponse(BaseModel):
    authorized: bool
    metadata: dict[str, Any] | None = None
    error: CustomAuthErrorDetail | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Cedar policy engine (backed by cedarpy / Rust Cedar)
# ─────────────────────────────────────────────────────────────────────────────


class CedarPolicyEngine:
    """Loads Cedar policies from a file and evaluates authorization via cedarpy."""

    def __init__(self, policy_file: str) -> None:
        self._policy_file = Path(policy_file)
        self._policies: str = ""
        self._mtime: float = 0.0
        self._lock = threading.Lock()
        self._reload()

    def _reload(self) -> None:
        """Reload policies from disk if the file has changed."""
        try:
            mtime = self._policy_file.stat().st_mtime
            if mtime == self._mtime:
                return
            text = self._policy_file.read_text()
            with self._lock:
                self._policies = text
                self._mtime = mtime
            logger.info("Loaded Cedar policies from %s (%d bytes)", self._policy_file, len(text))
        except FileNotFoundError:
            logger.error("Policy file not found: %s", self._policy_file)
        except Exception as e:
            logger.error("Failed to reload policies: %s", e)

    def is_authorized(
        self,
        principal: dict,
        action: str,
        resource: dict,
    ) -> tuple[bool, str]:
        """
        Evaluate a request against loaded Cedar policies via cedarpy.

        Returns (authorized, matched_info).
        """
        self._reload()

        with self._lock:
            policies = self._policies

        if not policies:
            return False, "no-policies-loaded"

        # Build the principal identifier — use username or "anonymous"
        username = principal.get("username") or "anonymous"
        groups: list[str] = principal.get("groups", [])
        scopes: list[str] = principal.get("scopes", [])
        path: str = resource.get("path", "/")

        # Cedar entities: typed objects with attributes
        entities = [
            {
                "uid": {"__entity": {"type": "Principal", "id": username}},
                "attrs": {
                    "groups": groups,
                    "scopes": scopes,
                    "username": username,
                },
                "parents": [],
            },
            {
                "uid": {"__entity": {"type": "Resource", "id": path}},
                "attrs": {
                    "path": path,
                },
                "parents": [],
            },
        ]

        # Cedar request: references entities by type::"id"
        cedar_request = {
            "principal": f'Principal::"{username}"',
            "action": f'Action::"{action}"',
            "resource": f'Resource::"{path}"',
            "context": {},
        }

        try:
            result: AuthzResult = is_authorized(cedar_request, policies, entities)

            if result.allowed:
                reasons = list(result.diagnostics.reasons)
                reason_str = ", ".join(reasons) if reasons else "permit-matched"
                return True, reason_str

            # Denied: check for errors vs normal deny
            errors = list(result.diagnostics.errors)
            if errors:
                error_str = "; ".join(errors)
                logger.warning("Cedar evaluation errors: %s", error_str)
                return False, f"evaluation-error: {error_str}"

            return False, "no-matching-permit (default-deny)"

        except Exception as e:
            logger.error("Cedar evaluation failed: %s", e, exc_info=True)
            return False, f"engine-error: {e}"


_engine = CedarPolicyEngine(POLICIES_FILE)

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cedar Policy Authorizer",
    description="MCP Gateway custom authorizer backed by the Cedar Policy engine (cedarpy).",
    version="1.0.0",
)


def _verify_api_key(authorization: str | None) -> None:
    if not API_KEY:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization scheme — expected Bearer")
    if authorization[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/authorize", response_model=CustomAuthorizerResponse)
async def authorize(
    payload: CustomAuthorizerPayload,
    authorization: str | None = Header(default=None),
) -> CustomAuthorizerResponse:
    """Evaluate Cedar policies and return an authorization decision."""
    _verify_api_key(authorization)

    # Build evaluation context from the native auth result (available in BOTH mode)
    # or default to anonymous (CUSTOM mode, where native_auth_result is None).
    native = payload.native_auth_result
    principal = {
        "username": (native.username or "") if native else "",
        "groups": (native.groups or []) if native else [],
        "scopes": (native.scopes or []) if native else [],
        "auth_method": (native.auth_method or "") if native else "",
        "client_id": (native.client_id or "") if native else "",
    }
    action = payload.request.method.upper()
    resource = {
        "path": payload.request.path,
    }

    logger.info(
        "Evaluating: user=%r groups=%r action=%s path=%s request_id=%s",
        principal["username"] or "(anonymous)",
        principal["groups"],
        action,
        resource["path"],
        payload.context.request_id,
    )

    authorized, matched_info = _engine.is_authorized(principal, action, resource)
    decision_label = "AUTHORIZED" if authorized else "DENIED"
    logger.info("%s — %s", decision_label, matched_info)

    if authorized:
        return CustomAuthorizerResponse(
            authorized=True,
            metadata={
                "engine": "cedarpy",
                "matched_rule": matched_info,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "policies_file": POLICIES_FILE,
            },
        )

    return CustomAuthorizerResponse(
        authorized=False,
        error=CustomAuthErrorDetail(
            code="CEDAR_DENY",
            message=f"Access denied by Cedar policy: {matched_info}",
            details={
                "username": principal["username"],
                "groups": principal["groups"],
                "action": action,
                "path": resource["path"],
            },
        ),
        metadata={
            "engine": "cedarpy",
            "matched_rule": matched_info,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "version": "1.0.0",
        "engine": "cedarpy",
        "policies_file": POLICIES_FILE,
        "policies_loaded": bool(_engine._policies),
    }
