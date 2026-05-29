#!/usr/bin/env python3
"""
Cedar Policy-based Custom Authorizer for MCP Gateway Registry.

Implements the custom authorizer webhook contract using a Cedar policy engine.
Policies are loaded from `policies.cedar` at startup and hot-reloaded on change.

The evaluator supports a practical Cedar subset:
  - permit / forbid rules
  - principal.groups.contains(), principal.scopes.contains()
  - action == Action::"METHOD"
  - resource.path.like("pattern")  (fnmatch-style globs)
  - when { } clauses with &&, ||, ! operators

Usage:
    pip install fastapi uvicorn pydantic watchfiles
    uvicorn app:app --host 0.0.0.0 --port 8090

Configuration (environment variables):
    POLICIES_FILE   Path to the Cedar policy file. Default: policies.cedar
    API_KEY         If set, requests must carry Authorization: Bearer <API_KEY>.
"""

import fnmatch
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
# Webhook contract models (mirror of auth_server/models/custom_authorizer.py)
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
# Cedar policy parser
# ─────────────────────────────────────────────────────────────────────────────

_RULE_RE = re.compile(
    r"(permit|forbid)\s*\(\s*principal[^,)]*,\s*action[^,)]*,\s*resource[^)]*\)"
    r"(?:\s*when\s*\{([^}]*)\})?",
    re.DOTALL,
)
_COMMENT_RE = re.compile(r"//[^\n]*")


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text)


def _parse_action_condition(action_clause: str) -> str | None:
    """Extract the HTTP method from an action clause like `action == Action::"GET"`."""
    m = re.search(r'action\s*==\s*Action::"(\w+)"', action_clause)
    return m.group(1).upper() if m else None


def _parse_rules(policy_text: str) -> list[dict]:
    """Parse Cedar policies into a list of rule dicts."""
    clean = _strip_comments(policy_text)
    rules: list[dict] = []
    for m in _RULE_RE.finditer(clean):
        effect = m.group(1)          # "permit" or "forbid"
        full_head = m.group(0)
        when_body = (m.group(2) or "").strip()
        action_clause = re.search(r"action[^,)]*", full_head)
        action_method = _parse_action_condition(action_clause.group(0)) if action_clause else None
        rules.append({"effect": effect, "action_method": action_method, "when": when_body})
    return rules


# ─────────────────────────────────────────────────────────────────────────────
# Cedar when-clause evaluator
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_condition(expr: str, ctx: dict) -> bool:
    """
    Evaluate a Cedar when-clause expression against a context dict.

    Supported patterns:
      principal.groups.contains("name")
      principal.scopes.contains("name")
      resource.path.like("glob/*")
      action == Action::"METHOD"  (always true here — pre-filtered by action_method)
      !<expr>
      <expr> && <expr>
      <expr> || <expr>
    """
    expr = expr.strip()
    if not expr:
        return True

    # Strip outer parentheses
    while expr.startswith("(") and expr.endswith(")"):
        inner = expr[1:-1]
        if _balanced(inner):
            expr = inner.strip()
        else:
            break

    # Logical NOT
    if expr.startswith("!"):
        return not _evaluate_condition(expr[1:], ctx)

    # Split on && / || respecting parenthesis depth
    for op, split_fn in [("&&", all), ("||", any)]:
        parts = _split_logical(expr, op)
        if len(parts) > 1:
            return split_fn(_evaluate_condition(p, ctx) for p in parts)

    # Leaf expressions
    principal = ctx.get("principal", {})
    resource = ctx.get("resource", {})
    action = ctx.get("action", "")

    # principal.groups.contains("value")
    m = re.match(r'principal\.groups\.contains\("([^"]+)"\)', expr)
    if m:
        return m.group(1) in principal.get("groups", [])

    # principal.scopes.contains("value")
    m = re.match(r'principal\.scopes\.contains\("([^"]+)"\)', expr)
    if m:
        return m.group(1) in principal.get("scopes", [])

    # resource.path.like("glob")
    m = re.match(r'resource\.path\.like\("([^"]+)"\)', expr)
    if m:
        return fnmatch.fnmatch(resource.get("path", ""), m.group(1).replace("*", "**").replace("**", "*"))

    # action == Action::"METHOD"
    m = re.match(r'action\s*==\s*Action::"(\w+)"', expr)
    if m:
        return m.group(1).upper() == action.upper()

    logger.warning("Unrecognised Cedar expression (treating as false): %r", expr)
    return False


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _split_logical(expr: str, op: str) -> list[str]:
    """Split expr on `op` (&&  or ||) respecting parenthesis depth."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(expr):
        if expr[i] == "(":
            depth += 1
            current.append(expr[i])
        elif expr[i] == ")":
            depth -= 1
            current.append(expr[i])
        elif depth == 0 and expr[i : i + len(op)] == op:
            parts.append("".join(current).strip())
            current = []
            i += len(op)
            continue
        else:
            current.append(expr[i])
        i += 1
    parts.append("".join(current).strip())
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Policy engine
# ─────────────────────────────────────────────────────────────────────────────

class CedarPolicyEngine:
    """Loads Cedar policies from a file and evaluates is_authorized requests."""

    def __init__(self, policy_file: str) -> None:
        self._policy_file = Path(policy_file)
        self._rules: list[dict] = []
        self._mtime: float = 0.0
        self._lock = threading.Lock()
        self._reload()

    def _reload(self) -> None:
        try:
            mtime = self._policy_file.stat().st_mtime
            if mtime == self._mtime:
                return
            text = self._policy_file.read_text()
            rules = _parse_rules(text)
            with self._lock:
                self._rules = rules
                self._mtime = mtime
            logger.info("Loaded %d Cedar rules from %s", len(rules), self._policy_file)
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
        Return (authorized, matched_rule).

        Cedar semantics: default-deny; a request is authorized only when at
        least one permit fires AND no forbid fires.
        """
        self._reload()
        ctx = {"principal": principal, "action": action, "resource": resource}
        permit_matched: str | None = None
        forbid_matched: str | None = None

        with self._lock:
            rules = list(self._rules)

        for rule in rules:
            # Action pre-filter
            if rule["action_method"] and rule["action_method"] != action.upper():
                continue

            # When-clause evaluation
            if not _evaluate_condition(rule["when"], ctx):
                continue

            if rule["effect"] == "permit" and permit_matched is None:
                permit_matched = f"permit(action={rule['action_method'] or '*'}, when={rule['when'][:60]})"
            elif rule["effect"] == "forbid" and forbid_matched is None:
                forbid_matched = f"forbid(action={rule['action_method'] or '*'}, when={rule['when'][:60]})"

        if forbid_matched:
            return False, forbid_matched
        if permit_matched:
            return True, permit_matched
        return False, "no-matching-permit (default-deny)"


_engine = CedarPolicyEngine(POLICIES_FILE)

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cedar Policy Authorizer",
    description="MCP Gateway custom authorizer backed by Cedar policies.",
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
    # or from request headers (CUSTOM mode, where native_auth_result is None).
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
        "method": action,
    }

    logger.info(
        "Evaluating: user=%r groups=%r action=%s path=%s request_id=%s",
        principal["username"] or "(anonymous)",
        principal["groups"],
        action,
        resource["path"],
        payload.context.request_id,
    )

    authorized, matched_rule = _engine.is_authorized(principal, action, resource)
    decision_label = "AUTHORIZED" if authorized else "DENIED"
    logger.info("%s — rule: %s", decision_label, matched_rule)

    if authorized:
        return CustomAuthorizerResponse(
            authorized=True,
            metadata={
                "engine": "cedar-subset",
                "matched_rule": matched_rule,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "policies_file": POLICIES_FILE,
            },
        )

    return CustomAuthorizerResponse(
        authorized=False,
        error=CustomAuthErrorDetail(
            code="CEDAR_DENY",
            message=f"Access denied by Cedar policy: {matched_rule}",
            details={
                "username": principal["username"],
                "groups": principal["groups"],
                "action": action,
                "path": resource["path"],
            },
        ),
        metadata={
            "engine": "cedar-subset",
            "matched_rule": matched_rule,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "version": "1.0.0",
        "policies_file": POLICIES_FILE,
        "rules_loaded": len(_engine._rules),
    }
