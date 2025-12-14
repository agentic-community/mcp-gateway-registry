#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, TextIO

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL: str = "http://localhost:8888"

ENV_BASE_URL: str = "ENFORCEAI_AUTH_SERVER_URL"
ENV_AUTHORIZATION: str = "ENFORCEAI_AUTHORIZATION"
ENV_X_AGENT_ID: str = "ENFORCEAI_X_AGENT_ID"
ENV_X_GATEWAY_TOKEN: str = "ENFORCEAI_X_GATEWAY_TOKEN"
ENV_X_API_KEY: str = "ENFORCEAI_X_API_KEY"


class CLIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        exit_code: int = 2,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _resolve_config_value(
    *,
    cli_value: Optional[str],
    env: Mapping[str, str],
    env_key: str,
) -> Optional[str]:
    if cli_value is not None and cli_value.strip():
        return cli_value.strip()

    raw_env = env.get(env_key)
    if raw_env is None:
        return None

    stripped = raw_env.strip()
    return stripped or None


def _parse_json_object(
    raw: Optional[str],
    *,
    label: str,
) -> Optional[dict[str, object]]:
    if raw is None:
        return None

    stripped = raw.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise CLIError(f"{label} must be valid JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise CLIError(f"{label} must be a JSON object")

    return parsed


def _parse_datetime_to_iso(
    raw: Optional[str],
    *,
    label: str,
) -> Optional[str]:
    if raw is None:
        return None

    stripped = raw.strip()
    if not stripped:
        return None

    normalized = stripped.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise CLIError(f"{label} must be an ISO-8601 datetime") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc).replace(microsecond=0)
    return parsed.isoformat().replace("+00:00", "Z")


def _coerce_bearer_value(
    raw: str,
) -> str:
    stripped = raw.strip()
    if not stripped:
        raise CLIError("authorization must be non-empty")

    if stripped.lower().startswith("bearer "):
        return stripped

    return f"Bearer {stripped}"


def _build_headers(
    *,
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> dict[str, str]:
    authorization = _resolve_config_value(
        cli_value=args.authorization,
        env=env,
        env_key=ENV_AUTHORIZATION,
    )
    x_gateway_token = _resolve_config_value(
        cli_value=args.x_gateway_token,
        env=env,
        env_key=ENV_X_GATEWAY_TOKEN,
    )
    x_api_key = _resolve_config_value(
        cli_value=args.x_api_key,
        env=env,
        env_key=ENV_X_API_KEY,
    )
    x_agent_id = _resolve_config_value(
        cli_value=args.x_agent_id,
        env=env,
        env_key=ENV_X_AGENT_ID,
    )

    provided = [
        value
        for value in (
            authorization,
            x_gateway_token,
            x_api_key,
        )
        if value is not None
    ]
    if not provided:
        raise CLIError(
            "Missing credentials: provide one of "
            "--authorization / ENFORCEAI_AUTHORIZATION, "
            "--x-gateway-token / ENFORCEAI_X_GATEWAY_TOKEN, "
            "or --x-api-key / ENFORCEAI_X_API_KEY"
        )
    if len(provided) != 1:
        raise CLIError(
            "Multiple credentials provided: choose exactly one of "
            "--authorization, --x-gateway-token, or --x-api-key"
        )

    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": "enforceai-cli/0.1",
    }

    if authorization is not None:
        headers["Authorization"] = _coerce_bearer_value(authorization)
    elif x_gateway_token is not None:
        headers["X-Gateway-Token"] = x_gateway_token
    elif x_api_key is not None:
        headers["X-API-Key"] = x_api_key

    if x_agent_id is not None:
        headers["X-Agent-Id"] = x_agent_id

    return headers


def _print_json(
    payload: object,
    *,
    pretty: bool,
    stream: TextIO,
) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
        stream.write("\n")
        return

    stream.write(json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str))
    stream.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enforceai_cli.py",
        description="EnforceAI management CLI (Stage 6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List agents (OIDC)
  uv run python cli/enforceai_cli.py --base-url http://localhost:8888 \\
    --authorization "$ENFORCEAI_AUTHORIZATION" --x-agent-id "$ENFORCEAI_X_AGENT_ID" \\
    agents list

  # Create agent
  uv run python cli/enforceai_cli.py --authorization "$ENFORCEAI_AUTHORIZATION" --x-agent-id "$ENFORCEAI_X_AGENT_ID" \\
    agents create --scope scope-mgmt --alias my-agent

  # Mint gateway token for an agent
  uv run python cli/enforceai_cli.py --authorization "$ENFORCEAI_AUTHORIZATION" --x-agent-id "$ENFORCEAI_X_AGENT_ID" \\
    tokens mint <agent_id> --scope scope-mgmt --ttl-seconds 3600
""".strip(),
    )

    parser.add_argument(
        "--base-url",
        help=f"Auth server base URL (default: {DEFAULT_BASE_URL}; env: {ENV_BASE_URL})",
    )
    parser.add_argument(
        "--authorization",
        help=f"Authorization header value (env: {ENV_AUTHORIZATION}); accepts raw token or 'Bearer <token>'",
    )
    parser.add_argument(
        "--x-agent-id",
        help=f"Agent binding for OIDC (env: {ENV_X_AGENT_ID})",
    )
    parser.add_argument(
        "--x-gateway-token",
        help=f"Gateway token header (env: {ENV_X_GATEWAY_TOKEN})",
    )
    parser.add_argument(
        "--x-api-key",
        help=f"API key header value (env: {ENV_X_API_KEY})",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (never logs secrets)",
    )

    subparsers = parser.add_subparsers(
        dest="resource",
        required=True,
    )

    agents = subparsers.add_parser("agents", help="Agent lifecycle operations")
    agents_sub = agents.add_subparsers(dest="action", required=True)

    agents_sub.add_parser("list", help="List your agents")

    agents_create = agents_sub.add_parser("create", help="Create an agent")
    agents_create.add_argument(
        "--scope",
        action="append",
        required=True,
        help="Scope to grant (repeatable)",
    )
    agents_create.add_argument(
        "--allowed-tool",
        action="append",
        help="Allowed tool name (repeatable; optional extra restriction)",
    )
    agents_create.add_argument("--alias", help="Optional alias")
    agents_create.add_argument(
        "--metadata-json",
        help="Optional JSON object for agent metadata",
    )

    agents_get = agents_sub.add_parser("get", help="Get an agent by id")
    agents_get.add_argument("agent_id")

    agents_update = agents_sub.add_parser("update", help="Update an agent by id")
    agents_update.add_argument("agent_id")
    agents_update.add_argument(
        "--scope",
        action="append",
        help="Replace scopes with this list (repeatable)",
    )
    agents_update.add_argument(
        "--allowed-tool",
        action="append",
        help="Replace allowed tools with this list (repeatable)",
    )
    agents_update.add_argument("--alias", help="Replace alias")
    agents_update.add_argument(
        "--metadata-json",
        help="Replace metadata with this JSON object",
    )

    agents_revoke = agents_sub.add_parser("revoke", help="Revoke an agent by id")
    agents_revoke.add_argument("agent_id")

    api_keys = subparsers.add_parser("api-keys", help="API key operations")
    api_keys_sub = api_keys.add_subparsers(dest="action", required=True)

    api_keys_create = api_keys_sub.add_parser("create", help="Create an API key for an agent")
    api_keys_create.add_argument("agent_id")
    api_keys_create.add_argument(
        "--scope",
        action="append",
        help="Optional scope restriction (repeatable)",
    )
    api_keys_create.add_argument(
        "--expires-at",
        help="Optional expiry (ISO-8601, e.g. 2025-01-01T00:00:00Z)",
    )

    api_keys_list = api_keys_sub.add_parser("list", help="List API keys for an agent")
    api_keys_list.add_argument("agent_id")

    api_keys_revoke = api_keys_sub.add_parser("revoke", help="Revoke an API key by key_id")
    api_keys_revoke.add_argument("key_id")

    tokens = subparsers.add_parser("tokens", help="Gateway token operations")
    tokens_sub = tokens.add_subparsers(dest="action", required=True)

    tokens_mint = tokens_sub.add_parser("mint", help="Mint a gateway token for an agent")
    tokens_mint.add_argument("agent_id")
    tokens_mint.add_argument(
        "--scope",
        action="append",
        required=True,
        help="Token scope (repeatable; must be subset of agent scopes)",
    )
    tokens_mint.add_argument(
        "--ttl-seconds",
        type=int,
        help="TTL in seconds",
    )
    tokens_mint.add_argument(
        "--expires-at",
        help="Explicit expiry (ISO-8601, mutually exclusive with --ttl-seconds)",
    )

    tokens_revoke = tokens_sub.add_parser("revoke", help="Revoke a gateway token (by token or jti)")
    tokens_revoke.add_argument(
        "--gateway-token",
        help="Revoke by supplying the full gateway token string",
    )
    tokens_revoke.add_argument(
        "--agent-id",
        help="Revoke by supplying agent_id + jti",
    )
    tokens_revoke.add_argument(
        "--jti",
        help="Revoke by supplying agent_id + jti",
    )
    tokens_revoke.add_argument(
        "--reason",
        help="Optional revocation reason",
    )

    tokens_revoke_all = tokens_sub.add_parser("revoke-all", help="Revoke all tokens for an agent")
    tokens_revoke_all.add_argument("agent_id")

    return parser


def _resolve_base_url(
    *,
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> str:
    resolved = _resolve_config_value(
        cli_value=args.base_url,
        env=env,
        env_key=ENV_BASE_URL,
    )
    return resolved or DEFAULT_BASE_URL


def _validate_tokens_revoke_args(
    *,
    args: argparse.Namespace,
) -> None:
    gateway_token = getattr(args, "gateway_token", None)
    agent_id = getattr(args, "agent_id", None)
    jti = getattr(args, "jti", None)

    if gateway_token is not None and gateway_token.strip():
        if (agent_id is not None and agent_id.strip()) or (jti is not None and jti.strip()):
            raise CLIError("tokens revoke: provide either --gateway-token or (--agent-id and --jti)")
        return

    if agent_id is None or not agent_id.strip() or jti is None or not jti.strip():
        raise CLIError("tokens revoke: provide either --gateway-token or both --agent-id and --jti")


async def _request_json(
    *,
    base_url: str,
    method: str,
    path: str,
    headers: Mapping[str, str],
    json_body: Optional[object] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> object:
    async with httpx.AsyncClient(
        base_url=base_url,
        headers=dict(headers),
        timeout=30.0,
        transport=transport,
    ) as client:
        response = await client.request(
            method,
            path,
            json=json_body,
        )

    content_type = response.headers.get("content-type", "")
    if response.status_code >= 400:
        detail: str = response.text
        if "application/json" in content_type:
            try:
                parsed = response.json()
                if isinstance(parsed, dict) and "detail" in parsed:
                    detail = str(parsed.get("detail"))
                else:
                    detail = json.dumps(parsed, separators=(",", ":"), sort_keys=True, default=str)
            except Exception:  # noqa: BLE001
                detail = response.text

        raise CLIError(f"HTTP {response.status_code}: {detail}", exit_code=1)

    if "application/json" in content_type:
        return response.json()

    if not response.text.strip():
        return {}

    raise CLIError("Server returned non-JSON response", exit_code=1)


@dataclass(frozen=True)
class _RunResult:
    payload: object


async def _run_command(
    *,
    args: argparse.Namespace,
    env: Mapping[str, str],
    transport: Optional[httpx.AsyncBaseTransport],
) -> _RunResult:
    base_url = _resolve_base_url(
        args=args,
        env=env,
    )
    headers = _build_headers(
        args=args,
        env=env,
    )

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    resource = args.resource
    action = args.action

    if resource == "agents":
        if action == "list":
            payload = await _request_json(
                base_url=base_url,
                method="GET",
                path="/enforceai/agents",
                headers=headers,
                transport=transport,
            )
            return _RunResult(payload=payload)

        if action == "create":
            metadata = _parse_json_object(
                getattr(args, "metadata_json", None),
                label="--metadata-json",
            )
            payload = await _request_json(
                base_url=base_url,
                method="POST",
                path="/enforceai/agents",
                headers=headers,
                json_body={
                    "scopes": list(getattr(args, "scope", []) or []),
                    "allowed_tools": getattr(args, "allowed_tool", None),
                    "alias": getattr(args, "alias", None),
                    "metadata": metadata,
                },
                transport=transport,
            )
            return _RunResult(payload=payload)

        if action == "get":
            agent_id = getattr(args, "agent_id")
            payload = await _request_json(
                base_url=base_url,
                method="GET",
                path=f"/enforceai/agents/{agent_id}",
                headers=headers,
                transport=transport,
            )
            return _RunResult(payload=payload)

        if action == "update":
            agent_id = getattr(args, "agent_id")
            metadata = _parse_json_object(
                getattr(args, "metadata_json", None),
                label="--metadata-json",
            )
            body: dict[str, object] = {}
            if getattr(args, "scope", None) is not None:
                body["scopes"] = list(getattr(args, "scope") or [])
            if getattr(args, "allowed_tool", None) is not None:
                body["allowed_tools"] = getattr(args, "allowed_tool")
            if getattr(args, "alias", None) is not None:
                body["alias"] = getattr(args, "alias")
            if getattr(args, "metadata_json", None) is not None:
                body["metadata"] = metadata

            payload = await _request_json(
                base_url=base_url,
                method="PATCH",
                path=f"/enforceai/agents/{agent_id}",
                headers=headers,
                json_body=body,
                transport=transport,
            )
            return _RunResult(payload=payload)

        if action == "revoke":
            agent_id = getattr(args, "agent_id")
            payload = await _request_json(
                base_url=base_url,
                method="POST",
                path=f"/enforceai/agents/{agent_id}/revoke",
                headers=headers,
                transport=transport,
            )
            return _RunResult(payload=payload)

    if resource == "api-keys":
        if action == "create":
            agent_id = getattr(args, "agent_id")
            expires_at = _parse_datetime_to_iso(
                getattr(args, "expires_at", None),
                label="--expires-at",
            )
            payload = await _request_json(
                base_url=base_url,
                method="POST",
                path=f"/enforceai/agents/{agent_id}/api-keys",
                headers=headers,
                json_body={
                    "scopes": getattr(args, "scope", None),
                    "expires_at": expires_at,
                },
                transport=transport,
            )
            return _RunResult(payload=payload)

        if action == "list":
            agent_id = getattr(args, "agent_id")
            payload = await _request_json(
                base_url=base_url,
                method="GET",
                path=f"/enforceai/agents/{agent_id}/api-keys",
                headers=headers,
                transport=transport,
            )
            return _RunResult(payload=payload)

        if action == "revoke":
            key_id = getattr(args, "key_id")
            payload = await _request_json(
                base_url=base_url,
                method="POST",
                path=f"/enforceai/api-keys/{key_id}/revoke",
                headers=headers,
                transport=transport,
            )
            return _RunResult(payload=payload)

    if resource == "tokens":
        if action == "mint":
            agent_id = getattr(args, "agent_id")
            expires_at = _parse_datetime_to_iso(
                getattr(args, "expires_at", None),
                label="--expires-at",
            )
            payload = await _request_json(
                base_url=base_url,
                method="POST",
                path=f"/enforceai/agents/{agent_id}/tokens/mint",
                headers=headers,
                json_body={
                    "scopes": list(getattr(args, "scope", []) or []),
                    "ttl_seconds": getattr(args, "ttl_seconds", None),
                    "expires_at": expires_at,
                },
                transport=transport,
            )
            return _RunResult(payload=payload)

        if action == "revoke":
            _validate_tokens_revoke_args(args=args)
            body: dict[str, object] = {
                "reason": getattr(args, "reason", None),
            }

            gateway_token = getattr(args, "gateway_token", None)
            if gateway_token is not None and gateway_token.strip():
                body["gateway_token"] = gateway_token.strip()
            else:
                body["agent_id"] = (getattr(args, "agent_id") or "").strip()
                body["jti"] = (getattr(args, "jti") or "").strip()

            payload = await _request_json(
                base_url=base_url,
                method="POST",
                path="/enforceai/tokens/revoke",
                headers=headers,
                json_body=body,
                transport=transport,
            )
            return _RunResult(payload=payload)

        if action == "revoke-all":
            agent_id = getattr(args, "agent_id")
            payload = await _request_json(
                base_url=base_url,
                method="POST",
                path=f"/enforceai/agents/{agent_id}/tokens/revoke-all",
                headers=headers,
                transport=transport,
            )
            return _RunResult(payload=payload)

    raise CLIError("Unsupported command", exit_code=2)


def main(
    argv: Optional[list[str]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    resolved_env = env or os.environ
    out = stdout or sys.stdout
    err = stderr or sys.stderr

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    try:
        result = asyncio.run(
            _run_command(
                args=args,
                env=resolved_env,
                transport=transport,
            )
        )
    except CLIError as exc:
        err.write(f"{exc}\n")
        return exc.exit_code

    _print_json(
        result.payload,
        pretty=bool(args.pretty),
        stream=out,
    )
    return 0


async def run_async(
    argv: list[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> tuple[int, str, str]:
    """Async entrypoint for tests (no sys.exit, no asyncio.run)."""
    resolved_env = env or os.environ
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 2
        return code, "", "Argument parsing failed"

    try:
        result = await _run_command(
            args=args,
            env=resolved_env,
            transport=transport,
        )
    except CLIError as exc:
        return exc.exit_code, "", str(exc)

    pretty = bool(args.pretty)
    output = (
        json.dumps(result.payload, indent=2, sort_keys=True, default=str)
        if pretty
        else json.dumps(result.payload, separators=(",", ":"), sort_keys=True, default=str)
    )
    return 0, f"{output}\n", ""


if __name__ == "__main__":
    raise SystemExit(main())
