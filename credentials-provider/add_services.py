#!/usr/bin/env python3
"""
Add Services to MCP Configuration

This script scans `registry/servers/*.json` and generates MCP client configuration files:
- `.oauth-tokens/vscode_mcp.json`
- `.oauth-tokens/mcp.json`

The generated configurations contain ONLY gateway ingress credentials. Upstream credentials
and upstream authentication are managed by the gateway and must not be included in client configs.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from urllib.parse import (
    urlparse,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _load_env_file() -> None:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    env_file = project_root / ".env"

    if not env_file.exists():
        return

    try:
        with open(env_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ[key] = value.strip().strip('"').strip("'")
    except Exception as exc:
        logger.warning(f"Failed to load .env file: {exc}")


def _load_json_file(
    file_path: Path,
) -> Optional[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error(f"Failed to load {file_path}: {exc}")
        return None


def _save_json_file(
    file_path: Path,
    data: Dict[str, Any],
    description: str,
) -> None:
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        os.chmod(file_path, 0o600)
        logger.info(f"Updated {description}: {file_path}")
    except Exception as exc:
        logger.error(f"Failed to save {description} to {file_path}: {exc}")


def _get_registry_servers_dir() -> Path:
    script_dir = Path(__file__).parent
    registry_dir = script_dir.parent / "registry" / "servers"

    if not registry_dir.exists():
        raise FileNotFoundError(f"Registry servers directory not found: {registry_dir}")

    return registry_dir


def _get_oauth_tokens_dir() -> Path:
    script_dir = Path(__file__).parent
    tokens_dir = script_dir.parent / ".oauth-tokens"

    if not tokens_dir.exists():
        tokens_dir.mkdir(mode=0o700, parents=True)
        logger.info(f"Created oauth tokens directory: {tokens_dir}")

    return tokens_dir


def _determine_transport_type(
    supported_transports: Optional[List[str]],
) -> str:
    if not supported_transports:
        return "streamable-http"

    if "streamable-http" in supported_transports:
        return "streamable-http"

    if "sse" in supported_transports:
        return "sse"

    return "streamable-http"


def _public_endpoint_suffix_for_transport(
    transport_type: str,
) -> str:
    if transport_type == "sse":
        return "sse"
    return "mcp"


def _server_exposes_suffix_via_proxy_pass_url(
    proxy_pass_url: str,
    suffix: str,
) -> bool:
    try:
        parsed = urlparse(proxy_pass_url)
        upstream_path = (parsed.path or "").rstrip("/")
        return upstream_path.endswith(f"/{suffix}")
    except Exception:
        return False


def _build_public_service_url(
    registry_url: str,
    service_path: str,
    proxy_pass_url: str,
    transport_type: str,
) -> str:
    base = registry_url.rstrip("/")
    path = service_path if service_path.startswith("/") else f"/{service_path}"

    suffix = _public_endpoint_suffix_for_transport(transport_type)

    if path.endswith("/"):
        return f"{base}{path.rstrip('/')}/{suffix}"

    if _server_exposes_suffix_via_proxy_pass_url(proxy_pass_url, suffix):
        return f"{base}{path}"

    return f"{base}{path}/{suffix}"


def _scan_services() -> List[Dict[str, Any]]:
    registry_dir = _get_registry_servers_dir()
    services: List[Dict[str, Any]] = []

    logger.info(f"Scanning registry servers directory: {registry_dir}")

    for json_file in registry_dir.glob("*.json"):
        if json_file.name == "server_state.json":
            continue

        server_config = _load_json_file(json_file)
        if not server_config:
            continue

        path = server_config.get("path", "")
        proxy_pass_url = server_config.get("proxy_pass_url", "")
        if not path or not proxy_pass_url:
            logger.debug(f"Skipping {json_file.name}: missing path or proxy_pass_url")
            continue

        services.append(
            {
                "server_name": server_config.get("server_name", "Unknown"),
                "path": path,
                "proxy_pass_url": proxy_pass_url,
                "supported_transports": server_config.get("supported_transports"),
                "auth_type": server_config.get("auth_type"),
                "description": server_config.get("description", ""),
                "file_name": json_file.name,
            }
        )

    return services


def _get_ingress_headers() -> Optional[Dict[str, str]]:
    tokens_dir = _get_oauth_tokens_dir()
    ingress_file = tokens_dir / "ingress.json"

    auth_provider = os.environ.get("AUTH_PROVIDER", "").lower()

    if auth_provider == "keycloak":
        agent_token_file = tokens_dir / "agent-ai-coding-assistant-m2m-token.json"
        if agent_token_file.exists():
            agent_data = _load_json_file(agent_token_file)
            if agent_data and agent_data.get("access_token"):
                return {
                    "X-Authorization": f"Bearer {agent_data.get('access_token', '')}",
                }

    if not ingress_file.exists():
        return None

    ingress_data = _load_json_file(ingress_file)
    if not ingress_data:
        return None

    return {
        "X-Authorization": f"Bearer {ingress_data.get('access_token', '')}",
        "X-User-Pool-Id": ingress_data.get("user_pool_id", ""),
        "X-Client-Id": ingress_data.get("client_id", ""),
        "X-Region": ingress_data.get("region", "us-east-1"),
    }


def _update_vscode_config(
    services: List[Dict[str, Any]],
    ingress_headers: Optional[Dict[str, str]],
) -> None:
    tokens_dir = _get_oauth_tokens_dir()
    vscode_file = tokens_dir / "vscode_mcp.json"

    config = _load_json_file(vscode_file) or {"mcp": {"servers": {}}}
    config.setdefault("mcp", {}).setdefault("servers", {})

    registry_url = os.environ.get("REGISTRY_URL", "https://mcpgateway.ddns.net")

    for service in services:
        server_key = service["path"].strip("/")
        if not server_key:
            continue

        transport_type = _determine_transport_type(service.get("supported_transports"))
        service_url = _build_public_service_url(
            registry_url=registry_url,
            service_path=service["path"],
            proxy_pass_url=service.get("proxy_pass_url", ""),
            transport_type=transport_type,
        )

        server_config: Dict[str, Any] = {"url": service_url}
        if ingress_headers:
            server_config["headers"] = ingress_headers.copy()

        config["mcp"]["servers"][server_key] = server_config

    _save_json_file(vscode_file, config, "VS Code MCP configuration")


def _update_roocode_config(
    services: List[Dict[str, Any]],
    ingress_headers: Optional[Dict[str, str]],
) -> None:
    tokens_dir = _get_oauth_tokens_dir()
    roocode_file = tokens_dir / "mcp.json"

    config = _load_json_file(roocode_file) or {"mcpServers": {}}
    config.setdefault("mcpServers", {})

    registry_url = os.environ.get("REGISTRY_URL", "https://mcpgateway.ddns.net")

    for service in services:
        server_key = service["path"].strip("/")
        if not server_key:
            continue

        transport_type = _determine_transport_type(service.get("supported_transports"))
        service_url = _build_public_service_url(
            registry_url=registry_url,
            service_path=service["path"],
            proxy_pass_url=service.get("proxy_pass_url", ""),
            transport_type=transport_type,
        )

        server_config: Dict[str, Any] = {
            "type": transport_type,
            "url": service_url,
            "disabled": False,
            "alwaysAllow": [],
        }
        if ingress_headers:
            server_config["headers"] = ingress_headers.copy()

        config["mcpServers"][server_key] = server_config

    _save_json_file(roocode_file, config, "Roo Code MCP configuration")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MCP configs for all registry services (gateway-only auth)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    return parser.parse_args()


def main() -> None:
    try:
        _load_env_file()

        args = _parse_arguments()
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        services = _scan_services()
        if not services:
            logger.info("No services found under registry/servers/")
            return

        ingress_headers = _get_ingress_headers()
        if not ingress_headers:
            logger.warning("No gateway ingress token found; configs will be written without auth headers")

        _update_vscode_config(services, ingress_headers)
        _update_roocode_config(services, ingress_headers)

    except Exception as exc:
        logger.error(f"Failed to update MCP configurations: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

