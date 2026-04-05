"""Token refresher for AgentCore CUSTOM_JWT gateways.

Reads token_refresh_manifest.json (produced by ``cli.agentcore sync``),
resolves client secrets per IdP vendor, fetches OAuth2 access tokens
via standard OIDC client_credentials grant, and PATCHes them into the
MCP Gateway Registry.

Usage::

    # One-time refresh
    uv run python -m cli.agentcore.token_refresher \
        --manifest token_refresh_manifest.json \
        --registry-url https://registry.example.com \
        --token-file .token

    # Continuous mode (sidecar)
    uv run python -m cli.agentcore.token_refresher \
        --manifest token_refresh_manifest.json \
        --registry-url https://registry.example.com \
        --token-file .token \
        --loop --interval 2700
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import boto3
import requests

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OIDC_DISCOVERY_TIMEOUT: int = 10
TOKEN_REQUEST_TIMEOUT: int = 15
REGISTRY_REQUEST_TIMEOUT: int = 15
SECURITY_SCAN_TIMEOUT: int = 120

IDP_PATTERNS: dict[str, str] = {
    "cognito-idp": "cognito",
    "auth0.com": "auth0",
    "okta.com": "okta",
    "microsoftonline.com": "entra",
    "/realms/": "keycloak",
}

IDP_SECRET_ENV_VARS: dict[str, str] = {
    "auth0": "AUTH0_CLIENT_SECRET",
    "okta": "OKTA_CLIENT_SECRET",
    "entra": "ENTRA_CLIENT_SECRET",
    "keycloak": "KEYCLOAK_CLIENT_SECRET",
}

ENV_VAR_PREFIX: str = "OAUTH_CLIENT_SECRET_"


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------


def _read_manifest(
    manifest_path: str,
) -> list[dict[str, Any]]:
    """Read token refresh manifest from JSON file.

    Args:
        manifest_path: Path to the manifest JSON file.

    Returns:
        List of manifest entries.

    Raises:
        FileNotFoundError: If manifest file does not exist.
        ValueError: If manifest file contains invalid JSON.
    """
    abs_path = os.path.abspath(manifest_path)
    try:
        with open(abs_path) as f:
            entries = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Manifest file not found: {abs_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in manifest file {abs_path}: {e}")

    if not isinstance(entries, list):
        raise ValueError(f"Manifest must be a JSON array, got {type(entries).__name__}")

    logger.info(f"Read {len(entries)} entries from {manifest_path}")
    return entries


def _detect_idp_vendor(
    discovery_url: str,
) -> str:
    """Detect IdP vendor from OIDC discovery URL.

    Matches known patterns in the URL string.

    Args:
        discovery_url: OIDC discovery URL.

    Returns:
        Vendor name (cognito, auth0, okta, entra, keycloak, or unknown).
    """
    for pattern, vendor in IDP_PATTERNS.items():
        if pattern in discovery_url:
            return vendor
    return "unknown"


def _get_cognito_client_secret(
    discovery_url: str,
    client_id: str,
) -> str | None:
    """Auto-retrieve client secret from Cognito.

    Parses user_pool_id and region from the discoveryUrl,
    calls describe_user_pool_client() via boto3.

    Args:
        discovery_url: Cognito OIDC discovery URL containing pool_id and region.
        client_id: Cognito app client ID.

    Returns:
        Client secret string, or None if not available.
    """
    try:
        # Parse: https://cognito-idp.{region}.amazonaws.com/{pool_id}/...
        region = discovery_url.split("cognito-idp.")[1].split(".amazonaws")[0]
        pool_id = discovery_url.split("amazonaws.com/")[1].split("/")[0]

        client = boto3.client("cognito-idp", region_name=region)
        response = client.describe_user_pool_client(
            UserPoolId=pool_id,
            ClientId=client_id,
        )
        secret = response["UserPoolClient"].get("ClientSecret")
        if secret:
            logger.info(f"Auto-retrieved client secret from Cognito (pool: {pool_id})")
        else:
            logger.warning(f"Cognito app client {client_id} has no client secret")
        return secret
    except Exception as e:
        logger.error(f"Failed to retrieve Cognito client secret: {e}")
        return None


def _get_client_secret(
    idp_vendor: str,
    discovery_url: str,
    client_id: str,
) -> str | None:
    """Resolve client secret using this priority order:

    1. Per-client env var: OAUTH_CLIENT_SECRET_<client_id>
    2. Cognito auto-retrieval via AWS API (cognito only)
    3. Vendor env var: AUTH0_CLIENT_SECRET, OKTA_CLIENT_SECRET, etc.

    Args:
        idp_vendor: Detected IdP vendor name.
        discovery_url: OIDC discovery URL (used for Cognito parsing).
        client_id: OAuth2 client ID.

    Returns:
        Client secret string, or None if not available.
    """
    # Priority 1: per-client env var (OAUTH_CLIENT_SECRET_<client_id>)
    env_var_name = f"{ENV_VAR_PREFIX}{client_id}"
    secret = os.environ.get(env_var_name)
    if secret:
        logger.info(
            f"Using client secret from env var {env_var_name}"
        )
        return secret

    # Priority 2: Cognito auto-retrieval via AWS API
    if idp_vendor == "cognito":
        return _get_cognito_client_secret(discovery_url, client_id)

    # Priority 3: vendor-specific env var
    vendor_env_var = IDP_SECRET_ENV_VARS.get(idp_vendor)
    if not vendor_env_var:
        logger.warning(f"No env var mapping for IdP vendor: {idp_vendor}")
        return None

    secret = os.environ.get(vendor_env_var)
    if not secret:
        logger.warning(f"Env var {vendor_env_var} not set for {idp_vendor}")
    return secret


def _get_token_endpoint(
    discovery_url: str,
) -> str | None:
    """Fetch token_endpoint from OIDC discovery document.

    GETs the discoveryUrl and extracts the token_endpoint field.
    Standard OIDC -- works for all providers.

    Args:
        discovery_url: OIDC discovery URL.

    Returns:
        Token endpoint URL, or None on failure.
    """
    try:
        response = requests.get(
            discovery_url,
            timeout=OIDC_DISCOVERY_TIMEOUT,
        )
        response.raise_for_status()
        token_endpoint = response.json().get("token_endpoint")
        if not token_endpoint:
            logger.error(f"No token_endpoint in OIDC discovery: {discovery_url}")
        return token_endpoint
    except Exception as e:
        logger.error(f"OIDC discovery failed for {discovery_url}: {e}")
        return None


def _request_token(
    token_endpoint: str,
    client_id: str,
    client_secret: str,
) -> str | None:
    """Request access token via OAuth2 client_credentials grant.

    Args:
        token_endpoint: OAuth2 token endpoint URL.
        client_id: OAuth2 client ID.
        client_secret: OAuth2 client secret.

    Returns:
        Access token string, or None on failure.
    """
    try:
        response = requests.post(
            token_endpoint,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=TOKEN_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            logger.error("Token response missing access_token field")
        return token
    except Exception as e:
        logger.error(f"Token request failed: {e}")
        return None


def _update_registry_credential(
    registry_url: str,
    registry_token: str,
    server_path: str,
    auth_credential: str,
) -> bool:
    """PATCH auth_credential for a server in the registry.

    Uses the /api/servers{path}/auth-credential endpoint.

    Args:
        registry_url: Registry base URL.
        registry_token: Registry auth token (Bearer).
        server_path: Server path in the registry (e.g., /my-server).
        auth_credential: New auth credential (access token).

    Returns:
        True if update succeeded, False otherwise.
    """
    url = f"{registry_url.rstrip('/')}/api/servers{server_path}/auth-credential"
    try:
        response = requests.patch(
            url,
            headers={
                "Authorization": f"Bearer {registry_token}",
                "Content-Type": "application/json",
            },
            json={
                "auth_scheme": "bearer",
                "auth_credential": auth_credential,
            },
            timeout=REGISTRY_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        logger.info(f"Updated auth_credential for {server_path}")
        return True
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status == 500 and "text/html" in (
            e.response.headers.get("content-type", "") if e.response is not None else ""
        ):
            logger.error(
                f"Failed to update credential for {server_path}: "
                f"HTTP {status} from nginx -- registry token may be expired, "
                f"regenerate and retry"
            )
        else:
            logger.error(f"Failed to update credential for {server_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to update credential for {server_path}: {e}")
        return False


def _trigger_security_scan(
    registry_url: str,
    registry_token: str,
    server_path: str,
) -> bool:
    """Trigger a security rescan for a server after credential update.

    POSTs to /api/servers/{path}/rescan. Requires admin privileges
    on the registry token.

    Args:
        registry_url: Registry base URL.
        registry_token: Registry auth token (Bearer).
        server_path: Server path in the registry (e.g., /my-server).

    Returns:
        True if scan was triggered successfully, False otherwise.
    """
    url = f"{registry_url.rstrip('/')}/api/servers{server_path}/rescan"
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {registry_token}",
                "Content-Type": "application/json",
            },
            timeout=SECURITY_SCAN_TIMEOUT,
        )
        response.raise_for_status()
        scan_data = response.json()
        is_safe = scan_data.get("is_safe", False)
        critical = scan_data.get("critical_issues", 0)
        high = scan_data.get("high_severity", 0)

        if is_safe:
            logger.info(f"Security scan passed for {server_path}")
        else:
            logger.warning(
                f"Security scan for {server_path}: "
                f"critical={critical}, high={high}, is_safe={is_safe}"
            )
        return True
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "?"
        if status_code == 403:
            logger.warning(
                f"Security scan skipped for {server_path}: "
                f"registry token lacks admin privileges"
            )
        else:
            logger.error(
                f"Security scan failed for {server_path}: HTTP {status_code}"
            )
        return False
    except Exception as e:
        logger.error(f"Security scan failed for {server_path}: {e}")
        return False


def _load_registry_token(
    token_file: str,
) -> str:
    """Load registry auth token from JSON file.

    Supports two formats:
    - Flat: ``{"access_token": "..."}`` or ``{"token": "..."}``
    - Nested: ``{"tokens": {"access_token": "..."}}``

    Args:
        token_file: Path to the token JSON file.

    Returns:
        Token string.

    Raises:
        FileNotFoundError: If token file does not exist.
        ValueError: If token file is invalid or missing token field.
    """
    abs_path = os.path.abspath(token_file)
    try:
        with open(abs_path) as f:
            data = json.load(f)
            # Try top-level first, then nested under "tokens"
            token = data.get("access_token") or data.get("token")
            if not token:
                tokens_obj = data.get("tokens", {})
                if isinstance(tokens_obj, dict):
                    token = tokens_obj.get("access_token") or tokens_obj.get("token")
            if not token:
                raise ValueError(
                    f"No access_token or token field in token file: {abs_path}"
                )
            return token
    except FileNotFoundError:
        raise FileNotFoundError(f"Token file not found: {abs_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in token file {abs_path}: {e}")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def refresh_all(
    manifest_path: str,
    registry_url: str,
    registry_token: str,
    run_scan: bool = True,
) -> dict[str, Any]:
    """Refresh tokens for all entries in the manifest.

    For each CUSTOM_JWT gateway:
    1. Resolve client_secret (per-client env -> Cognito auto -> vendor env)
    2. GET discoveryUrl -> extract token_endpoint
    3. POST client_credentials grant -> get access_token
    4. PATCH auth_credential in the registry
    5. Trigger security rescan (if run_scan is True)

    Args:
        manifest_path: Path to token_refresh_manifest.json.
        registry_url: Registry base URL.
        registry_token: Registry auth token (Bearer).
        run_scan: If True, trigger security rescan after each credential update.

    Returns:
        Summary dict with success/failure/skipped counts and scan results.
    """
    entries = _read_manifest(manifest_path)
    start_time = time.time()

    success_count = 0
    failure_count = 0
    skipped_count = 0
    scan_success_count = 0
    scan_failure_count = 0

    for entry in entries:
        server_path = entry["server_path"]
        discovery_url = entry["discovery_url"]
        allowed_clients = entry.get("allowed_clients", [])
        idp_vendor = entry.get("idp_vendor") or _detect_idp_vendor(discovery_url)

        if not allowed_clients:
            logger.warning(f"No allowed_clients for {server_path} -- skipping")
            skipped_count += 1
            continue

        client_id = allowed_clients[0]

        # Step 1: Resolve client_secret (per-client env -> auto -> vendor env)
        client_secret = _get_client_secret(
            idp_vendor, discovery_url, client_id
        )
        if not client_secret:
            skipped_count += 1
            continue

        # Step 2: Get token_endpoint via OIDC discovery
        token_endpoint = _get_token_endpoint(discovery_url)
        if not token_endpoint:
            failure_count += 1
            continue

        # Step 3: Request token
        token = _request_token(token_endpoint, client_id, client_secret)
        if not token:
            failure_count += 1
            continue

        # Step 4: Update registry
        updated = _update_registry_credential(
            registry_url, registry_token, server_path, token
        )
        if updated:
            success_count += 1
            entry["last_refreshed"] = datetime.now(timezone.utc).isoformat()

            # Step 5: Trigger security rescan
            if run_scan:
                scanned = _trigger_security_scan(
                    registry_url, registry_token, server_path
                )
                if scanned:
                    scan_success_count += 1
                else:
                    scan_failure_count += 1
        else:
            failure_count += 1

    # Update manifest with timestamps
    with open(manifest_path, "w") as f:
        json.dump(entries, f, indent=2)

    elapsed = time.time() - start_time
    summary: dict[str, Any] = {
        "total": len(entries),
        "success": success_count,
        "failed": failure_count,
        "skipped": skipped_count,
        "elapsed_seconds": round(elapsed, 1),
    }

    if run_scan:
        summary["scans_triggered"] = scan_success_count
        summary["scans_failed"] = scan_failure_count

    logger.info(f"Token refresh complete: {json.dumps(summary)}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run token refresh."""
    parser = argparse.ArgumentParser(
        description="Refresh auth tokens for AgentCore CUSTOM_JWT gateways",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # One-time refresh
    uv run python -m cli.agentcore.token_refresher \\
        --manifest token_refresh_manifest.json \\
        --registry-url https://registry.example.com \\
        --token-file .token

    # With per-client env vars (from .env)
    OAUTH_CLIENT_SECRET_49ujl0b9ser72gnp6q1ph9v6vs=mysecret \\
        uv run python -m cli.agentcore.token_refresher \\
        --manifest token_refresh_manifest.json \\
        --registry-url https://registry.example.com \\
        --token-file .token

    # Continuous mode (run as sidecar)
    uv run python -m cli.agentcore.token_refresher \\
        --manifest token_refresh_manifest.json \\
        --registry-url https://registry.example.com \\
        --token-file .token \\
        --loop --interval 2700

Secret resolution priority (per client_id):
    1. Per-client env var: OAUTH_CLIENT_SECRET_<client_id>=<secret>
    2. Cognito auto-retrieval via AWS API (cognito only)
    3. Vendor env var:
       AUTH0_CLIENT_SECRET      Client secret for Auth0 gateways
       OKTA_CLIENT_SECRET       Client secret for Okta gateways
       ENTRA_CLIENT_SECRET      Client secret for Entra gateways
       KEYCLOAK_CLIENT_SECRET   Client secret for Keycloak gateways
""",
    )
    parser.add_argument(
        "--manifest",
        default="token_refresh_manifest.json",
        help="Path to token refresh manifest (default: token_refresh_manifest.json)",
    )
    parser.add_argument(
        "--registry-url",
        default=os.environ.get("REGISTRY_URL", "http://localhost"),
        help="Registry base URL (default: REGISTRY_URL env or http://localhost)",
    )
    parser.add_argument(
        "--token-file",
        default=os.environ.get("REGISTRY_TOKEN_FILE", ".token"),
        help="Path to registry auth token file (default: REGISTRY_TOKEN_FILE env or .token)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously (for sidecar deployment)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=2700,
        help="Refresh interval in seconds (default: 2700 = 45 min)",
    )
    parser.add_argument(
        "--scan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trigger security rescan after each credential update (default: enabled, use --no-scan to disable)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    registry_token = _load_registry_token(args.token_file)

    if args.loop:
        logger.info(
            f"Running in continuous mode, interval: {args.interval}s"
        )
        while True:
            try:
                refresh_all(
                    args.manifest,
                    args.registry_url,
                    registry_token,
                    run_scan=args.scan,
                )
            except Exception as e:
                logger.error(f"Refresh cycle failed: {e}")
            logger.info(f"Sleeping {args.interval}s until next refresh...")
            time.sleep(args.interval)
    else:
        refresh_all(
            args.manifest,
            args.registry_url,
            registry_token,
            run_scan=args.scan,
        )


if __name__ == "__main__":
    main()
