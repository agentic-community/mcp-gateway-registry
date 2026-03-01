#!/usr/bin/env python3
"""
CLI tool to authenticate users and obtain access tokens for programmatic API access.

This script implements the OAuth2 Device Code Flow, which allows users to authenticate
by visiting a URL and entering a code, without needing to expose their credentials
to the CLI application.

Usage:
    # Authenticate and save token to file
    uv run python cli/get_user_token.py --output .token

    # Authenticate with custom output file
    uv run python cli/get_user_token.py --output my-token.json

    # Show token on stdout (don't save)
    uv run python cli/get_user_token.py --stdout

Environment Variables:
    ENTRA_TENANT_ID: Azure AD tenant ID
    ENTRA_CLIENT_ID: App registration client ID
    ENTRA_CLIENT_SECRET: App registration client secret (optional for public clients)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Default Entra ID login base URL
DEFAULT_ENTRA_LOGIN_BASE_URL = "https://login.microsoftonline.com"


def _get_env_or_error(name: str, default: Optional[str] = None) -> str:
    """Get environment variable or raise error if not set.

    Args:
        name: Environment variable name
        default: Default value if not set

    Returns:
        Environment variable value

    Raises:
        ValueError: If variable not set and no default
    """
    value = os.environ.get(name, default)
    if not value:
        raise ValueError(f"Environment variable {name} is required")
    return value


def _initiate_device_code_flow(tenant_id: str, client_id: str, scope: Optional[str] = None) -> dict:
    """Initiate device code flow.

    Args:
        tenant_id: Azure AD tenant ID
        client_id: App registration client ID
        scope: OAuth scopes to request

    Returns:
        Device code response from Entra ID
    """
    import requests

    login_base_url = os.environ.get("ENTRA_LOGIN_BASE_URL", DEFAULT_ENTRA_LOGIN_BASE_URL)

    device_code_url = f"{login_base_url}/{tenant_id}/oauth2/v2.0/devicecode"

    if not scope:
        scope = f"api://{client_id}/user_impersonation openid profile email"

    data = {"client_id": client_id, "scope": scope}

    response = requests.post(device_code_url, data=data, timeout=10)

    if response.status_code != 200:
        error_data = response.json()
        error_desc = error_data.get("error_description", error_data.get("error", "Unknown error"))
        logger.error(f"Device code request failed: {error_desc}")
        raise ValueError(f"Device code flow not available: {error_desc}")

    return response.json()


def _poll_for_token(
    tenant_id: str, client_id: str, device_code: str, interval: int = 5, timeout: int = 300
) -> dict:
    """Poll for token after user completes authentication.

    Args:
        tenant_id: Azure AD tenant ID
        client_id: App registration client ID
        device_code: Device code from initiation
        interval: Polling interval in seconds
        timeout: Maximum wait time in seconds

    Returns:
        Token response from Entra ID
    """
    import requests

    login_base_url = os.environ.get("ENTRA_LOGIN_BASE_URL", DEFAULT_ENTRA_LOGIN_BASE_URL)

    token_url = f"{login_base_url}/{tenant_id}/oauth2/v2.0/token"

    data = {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "client_id": client_id,
        "device_code": device_code,
    }

    start_time = time.time()

    while (time.time() - start_time) < timeout:
        response = requests.post(token_url, data=data, timeout=10)

        if response.status_code == 200:
            return response.json()

        error_data = response.json()
        error = error_data.get("error", "")

        if error == "authorization_pending":
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(interval)
            continue
        elif error == "slow_down":
            interval += 5
            time.sleep(interval)
            continue
        elif error == "expired_token":
            raise ValueError("Device code expired. Please try again.")
        elif error == "access_denied":
            raise ValueError("Authorization was denied.")
        else:
            error_desc = error_data.get("error_description", error)
            raise ValueError(f"Token request failed: {error_desc}")

    raise ValueError("Authentication timed out. Please try again.")


def _save_token(token_data: dict, output_path: str) -> None:
    """Save token data to file.

    Args:
        token_data: Token response from Entra ID
        output_path: Path to save token file
    """
    # Add metadata
    token_data["obtained_at"] = datetime.utcnow().isoformat()

    with open(output_path, "w") as f:
        json.dump(token_data, f, indent=2)

    # Set restrictive permissions
    os.chmod(output_path, 0o600)

    logger.info(f"Token saved to {output_path}")


def _extract_access_token(token_data: dict) -> str:
    """Extract just the access token from response.

    Args:
        token_data: Full token response

    Returns:
        Access token string
    """
    return token_data.get("access_token", "")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description="Authenticate with Entra ID and obtain an access token for API access",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Authenticate and save token to .token file
    uv run python cli/get_user_token.py --output .token

    # Authenticate and print token to stdout
    uv run python cli/get_user_token.py --stdout

    # Use the token with registry_management.py
    uv run python api/registry_management.py --token-file .token --registry-url http://localhost list

Environment Variables:
    ENTRA_TENANT_ID     Azure AD tenant ID (required)
    ENTRA_CLIENT_ID     App registration client ID (required)
    ENTRA_LOGIN_BASE_URL  Login URL (default: https://login.microsoftonline.com)
""",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to save the token file (default: .token)",
        default=".token",
    )

    parser.add_argument(
        "--stdout", action="store_true", help="Print token to stdout instead of saving to file"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Output full token response (with refresh token, expiry, etc.)",
    )

    parser.add_argument(
        "--scope",
        type=str,
        help="OAuth scopes to request (default: user_impersonation openid profile email)",
    )

    parser.add_argument(
        "--timeout", type=int, default=300, help="Authentication timeout in seconds (default: 300)"
    )

    args = parser.parse_args()

    try:
        # Get configuration from environment
        tenant_id = _get_env_or_error("ENTRA_TENANT_ID")
        client_id = _get_env_or_error("ENTRA_CLIENT_ID")

        logger.info("Starting device code authentication flow")
        logger.info(f"Tenant ID: {tenant_id}")
        logger.info(f"Client ID: {client_id}")

        # Initiate device code flow
        device_code_response = _initiate_device_code_flow(
            tenant_id=tenant_id, client_id=client_id, scope=args.scope
        )

        # Display instructions to user
        print("\n" + "=" * 60)
        print("AUTHENTICATION REQUIRED")
        print("=" * 60)
        print(f"\n{device_code_response.get('message', '')}\n")
        print(f"  URL:  {device_code_response.get('verification_uri', '')}")
        print(f"  Code: {device_code_response.get('user_code', '')}")
        print("\n" + "=" * 60)
        print("\nWaiting for authentication", end="")

        # Poll for token
        token_data = _poll_for_token(
            tenant_id=tenant_id,
            client_id=client_id,
            device_code=device_code_response["device_code"],
            interval=device_code_response.get("interval", 5),
            timeout=args.timeout,
        )

        print("\n\nAuthentication successful!")

        # Output token
        if args.stdout:
            if args.full:
                print(json.dumps(token_data, indent=2))
            else:
                print(token_data["access_token"])
        else:
            if args.full:
                _save_token(token_data, args.output)
            else:
                # Save just the access token for compatibility with CLI tools
                with open(args.output, "w") as f:
                    f.write(token_data["access_token"])
                os.chmod(args.output, 0o600)
                logger.info(f"Access token saved to {args.output}")

            print(f"\nToken saved to: {args.output}")
            print(f"Token expires in: {token_data.get('expires_in', 'unknown')} seconds")
            print(f"\nUsage:")
            print(
                f"  uv run python api/registry_management.py --token-file {args.output} --registry-url http://localhost list"
            )

        return 0

    except ValueError as e:
        logger.error(f"Authentication failed: {e}")
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
