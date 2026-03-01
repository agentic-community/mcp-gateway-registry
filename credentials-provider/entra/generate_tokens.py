#!/usr/bin/env python3
"""
Generate OAuth2 access tokens for identities using Microsoft Entra ID.

Reads identity credentials from an input JSON file and generates tokens
for each identity using the OAuth2 client credentials flow.
"""

import argparse
import json
import logging
import os
import sys
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Default Entra ID login base URL
DEFAULT_ENTRA_LOGIN_BASE_URL = "https://login.microsoftonline.com"
DEFAULT_IDENTITIES_FILE = ".oauth-tokens/entra-identities.json"


class Colors:
    """ANSI color codes for console output."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"


def _redact_sensitive_value(
    value: str,
    show_chars: int = 8,
) -> str:
    """Redact sensitive value for logging."""
    if not value or len(value) <= show_chars:
        return "*" * len(value) if value else ""
    return value[:show_chars] + "*" * (len(value) - show_chars)


def _get_token_from_entra(
    client_id: str,
    client_secret: str,
    tenant_id: str,
    scope: Optional[str] = None,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Request access token from Microsoft Entra ID using client credentials."""
    login_base_url = os.environ.get(
        "ENTRA_LOGIN_BASE_URL",
        DEFAULT_ENTRA_LOGIN_BASE_URL,
    )

    token_url = f"{login_base_url}/{tenant_id}/oauth2/v2.0/token"

    # Default scope for Entra ID M2M tokens
    if not scope:
        scope = f"api://{client_id}/.default"

    if verbose:
        print(f"{Colors.BLUE}[DEBUG]{Colors.NC} Token URL: {token_url}")
        print(f"{Colors.BLUE}[DEBUG]{Colors.NC} Client ID: {client_id}")
        print(f"{Colors.BLUE}[DEBUG]{Colors.NC} Tenant ID: {tenant_id}")
        print(f"{Colors.BLUE}[DEBUG]{Colors.NC} Scope: {scope}")

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        response = requests.post(token_url, data=data, headers=headers, timeout=30)

        # Check for error response before raise_for_status
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_msg = error_data.get(
                    "error_description", error_data.get("error", "Unknown error")
                )
                print(f"{Colors.RED}[ERROR]{Colors.NC} Entra ID error: {error_msg}")
                if verbose:
                    print(
                        f"{Colors.BLUE}[DEBUG]{Colors.NC} Full error response: {json.dumps(error_data, indent=2)}"
                    )
            except json.JSONDecodeError:
                print(
                    f"{Colors.RED}[ERROR]{Colors.NC} HTTP {response.status_code}: {response.text}"
                )
            return None

        token_data = response.json()

        if "error_description" in token_data:
            print(
                f"{Colors.RED}[ERROR]{Colors.NC} Token request failed: {token_data['error_description']}"
            )
            return None

        if "access_token" not in token_data:
            print(f"{Colors.RED}[ERROR]{Colors.NC} No access token in response")
            return None

        return token_data

    except requests.exceptions.RequestException as e:
        print(f"{Colors.RED}[ERROR]{Colors.NC} Failed to make token request to Entra ID: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}[ERROR]{Colors.NC} Invalid JSON response: {e}")
        return None


def _save_token_file(
    identity_name: str,
    token_data: Dict[str, Any],
    client_id: str,
    tenant_id: str,
    scope: str,
    output_dir: str,
) -> bool:
    """Save token to JSON file."""
    access_token = token_data["access_token"]
    expires_in = token_data.get("expires_in")

    os.makedirs(output_dir, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    expires_at = None
    if expires_in:
        expiry_timestamp = datetime.now(timezone.utc).timestamp() + expires_in
        expires_at = datetime.fromtimestamp(
            expiry_timestamp,
            timezone.utc,
        ).isoformat()

    token_json = {
        "identity_name": identity_name,
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": expires_in,
        "generated_at": generated_at,
        "expires_at": expires_at,
        "provider": "entra",
        "tenant_id": tenant_id,
        "client_id": client_id,
        "scope": scope,
    }

    json_file = os.path.join(output_dir, f"{identity_name}.json")

    try:
        with open(json_file, "w") as f:
            json.dump(token_json, f, indent=2)
        os.chmod(json_file, 0o600)
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.NC} Failed to save token file: {e}")
        return False

    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} Token saved to: {json_file}")

    redacted_token = _redact_sensitive_value(access_token, 8)
    print(f"\nAccess Token: {redacted_token}")
    if expires_in:
        print(f"Expires in: {expires_in} seconds")
        if expires_at:
            expiry_time = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            print(f"Expires at: {expiry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    return True


def _load_identities_file(
    file_path: str,
) -> Optional[List[Dict[str, Any]]]:
    """Load identities from JSON file."""
    if not os.path.exists(file_path):
        print(f"{Colors.RED}[ERROR]{Colors.NC} Identities file not found: {file_path}")
        return None

    try:
        with open(file_path, "r") as f:
            identities = json.load(f)

        if not isinstance(identities, list):
            print(f"{Colors.RED}[ERROR]{Colors.NC} Identities file must contain a JSON array")
            return None

        return identities

    except json.JSONDecodeError as e:
        print(f"{Colors.RED}[ERROR]{Colors.NC} Failed to parse identities file: {e}")
        return None
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.NC} Failed to load identities file: {e}")
        return None


def generate_tokens(
    identities_file: str,
    output_dir: str,
    verbose: bool = False,
) -> bool:
    """Generate tokens for all identities in the input file."""
    identities = _load_identities_file(identities_file)
    if identities is None:
        return False

    if not identities:
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} No identities found in file")
        return True

    print(
        f"{Colors.GREEN}[SUCCESS]{Colors.NC} Found {len(identities)} identity(ies) in {identities_file}"
    )

    success_count = 0
    total_count = len(identities)

    for identity in identities:
        identity_name = identity.get("identity_name")
        if not identity_name:
            print(f"{Colors.RED}[ERROR]{Colors.NC} Identity missing 'identity_name' field")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing identity: {identity_name}")
        print("=" * 60)

        client_id = identity.get("client_id")
        client_secret = identity.get("client_secret")
        tenant_id = identity.get("tenant_id")
        scope = identity.get("scope")

        if not client_id:
            print(f"{Colors.RED}[ERROR]{Colors.NC} Identity '{identity_name}' missing 'client_id'")
            continue
        if not client_secret:
            print(
                f"{Colors.RED}[ERROR]{Colors.NC} Identity '{identity_name}' missing 'client_secret'"
            )
            continue
        if not tenant_id:
            print(f"{Colors.RED}[ERROR]{Colors.NC} Identity '{identity_name}' missing 'tenant_id'")
            continue

        print(f"Requesting access token for identity: {identity_name}")

        token_data = _get_token_from_entra(
            client_id,
            client_secret,
            tenant_id,
            scope,
            verbose,
        )

        if not token_data:
            print(
                f"{Colors.RED}[ERROR]{Colors.NC} Failed to generate token for identity: {identity_name}"
            )
            continue

        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} Access token generated!")

        if not scope:
            scope = f"api://{client_id}/.default"

        if _save_token_file(
            identity_name,
            token_data,
            client_id,
            tenant_id,
            scope,
            output_dir,
        ):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"Token generation complete: {success_count}/{total_count} successful")
    print("=" * 60)

    return success_count == total_count


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate OAuth2 access tokens using Microsoft Entra ID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate tokens using default identities file
  python generate_tokens.py

  # Generate tokens using custom identities file
  python generate_tokens.py --identities-file /path/to/identities.json

  # Generate tokens with verbose output
  python generate_tokens.py --verbose

Identities File Format (JSON array):
  [
    {
      "identity_name": "admin",
      "tenant_id": "your-tenant-id",
      "client_id": "your-client-id",
      "client_secret": "your-client-secret",
      "scope": "api://your-app-id/.default"  // optional
    }
  ]

Environment Variables:
  ENTRA_LOGIN_BASE_URL - Login base URL (default: https://login.microsoftonline.com)
        """,
    )

    parser.add_argument(
        "--identities-file",
        type=str,
        default=DEFAULT_IDENTITIES_FILE,
        help=f"Path to JSON file with identity credentials (default: {DEFAULT_IDENTITIES_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".oauth-tokens",
        help="Output directory for token files (default: .oauth-tokens)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    # Keep --all-agents for backwards compatibility but ignore it
    parser.add_argument(
        "--all-agents",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    try:
        success = generate_tokens(
            identities_file=args.identities_file,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[WARNING]{Colors.NC} Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.NC} Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
