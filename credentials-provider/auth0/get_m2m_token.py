"""Get Auth0 M2M token using client credentials flow.

This script obtains a JWT token from Auth0 using OAuth2 client credentials grant.
The token is saved to a temporary file and the file path is printed.
"""

import argparse
import json
import logging
import os
import sys
import tempfile

import jwt
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


def _get_auth0_domain() -> str:
    """Get Auth0 domain from CLI arg or environment variable.

    Returns:
        Auth0 domain (e.g., dev-abc123.us.auth0.com)

    Raises:
        ValueError: If domain not provided
    """
    domain = os.getenv("AUTH0_DOMAIN")
    if domain:
        return domain.replace("https://", "").rstrip("/")

    raise ValueError("Auth0 domain must be provided via --auth0-domain or AUTH0_DOMAIN env var")


def _get_client_id() -> str:
    """Get client ID from CLI arg or environment variable.

    Returns:
        Auth0 M2M client ID

    Raises:
        ValueError: If client ID not provided
    """
    client_id = os.getenv("AUTH0_M2M_CLIENT_ID")
    if client_id:
        return client_id

    raise ValueError("Client ID must be provided via --client-id or AUTH0_M2M_CLIENT_ID env var")


def _get_client_secret() -> str:
    """Get client secret from CLI arg or environment variable.

    Returns:
        Auth0 M2M client secret

    Raises:
        ValueError: If client secret not provided
    """
    client_secret = os.getenv("AUTH0_M2M_CLIENT_SECRET")
    if client_secret:
        return client_secret

    raise ValueError(
        "Client secret must be provided via --client-secret or AUTH0_M2M_CLIENT_SECRET env var"
    )


def _request_m2m_token(
    auth0_domain: str,
    client_id: str,
    client_secret: str,
    audience: str | None = None,
) -> dict[str, str]:
    """Request M2M token from Auth0 using client credentials.

    Args:
        auth0_domain: Auth0 domain (e.g., dev-abc123.us.auth0.com)
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret
        audience: API audience (defaults to Management API: https://{domain}/api/v2/)

    Returns:
        Token response dictionary with access_token, token_type, expires_in

    Raises:
        ValueError: If token request fails
    """
    # Default to Management API audience if not provided
    if not audience:
        audience = f"https://{auth0_domain}/api/v2/"

    token_url = f"https://{auth0_domain}/oauth/token"

    logger.info(f"Requesting M2M token from {token_url}")
    logger.info(f"Audience: {audience}")

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "audience": audience,
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }

    try:
        response = requests.post(
            token_url,
            data=data,
            headers=headers,
            timeout=30,
        )

        # Log response details for debugging
        if response.status_code != 200:
            try:
                error_data = response.json()
                logger.error(f"Auth0 error response: {json.dumps(error_data, indent=2)}")
            except Exception:
                logger.error(f"Auth0 error response (non-JSON): {response.text}")

        response.raise_for_status()

        token_data = response.json()
        logger.info(
            f"Successfully obtained M2M token, expires in {token_data.get('expires_in', 'unknown')} seconds"
        )

        return token_data

    except requests.RequestException as e:
        logger.error(f"Failed to get M2M token: {e}")
        raise ValueError(f"M2M token request failed: {e}")


def _decode_token(access_token: str) -> dict[str, str]:
    """Decode JWT token without verification to display claims.

    Args:
        access_token: JWT access token string

    Returns:
        Dictionary of decoded token claims
    """
    try:
        claims = jwt.decode(access_token, options={"verify_signature": False})
        return claims
    except Exception as e:
        logger.warning(f"Failed to decode token: {e}")
        return {}


def _display_decoded_token(claims: dict[str, str]) -> None:
    """Display decoded token claims in a readable format.

    Args:
        claims: Dictionary of decoded JWT claims
    """
    if not claims:
        return

    print("\n" + "=" * 60)
    print("DECODED JWT TOKEN CLAIMS")
    print("=" * 60)
    print(json.dumps(claims, indent=2))
    print("\n" + "=" * 60)
    print("KEY INFORMATION")
    print("=" * 60)
    print(f"Grant ID (gty):   {claims.get('gty', 'N/A')}")
    print(f"Azure AD (azp):   {claims.get('azp', 'N/A')}")
    print(f"Subject (sub):    {claims.get('sub', 'N/A')}")
    print(f"Issuer (iss):     {claims.get('iss', 'N/A')}")
    print(f"Audience (aud):   {claims.get('aud', 'N/A')}")
    print(f"Scopes (scope):   {claims.get('scope', 'N/A')}")
    print(f"Permissions:      {claims.get('permissions', [])}")

    # Display expiration info
    if "exp" in claims and "iat" in claims:
        from datetime import datetime

        exp_time = datetime.fromtimestamp(claims["exp"])
        iat_time = datetime.fromtimestamp(claims["iat"])
        lifetime_hours = (claims["exp"] - claims["iat"]) / 3600
        print(f"\nIssued at:        {iat_time} UTC")
        print(f"Expires at:       {exp_time} UTC")
        print(f"Lifetime:         {lifetime_hours:.1f} hours")
    print("=" * 60 + "\n")


def _save_token_to_file(token_data: dict[str, str]) -> str:
    """Save token data to temporary file.

    Args:
        token_data: Token response dictionary

    Returns:
        Path to temporary file containing token
    """
    # Create temporary file with secure permissions (0600)
    fd, temp_path = tempfile.mkstemp(
        prefix="auth0_m2m_token_",
        suffix=".json",
        dir="/tmp",
    )

    try:
        # Write token data as JSON
        with os.fdopen(fd, "w") as f:
            json.dump(token_data, f, indent=2)

        # Ensure file has restrictive permissions
        os.chmod(temp_path, 0o600)

        logger.info(f"Token saved to {temp_path}")
        return temp_path

    except Exception as e:
        # Clean up on error
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise ValueError(f"Failed to save token to file: {e}")


def main() -> None:
    """Main function to get Auth0 M2M token and save to file."""
    parser = argparse.ArgumentParser(
        description="Get Auth0 M2M token using client credentials flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Using environment variables
    export AUTH0_DOMAIN=dev-abc123.us.auth0.com
    export AUTH0_M2M_CLIENT_ID=KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd
    export AUTH0_M2M_CLIENT_SECRET=lbjH6Z81GkovgAHwXRV-qiKV9f6sUVzsnheJoX7KJcu2ojGXMTjJ4i0Zn49kKfVm
    uv run python -m credentials-provider.auth0.get_m2m_token

    # Using CLI arguments (Management API)
    uv run python -m credentials-provider.auth0.get_m2m_token \\
        --auth0-domain dev-abc123.us.auth0.com \\
        --client-id KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd \\
        --client-secret lbjH6Z81GkovgAHwXRV-qiKV9f6sUVzsnheJoX7KJcu2ojGXMTjJ4i0Zn49kKfVm

    # Custom API audience
    uv run python -m credentials-provider.auth0.get_m2m_token \\
        --auth0-domain dev-abc123.us.auth0.com \\
        --client-id KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd \\
        --client-secret lbjH6Z81GkovgAHwXRV-qiKV9f6sUVzsnheJoX7KJcu2ojGXMTjJ4i0Zn49kKfVm \\
        --audience https://my-api.example.com
""",
    )

    parser.add_argument(
        "--auth0-domain",
        type=str,
        help="Auth0 domain (e.g., dev-abc123.us.auth0.com). Can also use AUTH0_DOMAIN env var.",
    )

    parser.add_argument(
        "--client-id",
        type=str,
        help="OAuth2 M2M client ID. Can also use AUTH0_M2M_CLIENT_ID env var.",
    )

    parser.add_argument(
        "--client-secret",
        type=str,
        help="OAuth2 M2M client secret. Can also use AUTH0_M2M_CLIENT_SECRET env var.",
    )

    parser.add_argument(
        "--audience",
        type=str,
        help="API audience (default: https://{domain}/api/v2/ for Management API)",
    )

    parser.add_argument(
        "--show-token",
        action="store_true",
        help="Display decoded token claims (default: True)",
        default=True,
    )

    parser.add_argument(
        "--no-show-token",
        action="store_true",
        help="Do not display decoded token claims",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Get configuration from CLI args or environment variables
        auth0_domain = args.auth0_domain or _get_auth0_domain()
        client_id = args.client_id or _get_client_id()
        client_secret = args.client_secret or _get_client_secret()

        # Request M2M token from Auth0
        token_data = _request_m2m_token(
            auth0_domain=auth0_domain,
            client_id=client_id,
            client_secret=client_secret,
            audience=args.audience,
        )

        # Decode and display token if requested
        show_token = args.show_token and not args.no_show_token
        if show_token and "access_token" in token_data:
            claims = _decode_token(token_data["access_token"])
            _display_decoded_token(claims)

        # Save token to temporary file
        token_file_path = _save_token_to_file(token_data)

        # Print the file path
        print(f"Token saved to: {token_file_path}")

    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
