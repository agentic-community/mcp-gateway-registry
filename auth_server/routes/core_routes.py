from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
)

from ..providers.factory import (
    get_auth_provider,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "simplified-auth-server"}


@router.get("/config")
async def get_auth_config() -> dict[str, object]:
    """Return the authentication configuration info."""
    try:
        auth_provider = get_auth_provider()
        provider_info = auth_provider.get_provider_info()

        if provider_info.get("provider_type") == "keycloak":
            return {
                "auth_type": "keycloak",
                "description": "Keycloak JWT token validation",
                "required_headers": [
                    "Authorization: Bearer <token>",
                ],
                "optional_headers": [],
                "provider_info": provider_info,
            }

        return {
            "auth_type": "cognito",
            "description": "Header-based Cognito token validation",
            "required_headers": [
                "Authorization: Bearer <token>",
                "X-User-Pool-Id: <pool_id>",
                "X-Client-Id: <client_id>",
            ],
            "optional_headers": [
                "X-Region: <region> (default: us-east-1)",
            ],
            "provider_info": provider_info,
        }
    except Exception as exc:
        logger.exception("Error getting auth config")
        return {
            "auth_type": "unknown",
            "description": f"Error getting provider config: {exc}",
            "error": str(exc),
        }

