import logging
import os
from typing import (
    Annotated,
)

import httpx
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import (
    HTMLResponse,
)
from fastapi.templating import (
    Jinja2Templates,
)

from ..auth.dependencies import (
    enhanced_auth,
)
from ..core.config import (
    settings,
)

logger = logging.getLogger(__name__)

router = APIRouter()

templates = Jinja2Templates(directory=settings.templates_dir)


@router.get("/tokens", response_class=HTMLResponse)
async def token_generation_page(
    request: Request,
    user_context: Annotated[dict, Depends(enhanced_auth)],
):
    """Show token generation page for authenticated users."""
    return templates.TemplateResponse(
        "token_generation.html",
        {
            "request": request,
            "username": user_context["username"],
            "user_context": user_context,
            "user_scopes": user_context["scopes"],
            "available_scopes": user_context["scopes"],
        },
    )


@router.post("/tokens/generate")
async def generate_user_token(
    request: Request,
    user_context: Annotated[dict, Depends(enhanced_auth)],
):
    """Generate a JWT token for the authenticated user."""
    try:
        try:
            body = await request.json()
        except Exception as e:
            logger.warning("Invalid JSON in token generation request: %s", e)
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in request body",
            ) from e

        requested_scopes = body.get("requested_scopes", [])
        expires_in_hours = body.get("expires_in_hours", 8)
        description = body.get("description", "")

        if not isinstance(expires_in_hours, int) or expires_in_hours <= 0 or expires_in_hours > 24:
            raise HTTPException(
                status_code=400,
                detail="expires_in_hours must be an integer between 1 and 24",
            )

        if requested_scopes and not isinstance(requested_scopes, list):
            raise HTTPException(
                status_code=400,
                detail="requested_scopes must be a list of scope names",
            )

        if requested_scopes:
            invalid_scopes = [s for s in requested_scopes if s not in user_context["scopes"]]
            if invalid_scopes:
                raise HTTPException(
                    status_code=403,
                    detail=f"You do not have access to scopes: {', '.join(invalid_scopes)}",
                )

        auth_request = {
            "username": user_context["username"],
            "scopes": requested_scopes or user_context["scopes"],
            "expires_in_hours": expires_in_hours,
            "description": description,
        }

        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}

            auth_server_url = settings.auth_server_url
            response = await client.post(
                f"{auth_server_url}/internal/tokens",
                json=auth_request,
                headers=headers,
                timeout=10.0,
            )

            if response.status_code == 200:
                token_data = response.json()
                logger.info("Successfully generated token for user '%s'", user_context["username"])

                formatted_response = {
                    "success": True,
                    "tokens": {
                        "access_token": token_data.get("access_token"),
                        "refresh_token": token_data.get("refresh_token"),
                        "expires_in": token_data.get("expires_in"),
                        "refresh_expires_in": token_data.get("refresh_expires_in"),
                        "token_type": token_data.get("token_type", "Bearer"),
                        "scope": token_data.get("scope", ""),
                    },
                    "keycloak_url": getattr(settings, "keycloak_url", None) or "http://keycloak:8080",
                    "realm": getattr(settings, "keycloak_realm", None) or "mcp-gateway",
                    "client_id": "user-generated",
                    "token_data": token_data,
                    "user_scopes": user_context["scopes"],
                    "requested_scopes": requested_scopes or user_context["scopes"],
                }

                return formatted_response

            error_detail = "Unknown error"
            try:
                error_response = response.json()
                error_detail = error_response.get("detail", "Unknown error")
            except Exception:
                error_detail = response.text

            logger.warning("Auth server returned error %s: %s", response.status_code, error_detail)
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Token generation failed: {error_detail}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error generating token for user '%s': %s", user_context["username"], e)
        raise HTTPException(
            status_code=500,
            detail="Internal error generating token",
        ) from e


@router.get("/admin/tokens")
async def get_admin_tokens(
    user_context: Annotated[dict, Depends(enhanced_auth)],
):
    """Admin-only endpoint to retrieve Keycloak M2M access token."""
    if not user_context.get("is_admin", False):
        logger.warning(
            "Non-admin user %s attempted to access admin tokens",
            user_context["username"],
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is only available to admin users",
        )

    try:
        from ..utils.keycloak_manager import (
            KEYCLOAK_ADMIN_URL,
            KEYCLOAK_REALM,
        )

        m2m_client_id = os.getenv("KEYCLOAK_M2M_CLIENT_ID", "mcp-gateway-m2m")
        m2m_client_secret = os.getenv("KEYCLOAK_M2M_CLIENT_SECRET")

        if not m2m_client_secret:
            raise HTTPException(
                status_code=500,
                detail="Keycloak M2M client secret not configured",
            )

        token_url = f"{KEYCLOAK_ADMIN_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"

        data = {
            "grant_type": "client_credentials",
            "client_id": m2m_client_id,
            "client_secret": m2m_client_secret,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(token_url, data=data, headers=headers)
            response.raise_for_status()

            token_data = response.json()

            refresh_token = None
            refresh_expires_in_seconds = 0

            logger.info(
                "Admin user %s retrieved Keycloak M2M tokens (no refresh token)",
                user_context["username"],
            )

            return {
                "success": True,
                "tokens": {
                    "access_token": token_data.get("access_token"),
                    "refresh_token": refresh_token,
                    "expires_in": token_data.get("expires_in"),
                    "refresh_expires_in": refresh_expires_in_seconds,
                    "token_type": token_data.get("token_type", "Bearer"),
                    "scope": token_data.get("scope", ""),
                },
                "keycloak_url": KEYCLOAK_ADMIN_URL,
                "realm": KEYCLOAK_REALM,
                "client_id": m2m_client_id,
            }

    except httpx.HTTPStatusError as e:
        logger.error("Failed to retrieve Keycloak tokens: HTTP %s", e.response.status_code)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to authenticate with Keycloak: HTTP {e.response.status_code}",
        ) from e
    except Exception as e:
        logger.error("Unexpected error retrieving Keycloak tokens: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal error retrieving Keycloak tokens",
        ) from e

