from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Optional

import jwt
from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Request,
)
from fastapi.responses import (
    JSONResponse,
)
from pydantic import BaseModel

try:
    from ..constants import (
        DEFAULT_TOKEN_LIFETIME_HOURS,
        JWT_AUDIENCE,
        JWT_ISSUER,
        MAX_TOKEN_LIFETIME_HOURS,
        MAX_TOKENS_PER_USER_PER_HOUR,
    )
    from ..routes.oauth2_context import (
        SECRET_KEY,
    )
    from ..scopes_config import (
        reload_scopes_config,
    )
except ImportError:  # pragma: no cover
    from constants import (  # type: ignore[no-redef]
        DEFAULT_TOKEN_LIFETIME_HOURS,
        JWT_AUDIENCE,
        JWT_ISSUER,
        MAX_TOKEN_LIFETIME_HOURS,
        MAX_TOKENS_PER_USER_PER_HOUR,
    )
    from routes.oauth2_context import (  # type: ignore[no-redef]
        SECRET_KEY,
    )
    from scopes_config import (  # type: ignore[no-redef]
        reload_scopes_config,
    )

logger = logging.getLogger(__name__)

router = APIRouter()

_user_token_generation_counts: dict[str, int] = {}


def _hash_username(
    username: str,
) -> str:
    if not username:
        return "anonymous"
    return f"user_{hashlib.sha256(username.encode()).hexdigest()[:8]}"


def _validate_scope_subset(
    user_scopes: list[str],
    requested_scopes: list[str],
) -> bool:
    if not requested_scopes:
        return True

    user_scope_set = set(user_scopes)
    requested_scope_set = set(requested_scopes)
    is_valid = requested_scope_set.issubset(user_scope_set)

    if not is_valid:
        invalid_scopes = requested_scope_set - user_scope_set
        logger.warning("Invalid scopes requested: %s", sorted(invalid_scopes))

    return is_valid


def _check_rate_limit(
    username: str,
) -> bool:
    current_time = int(time.time())
    current_hour = current_time // 3600

    keys_to_remove: list[str] = []
    for key in list(_user_token_generation_counts.keys()):
        stored_hour = int(key.split(":")[1])
        if current_hour - stored_hour > 1:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del _user_token_generation_counts[key]

    rate_key = f"{username}:{current_hour}"
    current_count = _user_token_generation_counts.get(rate_key, 0)

    if current_count >= MAX_TOKENS_PER_USER_PER_HOUR:
        logger.warning(
            "Rate limit exceeded for user %s: %s tokens this hour",
            _hash_username(username),
            current_count,
        )
        return False

    _user_token_generation_counts[rate_key] = current_count + 1
    return True


class GenerateTokenRequest(BaseModel):
    """Request model for token generation."""

    user_context: dict[str, Any]
    requested_scopes: list[str] = []
    expires_in_hours: int = DEFAULT_TOKEN_LIFETIME_HOURS
    description: Optional[str] = None


class GenerateTokenResponse(BaseModel):
    """Response model for token generation."""

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_in: int
    refresh_expires_in: Optional[int] = None
    scope: str
    issued_at: int
    description: Optional[str] = None


@router.post("/internal/tokens", response_model=GenerateTokenResponse)
async def generate_user_token(
    request: GenerateTokenRequest,
) -> GenerateTokenResponse:
    """Generate a JWT token for a user with specified scopes."""
    try:
        user_context = request.user_context
        username = user_context.get("username")
        user_scopes = user_context.get("scopes", [])

        if not username:
            raise HTTPException(
                status_code=400,
                detail="Username is required in user context",
                headers={"Connection": "close"},
            )

        if not _check_rate_limit(str(username)):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {MAX_TOKENS_PER_USER_PER_HOUR} tokens per hour.",
                headers={"Connection": "close"},
            )

        expires_in_hours = request.expires_in_hours
        if expires_in_hours <= 0 or expires_in_hours > MAX_TOKEN_LIFETIME_HOURS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid expiration time. Must be between 1 and "
                    f"{MAX_TOKEN_LIFETIME_HOURS} hours."
                ),
                headers={"Connection": "close"},
            )

        requested_scopes = request.requested_scopes if request.requested_scopes else user_scopes
        if not isinstance(requested_scopes, list):
            requested_scopes = []
        if not isinstance(user_scopes, list):
            user_scopes = []

        if not _validate_scope_subset(user_scopes, requested_scopes):
            invalid_scopes = set(requested_scopes) - set(user_scopes)
            raise HTTPException(
                status_code=403,
                detail=(
                    "Requested scopes exceed user permissions. Invalid scopes: "
                    f"{sorted(invalid_scopes)}"
                ),
                headers={"Connection": "close"},
            )

        current_time = int(time.time())
        expires_at = current_time + (expires_in_hours * 3600)

        access_payload: dict[str, Any] = {
            "iss": JWT_ISSUER,
            "aud": JWT_AUDIENCE,
            "sub": username,
            "scope": " ".join(requested_scopes),
            "exp": expires_at,
            "iat": current_time,
            "jti": str(uuid.uuid4()),
            "token_use": "access",
            "client_id": "user-generated",
            "token_type": "user_generated",
        }

        if request.description:
            access_payload["description"] = request.description

        access_token = jwt.encode(access_payload, SECRET_KEY, algorithm="HS256")

        logger.info(
            "Generated access token for user '%s' with scopes: %s, expires in %s hours",
            _hash_username(str(username)),
            requested_scopes,
            expires_in_hours,
        )

        return GenerateTokenResponse(
            access_token=access_token,
            refresh_token=None,
            expires_in=expires_in_hours * 3600,
            refresh_expires_in=0,
            scope=" ".join(requested_scopes),
            issued_at=current_time,
            description=request.description,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error generating token: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Internal error generating token",
            headers={"Connection": "close"},
        ) from exc


@router.post("/internal/reload-scopes")
async def reload_scopes(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> JSONResponse:
    """Reload the scopes.yml configuration file via Basic Auth."""
    del request

    if not authorization or not authorization.startswith("Basic "):
        logger.warning("No Basic Auth header found for reload-scopes request")
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    try:
        encoded_credentials = authorization.split(" ")[1]
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        username, password = decoded_credentials.split(":", 1)
    except Exception as exc:
        logger.warning("Failed to decode Basic Auth credentials: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication format",
            headers={"WWW-Authenticate": "Basic"},
        ) from exc

    admin_user = os.environ.get("ADMIN_USER", "admin")
    admin_password = os.environ.get("ADMIN_PASSWORD")

    if not admin_password:
        logger.error("ADMIN_PASSWORD environment variable not set")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error",
        )

    if username != admin_user or password != admin_password:
        logger.warning("Failed admin authentication attempt for reload-scopes from %s", username)
        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    try:
        config = reload_scopes_config()
        logger.info("Successfully reloaded scopes configuration by admin '%s'", username)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Scopes configuration reloaded successfully",
                "timestamp": datetime.utcnow().isoformat(),
                "group_mappings_count": len(config.get("group_mappings", {})),
            },
        )
    except Exception as exc:
        logger.error("Failed to reload scopes configuration: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload scopes: {exc}",
        ) from exc

