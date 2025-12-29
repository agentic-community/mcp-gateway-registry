from __future__ import annotations

import logging
import os
from urllib.parse import (
    urlparse,
)

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)

try:
    from ..enforceai_runtime import (
        get_enforceai_stores,
        get_identity_resolver,
        load_scope_catalog,
    )
except ImportError:  # pragma: no cover
    from enforceai_runtime import (  # type: ignore[no-redef]
        get_enforceai_stores,
        get_identity_resolver,
        load_scope_catalog,
    )

try:
    from .validate_common import (
        _parse_request_payload,
    )
    from .validate_enforceai import (
        _validate_request_with_enforceai,
    )
    from .validate_legacy import (
        _validate_request_legacy,
    )
except ImportError:  # pragma: no cover
    from validate_common import (  # type: ignore[no-redef]
        _parse_request_payload,
    )
    from validate_enforceai import (  # type: ignore[no-redef]
        _validate_request_with_enforceai,
    )
    from validate_legacy import (  # type: ignore[no-redef]
        _validate_request_legacy,
    )

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/validate")
async def validate_request(request: Request):
    """
    Validate a request by extracting configuration from headers and validating the bearer token.

    Expected headers:
    - Authorization: Bearer <token>
    - X-User-Pool-Id: <user_pool_id>
    - X-Client-Id: <client_id>
    - X-Region: <region> (optional, defaults to us-east-1)
    - X-Original-URL: <original_url> (optional, for scope validation)

    Returns:
        HTTP 200 with user info headers if valid, HTTP 401/403 if invalid

    Raises:
        HTTPException: If the token is missing, invalid, or configuration is incomplete
    """

    try:
        enforceai_enabled = bool(os.environ.get("ENFORCEAI_DB_PATH"))

        authorization = request.headers.get("X-Authorization")
        if not authorization:
            authorization = request.headers.get("Authorization")

        cookie_header = request.headers.get("Cookie", "")
        user_pool_id = request.headers.get("X-User-Pool-Id")
        client_id = request.headers.get("X-Client-Id")
        region = request.headers.get("X-Region", "us-east-1")
        original_url = request.headers.get("X-Original-URL")
        original_path = ""
        if original_url:
            try:
                original_path = urlparse(original_url).path or ""
            except Exception:
                original_path = ""

        body = request.headers.get("X-Body")
        request_payload = _parse_request_payload(body)

        is_registry_api_request = original_path.startswith("/api/")

        server_name_from_url = None
        if original_url:
            try:
                parsed_url = urlparse(original_url)
                path = parsed_url.path.strip("/")
                path_parts = path.split("/") if path else []
                server_name_from_url = path_parts[0] if path_parts else None
                logger.info(
                    "Extracted server_name '%s' from original_url: %s",
                    server_name_from_url,
                    original_url,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to extract server_name from original_url %s: %s",
                    original_url,
                    exc,
                )

        if is_registry_api_request:
            server_name_from_url = None

        server_name = server_name_from_url
        tool_name = None
        if request_payload and isinstance(request_payload, dict):
            tool_name = (
                request_payload.get("method")
                or request_payload.get("tool")
                or request_payload.get("name")
            )
            if not tool_name and "params" in request_payload and isinstance(
                request_payload.get("params"),
                dict,
            ):
                tool_name = (
                    request_payload["params"].get("method")
                    or request_payload["params"].get("tool")
                    or request_payload["params"].get("name")
                )

        if enforceai_enabled:
            return await _validate_request_with_enforceai(
                request=request,
                original_path=original_path,
                server_name_from_url=server_name_from_url,
                server_name=server_name,
                tool_name=tool_name,
                request_payload=request_payload,
                get_identity_resolver=get_identity_resolver,
                load_scope_catalog=load_scope_catalog,
                get_enforceai_stores=get_enforceai_stores,
            )

        return await _validate_request_legacy(
            request=request,
            authorization=authorization,
            cookie_header=cookie_header,
            user_pool_id=user_pool_id,
            client_id=client_id,
            region=region,
            original_url=original_url,
            server_name_from_url=server_name_from_url,
            request_payload=request_payload,
        )

    except ValueError as e:
        logger.warning("Token validation failed: %s", e)
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
        ) from e
    except HTTPException as e:
        if e.status_code in {400, 401, 403, 409, 424, 503}:
            raise

        logger.error("HTTP error during validation: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal validation error: {str(e)}",
            headers={"Connection": "close"},
        ) from e
    except Exception as e:
        logger.error("Unexpected error during validation: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal validation error: {str(e)}",
            headers={"Connection": "close"},
        ) from e

