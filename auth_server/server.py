"""
Simplified Authentication server that validates JWT tokens against Amazon Cognito.
Configuration is passed via headers instead of environment variables.
"""

import argparse
import logging
import os
import boto3
import jwt
import requests
import json
import yaml
import time
import uuid
import hashlib
from jwt.api_jwk import PyJWK
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import importlib
from typing import Dict, Optional, List, Any
from functools import lru_cache
from botocore.exceptions import ClientError
from fastapi import FastAPI, Header, HTTPException, Request, Cookie
from fastapi.responses import JSONResponse, Response, RedirectResponse
import uvicorn
from pydantic import BaseModel
from pathlib import Path
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import secrets
import urllib.parse
import httpx
from string import Template
import os as _os
import json as _json

from gateway_session import (
    build_session_cookie_payload,
)
from gateway_csrf import (
    validate_csrf_token,
)
from gateway_session import (
    normalize_session_data,
)
from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from auth_server.enforceai.stores.sqlite.session_store import (
    SqliteSessionStore,
)
from auth_server.enforceai.stores.sqlite.user_store import (
    SqliteUserStore,
)

# Import metrics middleware (support repo + Docker module layouts)
try:
    from .metrics_middleware import add_auth_metrics_middleware
except ImportError:  # pragma: no cover
    from metrics_middleware import add_auth_metrics_middleware

# Import core route modules (support repo + Docker module layouts)
try:
    from .routes.core_routes import router as core_router
except ImportError:  # pragma: no cover
    from routes.core_routes import router as core_router

# Import validate route module (support repo + Docker module layouts)
try:
    from .routes.validate_routes import router as validate_router
except ImportError:  # pragma: no cover
    from routes.validate_routes import router as validate_router

# Import OAuth2 config + routes (support repo + Docker module layouts)
try:
    from .routes.oauth2_context import (
        CSRF_TOKEN_MAX_AGE_SECONDS,
        SECRET_KEY,
        SESSION_COOKIE_NAME,
        signer,
    )
    from .routes.oauth2_routes import (
        router as oauth2_router,
    )
except ImportError:  # pragma: no cover
    from routes.oauth2_context import (
        CSRF_TOKEN_MAX_AGE_SECONDS,
        SECRET_KEY,
        SESSION_COOKIE_NAME,
        signer,
    )
    from routes.oauth2_routes import (
        router as oauth2_router,
    )

# Import provider factory (support repo + Docker module layouts)
try:
    from .providers.factory import get_auth_provider
except ImportError:  # pragma: no cover
    from providers.factory import get_auth_provider

try:
    from .providers.cognito_validator import SimplifiedCognitoValidator
except ImportError:  # pragma: no cover
    from providers.cognito_validator import SimplifiedCognitoValidator

try:
    from .constants import (
        JWT_AUDIENCE,
        JWT_ISSUER,
    )
except ImportError:  # pragma: no cover
    from constants import (
        JWT_AUDIENCE,
        JWT_ISSUER,
    )

try:
    from .routes.internal_routes import (
        router as internal_router,
    )
except ImportError:  # pragma: no cover
    from routes.internal_routes import (
        router as internal_router,
    )

try:
    from .validation_utils import (
        anonymize_ip,
        hash_username,
        mask_headers,
        mask_sensitive_id,
        validate_server_tool_access,
        validate_session_cookie,
    )
except ImportError:  # pragma: no cover
    from validation_utils import (  # type: ignore[no-redef]
        anonymize_ip,
        hash_username,
        mask_headers,
        mask_sensitive_id,
        validate_server_tool_access,
        validate_session_cookie,
    )

try:
    from .enforceai_support import (
        emit_enforceai_audit_event as _emit_enforceai_audit_event,
        resolve_enforceai_scopes_catalog_path as _resolve_enforceai_scopes_catalog_path,
    )
except ImportError:  # pragma: no cover
    from enforceai_support import (  # type: ignore[no-redef]
        emit_enforceai_audit_event as _emit_enforceai_audit_event,
        resolve_enforceai_scopes_catalog_path as _resolve_enforceai_scopes_catalog_path,
    )

try:
    from .middleware.enforceai_csrf import (
        add_enforceai_csrf_middleware,
    )
except ImportError:  # pragma: no cover
    from middleware.enforceai_csrf import (  # type: ignore[no-redef]
        add_enforceai_csrf_middleware,
    )

try:
    import auth_server.scopes_config as scopes_config
except Exception:  # noqa: BLE001
    import scopes_config  # type: ignore[no-redef]

try:
    from .enforceai_runtime import (
        _load_enforceai_runtime,
        evaluate_tool_call,
        get_enforceai_settings,
        get_enforceai_stores,
        get_identity_resolver,
        get_upstream_oauth_token_client,
        load_enforceai_management_router,
        load_scope_catalog,
        resolve_callable_tools_for_server,
    )
except ImportError:  # pragma: no cover
    from enforceai_runtime import (  # type: ignore[no-redef]
        _load_enforceai_runtime,
        evaluate_tool_call,
        get_enforceai_settings,
        get_enforceai_stores,
        get_identity_resolver,
        get_upstream_oauth_token_client,
        load_enforceai_management_router,
        load_scope_catalog,
        resolve_callable_tools_for_server,
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# EnforceAI required guard (fail-fast)
def _enforce_enforceai_required() -> None:
    enforceai_required_raw = _os.environ.get("ENFORCEAI_REQUIRED", "false")
    enforceai_required = enforceai_required_raw.strip().lower() in {"1", "true", "yes"}
    if not enforceai_required:
        return

    db_path = (_os.environ.get("ENFORCEAI_DB_PATH") or "").strip()
    if not db_path:
        raise RuntimeError(
            "ENFORCEAI_REQUIRED=true but ENFORCEAI_DB_PATH is not set; refusing to start"
        )

    scopes_catalog_path = (
        (_os.environ.get("ENFORCEAI_SCOPES_CATALOG_PATH") or "").strip()
        or (_os.environ.get("SCOPES_CATALOG_PATH") or "").strip()
    )
    if not scopes_catalog_path:
        raise RuntimeError(
            "ENFORCEAI_REQUIRED=true but ENFORCEAI_SCOPES_CATALOG_PATH is not set; refusing to start"
        )

    try:
        EnforceAIDataLayer(db_path=Path(db_path)).initialize()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ENFORCEAI_REQUIRED=true but EnforceAI DB initialization failed; refusing to start"
        ) from exc


_enforce_enforceai_required()

# Create FastAPI app
app = FastAPI(
    title="Simplified Auth Server",
    description="Authentication server for validating JWT tokens against Amazon Cognito with header-based configuration",
    version="0.1.0"
)

# Add metrics collection middleware
add_auth_metrics_middleware(app)
app.include_router(core_router)
app.include_router(oauth2_router)
app.include_router(internal_router)
app.include_router(validate_router)
app.state.session_secret_key = SECRET_KEY
app.state.session_signer = signer
add_enforceai_csrf_middleware(
    app,
    secret_key=SECRET_KEY,
    session_cookie_name=SESSION_COOKIE_NAME,
    signer=signer,
    csrf_token_max_age_seconds=CSRF_TOKEN_MAX_AGE_SECONDS,
)

try:
    if _os.environ.get("ENFORCEAI_DB_PATH"):
        enforceai_management_router = load_enforceai_management_router()
        if enforceai_management_router is not None:
            app.include_router(enforceai_management_router)
    else:
        logger.info("ENFORCEAI_DB_PATH not set; skipping EnforceAI management routes")
except Exception:  # noqa: BLE001 - best-effort; server should still start
    logger.exception("Failed to mount EnforceAI management routes")

try:
    from auth_server.enforceai.errors import (  # type: ignore[import-not-found]
        DependencyUnavailableError,
        EnforceAIError,
    )
except Exception:  # noqa: BLE001
    DependencyUnavailableError = None  # type: ignore[assignment]
    EnforceAIError = None  # type: ignore[assignment]


if EnforceAIError is not None:

    @app.exception_handler(EnforceAIError)  # type: ignore[arg-type]
    async def _handle_enforceai_error(
        request: Request,
        exc: Any,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.public_message},
        )


@app.on_event("startup")
def _seed_enforceai_password_admin_user() -> None:
    db_path_raw = _os.environ.get("ENFORCEAI_DB_PATH")
    if not db_path_raw:
        return

    admin_username = _os.environ.get("ADMIN_USER", "admin").strip() or "admin"
    admin_password = _os.environ.get("ADMIN_PASSWORD")
    if not admin_password or not admin_password.strip():
        logger.warning("ADMIN_PASSWORD missing; skipping EnforceAI admin user seeding")
        return

    admin_email = _os.environ.get("ADMIN_EMAIL")
    if not admin_email or not admin_email.strip():
        admin_email = f"{admin_username}@local"

    try:
        db_path = Path(db_path_raw.strip())
        EnforceAIDataLayer(db_path=db_path).initialize()
        user_store = SqliteUserStore(db_path=db_path)

        existing = user_store.get_user_by_id(user_id=f"local|{admin_username}")
        if existing is not None:
            return

        from auth_server.enforceai.users.passwords import (
            hash_password,
        )

        password_hash = hash_password(admin_password).encoded
        user_store.create_local_user(
            username=admin_username,
            email=admin_email,
            password_hash=password_hash,
            role="admin",
        )
        logger.info(
            "Seeded EnforceAI admin user record for password login",
        )
    except Exception:
        logger.exception("Failed to seed EnforceAI password admin user")

# Create global validator instance
validator = SimplifiedCognitoValidator(
    secret_key=SECRET_KEY,
    jwt_issuer=JWT_ISSUER,
    jwt_audience=JWT_AUDIENCE,
)
app.state.validator = validator

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simplified Auth Server")

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the server to listen on (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port for the server to listen on (default: 8888)",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="Default AWS region (default: us-east-1)",
    )

    return parser.parse_args()

def main():
    """Run the server"""
    args = parse_arguments()
    
    # Update global validator with default region
    global validator
    validator = SimplifiedCognitoValidator(
        region=args.region,
        secret_key=SECRET_KEY,
        jwt_issuer=JWT_ISSUER,
        jwt_audience=JWT_AUDIENCE,
    )
    app.state.validator = validator
    
    logger.info(f"Starting simplified auth server on {args.host}:{args.port}")
    logger.info(f"Default region: {args.region}")
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
