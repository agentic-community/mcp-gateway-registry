from __future__ import annotations

import hashlib
import logging
import os
import secrets
import urllib.parse
import uuid
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from typing import Any, Optional

import httpx
import jwt
from fastapi import (
    APIRouter,
    Cookie,
    HTTPException,
    Request,
)
from fastapi.responses import (
    RedirectResponse,
)
from itsdangerous import (
    BadSignature,
    SignatureExpired,
)

from gateway_session import (
    build_session_cookie_payload,
    normalize_session_data,
)
from ..enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from ..enforceai.stores.sqlite.session_store import (
    SqliteSessionStore,
)
from ..enforceai.stores.sqlite.user_store import (
    SqliteUserStore,
)
from .oauth2_context import (
    OAUTH2_CONFIG,
    SESSION_COOKIE_NAME,
    signer,
)

try:
    from ..providers.cognito_validator import (
        SimplifiedCognitoValidator,
    )
except ImportError:  # pragma: no cover
    from providers.cognito_validator import (  # type: ignore[no-redef]
        SimplifiedCognitoValidator,
    )

logger = logging.getLogger(__name__)

router = APIRouter()


def _hash_username(
    username: str,
) -> str:
    if not username:
        return "anonymous"
    return f"user_{hashlib.sha256(username.encode()).hexdigest()[:8]}"


def _normalize_string_list(
    value: object,
) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def get_enabled_providers() -> list[dict[str, str]]:
    """Get list of enabled OAuth2 providers, filtered by AUTH_PROVIDER env var if set."""
    enabled: list[dict[str, str]] = []

    auth_provider_env = os.getenv("AUTH_PROVIDER")

    yaml_enabled_providers: list[str] = []
    for provider_name, config in OAUTH2_CONFIG.get("providers", {}).items():
        if config.get("enabled", False):
            yaml_enabled_providers.append(provider_name)

    if auth_provider_env:
        logger.info(f"AUTH_PROVIDER is set to '{auth_provider_env}', filtering providers accordingly")

        if auth_provider_env not in OAUTH2_CONFIG.get("providers", {}):
            logger.error(
                f"AUTH_PROVIDER '{auth_provider_env}' not found in oauth2_providers.yml configuration",
            )
            return []

        provider_config = OAUTH2_CONFIG["providers"][auth_provider_env]
        if not provider_config.get("enabled", False):
            logger.warning(
                f"AUTH_PROVIDER '{auth_provider_env}' is set but this provider is disabled in oauth2_providers.yml",
            )
            logger.warning(
                f"To fix this, either set AUTH_PROVIDER to one of the enabled providers: {yaml_enabled_providers} "
                f"or enable '{auth_provider_env}' in oauth2_providers.yml",
            )
            return []

        filtered_providers = [provider for provider in yaml_enabled_providers if provider != auth_provider_env]
        if filtered_providers:
            logger.warning(
                f"AUTH_PROVIDER override: Filtering out enabled providers {filtered_providers} "
                f"- only showing '{auth_provider_env}'",
            )
            logger.warning("To show all enabled providers, remove the AUTH_PROVIDER environment variable")
    else:
        logger.info("AUTH_PROVIDER not set, returning all enabled providers from config")

    for provider_name, config in OAUTH2_CONFIG.get("providers", {}).items():
        if config.get("enabled", False):
            if auth_provider_env and provider_name != auth_provider_env:
                logger.debug(f"Skipping provider '{provider_name}' due to AUTH_PROVIDER filter")
                continue

            enabled.append(
                {
                    "name": provider_name,
                    "display_name": config.get("display_name", provider_name.title()),
                },
            )
            logger.debug(f"Enabled provider: {provider_name}")

    logger.info(f"Returning {len(enabled)} enabled providers: {[p['name'] for p in enabled]}")
    return enabled


@router.get("/oauth2/providers")
async def get_oauth2_providers() -> dict[str, object]:
    """Get list of enabled OAuth2 providers for the login page."""
    try:
        auth_provider_env = os.getenv("AUTH_PROVIDER")
        logger.info(f"Debug: AUTH_PROVIDER environment variable = '{auth_provider_env}'")

        providers = get_enabled_providers()
        return {"providers": providers}
    except Exception as exc:
        logger.error(f"Error getting OAuth2 providers: {exc}")
        return {"providers": [], "error": str(exc)}


@router.get("/oauth2/login/{provider}")
async def oauth2_login(
    provider: str,
    request: Request,
    redirect_uri: Optional[str] = None,
) -> RedirectResponse:
    """Initiate OAuth2 login flow."""
    try:
        if provider not in OAUTH2_CONFIG.get("providers", {}):
            raise HTTPException(status_code=404, detail=f"Provider {provider} not found")

        provider_config = OAUTH2_CONFIG["providers"][provider]
        if not provider_config.get("enabled", False):
            raise HTTPException(status_code=400, detail=f"Provider {provider} is disabled")

        state = secrets.token_urlsafe(32)

        session_data = {
            "state": state,
            "provider": provider,
            "redirect_uri": redirect_uri or OAUTH2_CONFIG.get("registry", {}).get("success_redirect", "/"),
        }

        temp_session = signer.dumps(session_data)

        auth_server_external_url = os.environ.get("AUTH_SERVER_EXTERNAL_URL")
        if auth_server_external_url:
            auth_server_url = auth_server_external_url.rstrip("/")
            logger.info(f"Using configured AUTH_SERVER_EXTERNAL_URL: {auth_server_url}")
        else:
            host = request.headers.get("host", "localhost:8888")
            scheme = (
                "https"
                if request.headers.get("x-forwarded-proto") == "https"
                or request.url.scheme == "https"
                else "http"
            )

            if "localhost" in host and ":" not in host:
                auth_server_url = f"{scheme}://localhost:8888"
            else:
                auth_server_url = f"{scheme}://{host}"

            logger.warning(f"AUTH_SERVER_EXTERNAL_URL not set, using dynamic URL: {auth_server_url}")

        callback_uri = f"{auth_server_url}/oauth2/callback/{provider}"
        logger.info(f"OAuth2 callback URI: {callback_uri}")

        auth_params = {
            "client_id": provider_config["client_id"],
            "response_type": provider_config["response_type"],
            "scope": " ".join(provider_config["scopes"]),
            "state": state,
            "redirect_uri": callback_uri,
        }

        auth_url = f"{provider_config['auth_url']}?{urllib.parse.urlencode(auth_params)}"

        response = RedirectResponse(url=auth_url, status_code=302)
        response.set_cookie(
            key="oauth2_temp_session",
            value=temp_session,
            max_age=600,
            httponly=True,
            samesite="lax",
        )

        logger.info(f"Initiated OAuth2 login for provider {provider}")
        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error initiating OAuth2 login for {provider}: {exc}")
        error_url = OAUTH2_CONFIG.get("registry", {}).get("error_redirect", "/login")
        return RedirectResponse(url=f"{error_url}?error=oauth2_init_failed", status_code=302)


@router.get("/oauth2/callback/{provider}")
async def oauth2_callback(
    provider: str,
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    oauth2_temp_session: Optional[str] = Cookie(None),
) -> RedirectResponse:
    """Handle OAuth2 callback and create user session."""
    try:
        if error:
            logger.warning(f"OAuth2 error from {provider}: {error}")
            error_url = OAUTH2_CONFIG.get("registry", {}).get("error_redirect", "/login")
            return RedirectResponse(url=f"{error_url}?error=oauth2_error&details={error}", status_code=302)

        if not code or not state or not oauth2_temp_session:
            raise HTTPException(status_code=400, detail="Missing required OAuth2 parameters")

        try:
            temp_session_data = signer.loads(oauth2_temp_session, max_age=600)
        except (SignatureExpired, BadSignature) as exc:
            raise HTTPException(status_code=400, detail="Invalid or expired OAuth2 session") from exc

        if state != temp_session_data.get("state"):
            raise HTTPException(status_code=400, detail="Invalid state parameter")

        if provider != temp_session_data.get("provider"):
            raise HTTPException(status_code=400, detail="Provider mismatch")

        provider_config = OAUTH2_CONFIG["providers"][provider]

        auth_server_external_url = os.environ.get("AUTH_SERVER_EXTERNAL_URL")
        if auth_server_external_url:
            auth_server_url = auth_server_external_url.rstrip("/")
            logger.info(f"Using configured AUTH_SERVER_EXTERNAL_URL for token exchange: {auth_server_url}")
        else:
            host = request.headers.get("host", "localhost:8888")
            scheme = (
                "https"
                if request.headers.get("x-forwarded-proto") == "https"
                or request.url.scheme == "https"
                else "http"
            )

            if "localhost" in host and ":" not in host:
                auth_server_url = f"{scheme}://localhost:8888"
            else:
                auth_server_url = f"{scheme}://{host}"

            logger.warning(
                f"AUTH_SERVER_EXTERNAL_URL not set, using dynamic URL for token exchange: {auth_server_url}",
            )

        token_data = await exchange_code_for_token(provider, code, provider_config, auth_server_url)
        logger.info(f"Token data keys: {list(token_data.keys())}")

        user_info: dict[str, Any] | None = None

        if provider in ["cognito", "keycloak"]:
            try:
                if provider == "cognito":
                    user_pool_id = os.environ.get("COGNITO_USER_POOL_ID")
                    client_id = provider_config["client_id"]
                    region = os.environ.get("AWS_REGION", "us-east-1")

                    if user_pool_id and client_id:
                        validator = SimplifiedCognitoValidator(region)
                        token_validation = validator.validate_token(
                            token_data["access_token"],
                            user_pool_id,
                            client_id,
                            region,
                        )

                        logger.info(f"Token validation result: {token_validation}")

                        mapped_user = {
                            "username": token_validation.get("username"),
                            "email": token_validation.get("username"),
                            "name": token_validation.get("username"),
                            "groups": token_validation.get("groups", []),
                        }
                        logger.info(f"User extracted from JWT token: {mapped_user}")
                    else:
                        logger.warning(
                            "Missing Cognito configuration for JWT validation, falling back to userInfo",
                        )
                        raise ValueError("Missing Cognito config")
                elif provider == "keycloak":
                    if "id_token" in token_data:
                        id_token_claims = jwt.decode(token_data["id_token"], options={"verify_signature": False})
                        logger.info(f"ID token claims: {id_token_claims}")

                        mapped_user = {
                            "username": id_token_claims.get("preferred_username") or id_token_claims.get("sub"),
                            "email": id_token_claims.get("email"),
                            "name": id_token_claims.get("name") or id_token_claims.get("given_name"),
                            "groups": id_token_claims.get("groups", []),
                        }
                        logger.info(f"User extracted from Keycloak ID token: {mapped_user}")
                    else:
                        logger.warning("No ID token found in Keycloak response, falling back to userInfo")
                        raise ValueError("Missing ID token")

            except Exception as exc:
                logger.warning(f"JWT token validation failed: {exc}, falling back to userInfo endpoint")
                user_info = await get_user_info(token_data["access_token"], provider_config)
                logger.info(f"Raw user info from {provider}: {user_info}")
                mapped_user = map_user_info(user_info, provider_config)
                logger.info(f"Mapped user info from userInfo: {mapped_user}")
        elif provider == "entra":
            try:
                if "id_token" in token_data:
                    id_token_claims = jwt.decode(token_data["id_token"], options={"verify_signature": False})
                    logger.info(f"Entra ID token claims: {id_token_claims}")

                    groups = id_token_claims.get("groups", [])
                    if not groups:
                        groups = id_token_claims.get("roles", [])

                    mapped_user = {
                        "username": id_token_claims.get("preferred_username")
                        or id_token_claims.get("email")
                        or id_token_claims.get("upn")
                        or id_token_claims.get("sub"),
                        "email": id_token_claims.get("email") or id_token_claims.get("preferred_username"),
                        "name": id_token_claims.get("name") or id_token_claims.get("given_name"),
                        "groups": groups,
                    }
                    logger.info(f"User extracted from Entra ID token: {mapped_user}")
                else:
                    logger.warning("No ID token found in Entra ID response, falling back to userInfo")
                    raise ValueError("Missing ID token")

            except Exception as exc:
                logger.warning(f"Entra ID token parsing failed: {exc}, falling back to userInfo endpoint")
                user_info = await get_user_info(token_data["access_token"], provider_config)
                logger.info(f"Raw user info from {provider}: {user_info}")
                mapped_user = map_user_info(user_info, provider_config)
                logger.info(f"Mapped user info from userInfo: {mapped_user}")
        else:
            user_info = await get_user_info(token_data["access_token"], provider_config)
            logger.info(f"Raw user info from {provider}: {user_info}")
            mapped_user = map_user_info(user_info, provider_config)
            logger.info(f"Mapped user info: {mapped_user}")

        mapped_groups = mapped_user.get("groups", [])
        if not isinstance(mapped_groups, list):
            mapped_groups = []
            mapped_user["groups"] = mapped_groups

        if not mapped_groups:
            default_groups = _normalize_string_list(provider_config.get("default_groups"))
            if default_groups:
                mapped_user["groups"] = default_groups
                logger.info(
                    "Applied OAuth2 default groups for provider '%s': %s",
                    provider,
                    default_groups,
                )

        issuer = mapped_user.get("iss") or mapped_user.get("issuer")
        subject = mapped_user.get("sub") or mapped_user.get("subject")

        if provider == "google":
            issuer = issuer or "https://accounts.google.com"
            if subject is None and isinstance(user_info, dict):
                raw_subject = user_info.get("sub") or user_info.get("id")
                if isinstance(raw_subject, str) and raw_subject.strip():
                    subject = raw_subject.strip()

        if issuer is None or subject is None:
            try:
                if "id_token" in token_data:
                    id_claims = jwt.decode(token_data["id_token"], options={"verify_signature": False})
                    issuer = issuer or id_claims.get("iss")
                    subject = subject or id_claims.get("sub")
            except Exception:
                issuer = issuer
                subject = subject

        if issuer is None or subject is None:
            try:
                access_claims = jwt.decode(token_data["access_token"], options={"verify_signature": False})
                issuer = issuer or access_claims.get("iss")
                subject = subject or access_claims.get("sub")
            except Exception:
                issuer = issuer
                subject = subject

        session_id = str(uuid.uuid4())
        user_id_value = None
        if isinstance(issuer, str) and issuer.strip() and isinstance(subject, str) and subject.strip():
            user_id_value = f"{issuer.strip()}|{subject.strip()}"

        db_path_raw = os.environ.get("ENFORCEAI_DB_PATH")
        if user_id_value is None:
            email_value = mapped_user.get("email")
            if isinstance(email_value, str) and email_value.strip():
                user_id_value = f"{provider}|{email_value.strip().lower()}"
            else:
                username_value = mapped_user.get("username")
                if isinstance(username_value, str) and username_value.strip():
                    user_id_value = f"{provider}|{username_value.strip()}"

        if user_id_value and db_path_raw:
            try:
                db_path = Path(db_path_raw.strip())
                EnforceAIDataLayer(db_path=db_path).initialize()
                store = SqliteSessionStore(db_path=db_path)
                store.create_session(
                    session_id=session_id,
                    user_id=user_id_value,
                    auth_method="oidc",
                    expires_at=datetime.now(timezone.utc).replace(microsecond=0)
                    + timedelta(seconds=OAUTH2_CONFIG.get("session", {}).get("max_age_seconds", 28800)),
                )
                email_value = mapped_user.get("email")
                if isinstance(email_value, str) and email_value.strip():
                    groups_value = mapped_user.get("groups", [])
                    role_value = (
                        "admin"
                        if isinstance(groups_value, list) and "enforceai-admin" in groups_value
                        else "user"
                    )
                    SqliteUserStore(db_path=db_path).upsert_oidc_user(
                        user_id=user_id_value,
                        email=email_value.strip(),
                        role=role_value,
                    )
            except Exception:
                logger.exception(
                    "Failed to persist OAuth2 session (continuing without server-side invalidation)",
                )

        session_data = build_session_cookie_payload(
            username=mapped_user["username"],
            email=mapped_user.get("email"),
            name=mapped_user.get("name"),
            groups=mapped_user.get("groups", []),
            provider=provider,
            legacy_auth_method="oauth2",
            max_age_seconds=OAUTH2_CONFIG.get("session", {}).get("max_age_seconds", 28800),
            session_id=session_id,
            user_id=user_id_value,
        )
        if user_id_value and isinstance(issuer, str) and issuer.strip() and isinstance(subject, str) and subject.strip():
            session_data["iss"] = issuer.strip()
            session_data["sub"] = subject.strip()
            session_data["user_id"] = user_id_value

        registry_session = signer.dumps(session_data)

        redirect_url = temp_session_data.get(
            "redirect_uri",
            OAUTH2_CONFIG.get("registry", {}).get("success_redirect", "/"),
        )
        response = RedirectResponse(url=redirect_url, status_code=302)

        x_forwarded_proto = request.headers.get("x-forwarded-proto", "")
        is_https = x_forwarded_proto == "https" or request.url.scheme == "https"

        cookie_secure_config = OAUTH2_CONFIG.get("session", {}).get("secure", False)
        cookie_secure = cookie_secure_config and is_https
        cookie_samesite = OAUTH2_CONFIG.get("session", {}).get("samesite", "lax")
        cookie_domain = OAUTH2_CONFIG.get("session", {}).get("domain", "")

        if not cookie_domain or cookie_domain == "${SESSION_COOKIE_DOMAIN}":
            cookie_domain = None
            logger.info("No cookie domain configured - cookie will be set for exact host only")
        else:
            logger.info(f"Using explicitly configured cookie domain: {cookie_domain}")

        logger.info(
            "Auth server setting session cookie: secure=%s (config=%s, is_https=%s), "
            "samesite=%s, domain=%s, x-forwarded-proto=%s, request_scheme=%s",
            cookie_secure,
            cookie_secure_config,
            is_https,
            cookie_samesite,
            cookie_domain or "not set",
            x_forwarded_proto,
            request.url.scheme,
        )

        cookie_params = {
            "key": SESSION_COOKIE_NAME,
            "value": registry_session,
            "max_age": OAUTH2_CONFIG.get("session", {}).get("max_age_seconds", 28800),
            "httponly": OAUTH2_CONFIG.get("session", {}).get("httponly", True),
            "samesite": cookie_samesite,
            "secure": cookie_secure,
            "path": "/",
        }

        if cookie_domain:
            cookie_params["domain"] = cookie_domain

        response.set_cookie(**cookie_params)
        response.delete_cookie("oauth2_temp_session")

        logger.info(f"Successfully authenticated user {_hash_username(mapped_user['username'])} via {provider}")
        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error in OAuth2 callback for {provider}: {exc}")
        error_url = OAUTH2_CONFIG.get("registry", {}).get("error_redirect", "/login")
        return RedirectResponse(url=f"{error_url}?error=oauth2_callback_failed", status_code=302)


async def exchange_code_for_token(
    provider: str,
    code: str,
    provider_config: dict,
    auth_server_url: Optional[str] = None,
) -> dict:
    """Exchange authorization code for access token."""
    if auth_server_url is None:
        auth_server_url = os.environ.get("AUTH_SERVER_URL", "http://localhost:8888")

    async with httpx.AsyncClient() as client:
        token_data = {
            "grant_type": provider_config["grant_type"],
            "client_id": provider_config["client_id"],
            "client_secret": provider_config["client_secret"],
            "code": code,
            "redirect_uri": f"{auth_server_url}/oauth2/callback/{provider}",
        }

        headers = {"Accept": "application/json"}
        if provider == "github":
            headers["Accept"] = "application/json"

        response = await client.post(
            provider_config["token_url"],
            data=token_data,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


async def get_user_info(
    access_token: str,
    provider_config: dict,
) -> dict:
    """Get user information from OAuth2 provider."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}

        response = await client.get(
            provider_config["user_info_url"],
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


def map_user_info(
    user_info: dict,
    provider_config: dict,
) -> dict:
    """Map provider-specific user info to our standard format."""
    mapped = {
        "username": user_info.get(provider_config["username_claim"]),
        "email": user_info.get(provider_config["email_claim"]),
        "name": user_info.get(provider_config["name_claim"]),
        "groups": [],
    }

    groups_claim = provider_config.get("groups_claim")
    logger.info(f"Looking for groups using claim: {groups_claim}")
    logger.info(f"Available claims in user_info: {list(user_info.keys())}")

    if groups_claim and groups_claim in user_info:
        groups = user_info[groups_claim]
        if isinstance(groups, list):
            mapped["groups"] = groups
        elif isinstance(groups, str):
            mapped["groups"] = [groups]
        logger.info(f"Found groups via {groups_claim}: {mapped['groups']}")
    else:
        for possible_group_claim in ["cognito:groups", "groups", "custom:groups"]:
            if possible_group_claim in user_info:
                groups = user_info[possible_group_claim]
                if isinstance(groups, list):
                    mapped["groups"] = groups
                elif isinstance(groups, str):
                    mapped["groups"] = [groups]
                logger.info(f"Found groups via alternative claim {possible_group_claim}: {mapped['groups']}")
                break

        if not mapped["groups"]:
            logger.warning(f"No groups found in user_info. Available fields: {list(user_info.keys())}")

    return mapped


@router.get("/oauth2/logout/{provider}")
async def oauth2_logout(
    provider: str,
    request: Request,
    redirect_uri: Optional[str] = None,
) -> RedirectResponse:
    """Initiate OAuth2 logout flow to clear provider session."""
    try:
        if provider not in OAUTH2_CONFIG.get("providers", {}):
            raise HTTPException(status_code=404, detail=f"Provider {provider} not found")

        provider_config = OAUTH2_CONFIG["providers"][provider]
        logout_url = provider_config.get("logout_url")

        if not logout_url:
            redirect_url = redirect_uri or OAUTH2_CONFIG.get("registry", {}).get("success_redirect", "/login")
            response = RedirectResponse(url=redirect_url, status_code=302)
            response.delete_cookie(SESSION_COOKIE_NAME)
            return response

        full_redirect_uri = redirect_uri or "/logout"
        if not full_redirect_uri.startswith("http"):
            registry_base = os.environ.get("REGISTRY_URL")
            if not registry_base:
                referer = request.headers.get("referer", "")
                if referer:
                    from urllib.parse import urlparse

                    parsed = urlparse(referer)
                    registry_base = f"{parsed.scheme}://{parsed.netloc}"
                else:
                    registry_base = "http://localhost"

            full_redirect_uri = f"{registry_base.rstrip('/')}{full_redirect_uri}"

        logout_params = {
            "client_id": provider_config["client_id"],
            "logout_uri": full_redirect_uri,
        }

        logout_redirect_url = f"{logout_url}?{urllib.parse.urlencode(logout_params)}"

        logger.info(f"Redirecting to {provider} logout: {logout_redirect_url}")
        response = RedirectResponse(url=logout_redirect_url, status_code=302)

        cookie_value = request.cookies.get(SESSION_COOKIE_NAME)
        if cookie_value:
            try:
                session_payload = signer.loads(cookie_value, max_age=28800)
                normalized = normalize_session_data(
                    session_payload,
                    default_provider="local",
                    max_age_seconds=28800,
                )
                db_path_raw = os.environ.get("ENFORCEAI_DB_PATH")
                if db_path_raw:
                    db_path = Path(db_path_raw.strip())
                    EnforceAIDataLayer(db_path=db_path).initialize()
                    SqliteSessionStore(db_path=db_path).revoke_session(
                        session_id=normalized.session_id,
                        revoked_reason="logout",
                    )
            except Exception:
                logger.exception("Failed to revoke server-side session on oauth2 logout")

        response.delete_cookie(SESSION_COOKIE_NAME)
        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error initiating logout for {provider}: {exc}")
        redirect_url = redirect_uri or OAUTH2_CONFIG.get("registry", {}).get("success_redirect", "/login")
        return RedirectResponse(url=redirect_url, status_code=302)
