from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from string import Template
from typing import Any

import yaml
from itsdangerous import (
    URLSafeTimedSerializer,
)

logger = logging.getLogger(__name__)

_ENV_TRUE_VALUES: set[str] = {"1", "true", "yes", "on"}

CSRF_TOKEN_MAX_AGE_SECONDS: int = int(os.environ.get("CSRF_TOKEN_MAX_AGE_SECONDS", "3600"))
SESSION_COOKIE_NAME: str = "mcp_gateway_session"

SECRET_KEY: str = os.environ.get("SECRET_KEY") or ""
if not SECRET_KEY.strip():
    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "No SECRET_KEY environment variable found. Using a randomly generated key. "
        "While this is more secure than a hardcoded default, it will change on restart. "
        "Set a permanent SECRET_KEY environment variable for production.",
    )

signer = URLSafeTimedSerializer(SECRET_KEY)


def _parse_env_bool(
    value: str,
) -> bool | None:
    candidate = value.strip().lower()
    if not candidate:
        return None

    if candidate in _ENV_TRUE_VALUES:
        return True

    if candidate in {"0", "false", "no", "off"}:
        return False

    return None


def _apply_provider_enabled_env(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Enable OAuth2 providers via `<PROVIDER>_ENABLED=true` env vars.

    Notes:
    - This is intentionally enable-only (truthy values). Many deploy configs set
      `*_ENABLED=false` by default, and we do not want those defaults to disable
      providers that are enabled in YAML.
    """
    providers = config.get("providers")
    if not isinstance(providers, dict):
        return config

    for provider_name, provider_config in providers.items():
        if not isinstance(provider_config, dict):
            continue

        env_key = f"{provider_name.upper()}_ENABLED"
        env_value = os.environ.get(env_key)
        if env_value is None:
            continue

        parsed = _parse_env_bool(env_value)
        if parsed is None:
            logger.warning(
                f"Ignoring {env_key}={env_value!r}: expected one of {sorted(_ENV_TRUE_VALUES)} "
                "(enable-only).",
            )
            continue

        if parsed is True:
            provider_config["enabled"] = True
            logger.info(f"Enabled OAuth2 provider '{provider_name}' via {env_key}=true")

    return config


def auto_derive_cognito_domain(
    user_pool_id: str,
) -> str:
    """Auto-derive Cognito domain from User Pool ID.

    Example: `us-east-1_KmP5A3La3` -> `us-east-1kmp5a3la3`
    """
    if not user_pool_id:
        return ""

    domain = user_pool_id.replace("_", "").lower()
    logger.info(f"Auto-derived Cognito domain '{domain}' from user pool ID '{user_pool_id}'")
    return domain


def substitute_env_vars(
    config: Any,
) -> Any:
    """Recursively substitute environment variables in configuration."""
    if isinstance(config, dict):
        return {key: substitute_env_vars(value) for key, value in config.items()}
    if isinstance(config, list):
        return [substitute_env_vars(item) for item in config]
    if isinstance(config, str) and "${" in config:
        try:
            if "COGNITO_DOMAIN:-auto" in config:
                cognito_domain = os.environ.get("COGNITO_DOMAIN")
                if not cognito_domain:
                    user_pool_id = os.environ.get("COGNITO_USER_POOL_ID", "")
                    cognito_domain = auto_derive_cognito_domain(user_pool_id)

                config = config.replace("${COGNITO_DOMAIN:-auto}", cognito_domain)

            template = Template(config)
            return template.substitute(os.environ)
        except KeyError as exc:
            logger.warning(f"Environment variable not found for template {config}: {exc}")
            return config
    return config


def load_oauth2_config() -> dict[str, Any]:
    """Load the OAuth2 providers configuration from `oauth2_providers.yml`."""
    try:
        oauth2_file = Path(__file__).resolve().parent.parent / "oauth2_providers.yml"
        config = yaml.safe_load(oauth2_file.read_text()) or {}

        processed_config = substitute_env_vars(config)
        if isinstance(processed_config, dict):
            processed_config = _apply_provider_enabled_env(processed_config)
            return processed_config
        return {"providers": {}, "session": {}, "registry": {}}
    except Exception as exc:
        logger.error(f"Failed to load OAuth2 configuration: {exc}")
        return {"providers": {}, "session": {}, "registry": {}}


OAUTH2_CONFIG: dict[str, Any] = load_oauth2_config()
