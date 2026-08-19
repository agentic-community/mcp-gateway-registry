"""Factory for creating authentication provider instances."""

import ipaddress
import logging
import os
from urllib.parse import urlparse

from .auth0 import Auth0Provider
from .base import AuthProvider
from .cognito import CognitoProvider
from .entra import EntraIdProvider
from .keycloak import KeycloakProvider
from .okta import OktaProvider
from .pingfederate import PingFederateProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


# Hostnames that only ever mean "this machine". Kept separate from the
# misconfiguration check because loopback is the shipped default for local
# development and genuinely works for a client on the same host.
_LOOPBACK_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


def _host_is_loopback(host: str) -> bool:
    """Whether `host` refers to the local machine."""
    if host.lower() in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _unroutable_host_reason(host: str) -> str | None:
    """Return why `host` is unreachable from outside the deployment, or None.

    Loopback is deliberately NOT reported here: it is the shipped default for
    local development and is usable by a client on the same host, so the caller
    reports it separately at a lower level rather than as a misconfiguration.

    Args:
        host: Hostname or IP literal taken from a configured URL.

    Returns:
        A human-readable reason, or None if the host looks externally routable.
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Not an IP literal. A name with no dot is a container/service DNS name
        # (e.g. "keycloak"), which resolves only inside the deployment network.
        if host.lower() in _LOOPBACK_HOSTNAMES or "." in host:
            return None
        return (
            f"'{host}' is a single-label hostname, which normally means a container "
            f"or service DNS name that resolves only inside the deployment network"
        )

    if ip.is_loopback:
        return None
    if ip.is_private or ip.is_link_local or ip.is_reserved:
        return f"'{host}' is a private or non-routable address"
    return None


def _check_keycloak_external_url(external_url: str, *, configured: bool) -> None:
    """Log a startup diagnostic when the advertised Keycloak URL cannot work.

    Both the registry and the auth server publish this value in their OAuth
    discovery documents (`.well-known/oauth-authorization-server`). A value that
    resolves only inside the container network produces a well-formed metadata
    document that no external MCP client can act on. Nothing fails server-side,
    and the client sees a generic OAuth error, so the misconfiguration is close
    to invisible without this line.

    Args:
        external_url: The URL that will be advertised to external clients.
        configured: Whether KEYCLOAK_EXTERNAL_URL was actually set, as opposed
            to having fallen back to the internal KEYCLOAK_URL.
    """
    if not configured:
        logger.warning(
            f"KEYCLOAK_EXTERNAL_URL is not set, so OAuth discovery metadata will advertise "
            f"the internal URL '{external_url}'. External MCP clients cannot reach it. Set "
            f"KEYCLOAK_EXTERNAL_URL to the address clients use to reach this deployment."
        )
        return

    host = urlparse(external_url).hostname
    if not host:
        logger.warning(
            f"KEYCLOAK_EXTERNAL_URL '{external_url}' has no host component, so the OAuth "
            f"discovery metadata built from it will not be usable by external clients."
        )
        return

    reason = _unroutable_host_reason(host)
    if reason:
        logger.warning(
            f"KEYCLOAK_EXTERNAL_URL '{external_url}' is advertised in OAuth discovery "
            f"metadata, but {reason}. External MCP clients will fail discovery."
        )
    elif _host_is_loopback(host):
        logger.info(
            f"KEYCLOAK_EXTERNAL_URL '{external_url}' is a loopback address, so OAuth "
            f"discovery only works for clients on this host. Set it to the deployment's "
            f"externally reachable address to admit remote clients."
        )


def _parse_allowed_audiences(env_var: str) -> list[str]:
    """Parse a comma/whitespace-separated M2M audience allowlist from an env var.

    Returns the de-duplicated, order-preserving list of non-empty audience
    values. An unset or empty variable yields an empty list, which makes the
    provider fail closed (only the configured client ids are accepted).

    Args:
        env_var: Name of the environment variable to read.

    Returns:
        List of accepted M2M audience strings (may be empty).
    """
    raw = os.environ.get(env_var, "")
    seen: set[str] = set()
    audiences: list[str] = []
    for token in raw.replace(",", " ").split():
        value = token.strip()
        if value and value not in seen:
            seen.add(value)
            audiences.append(value)
    return audiences


def get_auth_provider(provider_type: str | None = None) -> AuthProvider:
    """Factory function to get the appropriate auth provider.

    Args:
        provider_type: Type of provider to create ('cognito', 'keycloak', or 'entra').
                      If None, uses AUTH_PROVIDER environment variable.

    Returns:
        AuthProvider instance configured for the specified provider

    Raises:
        ValueError: If provider type is unknown or required config is missing
    """
    provider_type = provider_type or os.environ.get("AUTH_PROVIDER", "cognito")

    logger.info(f"Creating authentication provider: {provider_type}")

    if provider_type == "keycloak":
        return _create_keycloak_provider()
    elif provider_type == "cognito":
        return _create_cognito_provider()
    elif provider_type == "entra":
        return _create_entra_provider()
    elif provider_type == "okta":
        return _create_okta_provider()
    elif provider_type == "auth0":
        return _create_auth0_provider()
    elif provider_type == "pingfederate":
        return _create_pingfederate_provider()
    else:
        raise ValueError(f"Unknown auth provider: {provider_type}")


def _create_keycloak_provider() -> KeycloakProvider:
    """Create and configure Keycloak provider."""
    # Required configuration
    keycloak_url = os.environ.get("KEYCLOAK_URL")
    keycloak_external_url = os.environ.get("KEYCLOAK_EXTERNAL_URL", keycloak_url)
    realm = os.environ.get("KEYCLOAK_REALM", "mcp-gateway")
    client_id = os.environ.get("KEYCLOAK_CLIENT_ID")
    client_secret = os.environ.get("KEYCLOAK_CLIENT_SECRET")

    # Optional M2M configuration
    m2m_client_id = os.environ.get("KEYCLOAK_M2M_CLIENT_ID")
    m2m_client_secret = os.environ.get("KEYCLOAK_M2M_CLIENT_SECRET")

    # Validate required configuration
    missing_vars = []
    if not keycloak_url:
        missing_vars.append("KEYCLOAK_URL")
    if not client_id:
        missing_vars.append("KEYCLOAK_CLIENT_ID")
    if not client_secret:
        missing_vars.append("KEYCLOAK_CLIENT_SECRET")

    if missing_vars:
        raise ValueError(
            f"Missing required Keycloak configuration: {', '.join(missing_vars)}. "
            "Please set these environment variables."
        )

    logger.info(
        f"Initializing Keycloak provider for realm '{realm}' at {keycloak_url} (external: {keycloak_external_url})"
    )
    _check_keycloak_external_url(
        keycloak_external_url,
        configured="KEYCLOAK_EXTERNAL_URL" in os.environ,
    )

    return KeycloakProvider(
        keycloak_url=keycloak_url,
        keycloak_external_url=keycloak_external_url,
        realm=realm,
        client_id=client_id,
        client_secret=client_secret,
        m2m_client_id=m2m_client_id,
        m2m_client_secret=m2m_client_secret,
    )


def _create_cognito_provider() -> CognitoProvider:
    """Create and configure Cognito provider."""
    # Required configuration
    user_pool_id = os.environ.get("COGNITO_USER_POOL_ID")
    client_id = os.environ.get("COGNITO_CLIENT_ID")
    client_secret = os.environ.get("COGNITO_CLIENT_SECRET")
    region = os.environ.get("AWS_REGION", "us-east-1")

    # Optional configuration
    domain = os.environ.get("COGNITO_DOMAIN")
    # Public IDE client (PR #1224). Its access tokens are also accepted so the
    # IDE OAuth login flow works alongside the web client. Empty = not used.
    ide_oauth_client_id = os.environ.get("IDE_OAUTH_CLIENT_ID") or None
    # M2M (client_credentials) app-client id allowlist. Comma/space-separated,
    # default-empty (fail closed): a machine token whose client_id is not listed
    # is rejected. Reuses the audience-allowlist parser (same shape).
    m2m_client_ids = _parse_allowed_audiences("COGNITO_M2M_CLIENT_IDS")

    # Validate required configuration
    missing_vars = []
    if not user_pool_id:
        missing_vars.append("COGNITO_USER_POOL_ID")
    if not client_id:
        missing_vars.append("COGNITO_CLIENT_ID")
    if not client_secret:
        missing_vars.append("COGNITO_CLIENT_SECRET")

    if missing_vars:
        raise ValueError(
            f"Missing required Cognito configuration: {', '.join(missing_vars)}. "
            "Please set these environment variables."
        )

    logger.info(
        f"Initializing Cognito provider for user pool '{user_pool_id}' in region '{region}'"
    )

    return CognitoProvider(
        user_pool_id=user_pool_id,
        client_id=client_id,
        client_secret=client_secret,
        region=region,
        domain=domain,
        ide_oauth_client_id=ide_oauth_client_id,
        m2m_client_ids=m2m_client_ids,
    )


def _create_entra_provider() -> EntraIdProvider:
    """Create and configure Entra ID provider."""
    # Required configuration
    tenant_id = os.environ.get("ENTRA_TENANT_ID")
    client_id = os.environ.get("ENTRA_CLIENT_ID")
    client_secret = os.environ.get("ENTRA_CLIENT_SECRET")

    # Optional PRM scope-advertisement config (issue #990). scope_format
    # defaults to "v2" (bare scopes, backward-compatible). application_id_uri
    # is the api://<app-id-or-uri> registered on the Entra app, used as the v1
    # scope prefix and accepted as a token audience.
    scope_format = os.environ.get("ENTRA_SCOPE_FORMAT")
    application_id_uri = os.environ.get("ENTRA_APPLICATION_ID_URI")

    # Validate required configuration
    missing_vars = []
    if not tenant_id:
        missing_vars.append("ENTRA_TENANT_ID")
    if not client_id:
        missing_vars.append("ENTRA_CLIENT_ID")
    if not client_secret:
        missing_vars.append("ENTRA_CLIENT_SECRET")

    if missing_vars:
        raise ValueError(
            f"Missing required Entra ID configuration: {', '.join(missing_vars)}. "
            "Please set these environment variables."
        )

    logger.info(
        f"Initializing Entra ID provider for tenant '{tenant_id}' "
        f"(scope_format={scope_format or 'v2'})"
    )

    return EntraIdProvider(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        scope_format=scope_format,
        application_id_uri=application_id_uri,
    )


def _create_okta_provider() -> OktaProvider:
    """Create and configure Okta provider."""
    okta_domain = os.environ.get("OKTA_DOMAIN")
    client_id = os.environ.get("OKTA_CLIENT_ID")
    client_secret = os.environ.get("OKTA_CLIENT_SECRET")
    m2m_client_id = os.environ.get("OKTA_M2M_CLIENT_ID")
    m2m_client_secret = os.environ.get("OKTA_M2M_CLIENT_SECRET")
    m2m_allowed_audiences = _parse_allowed_audiences("OKTA_M2M_ALLOWED_AUDIENCES")

    missing_vars = []
    if not okta_domain:
        missing_vars.append("OKTA_DOMAIN")
    if not client_id:
        missing_vars.append("OKTA_CLIENT_ID")
    if not client_secret:
        missing_vars.append("OKTA_CLIENT_SECRET")

    if missing_vars:
        raise ValueError(
            f"Missing required Okta configuration: {', '.join(missing_vars)}. "
            "Please set these environment variables."
        )

    logger.info(f"Initializing Okta provider for domain '{okta_domain}'")

    return OktaProvider(
        okta_domain=okta_domain,
        client_id=client_id,
        client_secret=client_secret,
        m2m_client_id=m2m_client_id,
        m2m_client_secret=m2m_client_secret,
        m2m_allowed_audiences=m2m_allowed_audiences,
    )


def _create_auth0_provider() -> Auth0Provider:
    """Create and configure Auth0 provider."""
    # Required configuration
    domain = os.environ.get("AUTH0_DOMAIN")
    client_id = os.environ.get("AUTH0_CLIENT_ID")
    client_secret = os.environ.get("AUTH0_CLIENT_SECRET")

    # Optional configuration
    audience = os.environ.get("AUTH0_AUDIENCE")
    m2m_client_id = os.environ.get("AUTH0_M2M_CLIENT_ID")
    m2m_client_secret = os.environ.get("AUTH0_M2M_CLIENT_SECRET")
    groups_claim = os.environ.get("AUTH0_GROUPS_CLAIM", "https://mcp-gateway/groups")

    # Validate required configuration
    missing_vars = []
    if not domain:
        missing_vars.append("AUTH0_DOMAIN")
    if not client_id:
        missing_vars.append("AUTH0_CLIENT_ID")
    if not client_secret:
        missing_vars.append("AUTH0_CLIENT_SECRET")

    if missing_vars:
        raise ValueError(
            f"Missing required Auth0 configuration: {', '.join(missing_vars)}. "
            "Please set these environment variables."
        )

    logger.info(f"Initializing Auth0 provider for domain '{domain}'")

    return Auth0Provider(
        domain=domain,
        client_id=client_id,
        client_secret=client_secret,
        audience=audience,
        m2m_client_id=m2m_client_id,
        m2m_client_secret=m2m_client_secret,
        groups_claim=groups_claim,
    )


def _create_pingfederate_provider() -> PingFederateProvider:
    """Create and configure PingFederate provider."""
    base_url = os.environ.get("PINGFEDERATE_BASE_URL")
    client_id = os.environ.get("PINGFEDERATE_CLIENT_ID")
    client_secret = os.environ.get("PINGFEDERATE_CLIENT_SECRET")
    m2m_client_id = os.environ.get("PINGFEDERATE_M2M_CLIENT_ID")
    m2m_client_secret = os.environ.get("PINGFEDERATE_M2M_CLIENT_SECRET")
    application_id_uri = os.environ.get("PINGFEDERATE_APPLICATION_ID_URI")
    groups_claim = os.environ.get("PINGFEDERATE_GROUPS_CLAIM", "groups")
    m2m_allowed_audiences = _parse_allowed_audiences("PINGFEDERATE_M2M_ALLOWED_AUDIENCES")

    missing_vars = []
    if not base_url:
        missing_vars.append("PINGFEDERATE_BASE_URL")
    if not client_id:
        missing_vars.append("PINGFEDERATE_CLIENT_ID")
    if not client_secret:
        missing_vars.append("PINGFEDERATE_CLIENT_SECRET")

    if missing_vars:
        raise ValueError(
            f"Missing required PingFederate configuration: {', '.join(missing_vars)}. "
            "Please set these environment variables."
        )

    logger.info(f"Initializing PingFederate provider for base URL '{base_url}'")

    return PingFederateProvider(
        base_url=base_url,
        client_id=client_id,
        client_secret=client_secret,
        m2m_client_id=m2m_client_id,
        m2m_client_secret=m2m_client_secret,
        application_id_uri=application_id_uri,
        groups_claim=groups_claim,
        m2m_allowed_audiences=m2m_allowed_audiences,
    )


def _get_provider_health_info() -> dict:
    """Get health information for the current provider."""
    try:
        provider = get_auth_provider()
        if hasattr(provider, "get_provider_info"):
            return provider.get_provider_info()
        else:
            return {
                "provider_type": os.environ.get("AUTH_PROVIDER", "cognito"),
                "status": "unknown",
            }
    except Exception as e:
        logger.error(f"Failed to get provider health info: {e}")
        return {
            "provider_type": os.environ.get("AUTH_PROVIDER", "cognito"),
            "status": "error",
            "error": str(e),
        }
