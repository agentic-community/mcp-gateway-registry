"""Authentication provider package for MCP Gateway Registry."""

from .auth0 import Auth0Provider
from .base import AuthProvider
from .cognito import CognitoProvider
from .entra import EntraIdProvider
from .factory import get_auth_provider
from .keycloak import KeycloakProvider
from .okta import OktaProvider

__all__ = [
    "Auth0Provider",
    "AuthProvider",
    "CognitoProvider",
    "EntraIdProvider",
    "KeycloakProvider",
    "OktaProvider",
    "get_auth_provider",
]
