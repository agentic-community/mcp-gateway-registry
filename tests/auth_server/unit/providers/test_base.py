"""
Unit tests for auth_server/providers/base.py

Tests the abstract base class interface for authentication providers.
"""

import logging
from typing import Any

import pytest

logger = logging.getLogger(__name__)


# Mark all tests in this file
pytestmark = [pytest.mark.unit, pytest.mark.auth]


# =============================================================================
# BASE PROVIDER INTERFACE TESTS
# =============================================================================


class TestAuthProviderInterface:
    """Tests for AuthProvider abstract base class."""

    def test_auth_provider_is_abstract(self):
        """Test that AuthProvider is an abstract base class."""
        from auth_server.providers.base import AuthProvider

        # Act & Assert - cannot instantiate abstract class
        with pytest.raises(TypeError):
            AuthProvider()

    def test_auth_provider_has_required_methods(self):
        """Test that AuthProvider defines all required abstract methods."""
        import inspect

        from auth_server.providers.base import AuthProvider

        # Act
        abstract_methods = {
            name
            for name, method in inspect.getmembers(AuthProvider)
            if getattr(method, "__isabstractmethod__", False)
        }

        # Assert
        expected_methods = {
            "validate_token",
            "get_jwks",
            "exchange_code_for_token",
            "get_user_info",
            "get_auth_url",
            "get_logout_url",
            "refresh_token",
            "validate_m2m_token",
            "get_m2m_token",
            "authorization_server_metadata",
        }

        assert abstract_methods == expected_methods


class TestConcreteImplementation:
    """Tests for concrete implementation of AuthProvider."""

    def test_concrete_provider_implementation(self):
        """Test that a concrete provider implements all methods."""
        from auth_server.providers.base import AuthProvider

        # Arrange - create concrete implementation
        class TestProvider(AuthProvider):
            """Test implementation of AuthProvider."""

            def validate_token(self, token: str, **kwargs: Any) -> dict[str, Any]:
                return {"valid": True, "username": "test"}

            def get_jwks(self) -> dict[str, Any]:
                return {"keys": []}

            def exchange_code_for_token(self, code: str, redirect_uri: str) -> dict[str, Any]:
                return {"access_token": "test"}

            def get_user_info(self, access_token: str) -> dict[str, Any]:
                return {"username": "test"}

            def get_auth_url(self, redirect_uri: str, state: str, scope: str = None) -> str:
                return "https://auth.example.com/authorize"

            def get_logout_url(self, redirect_uri: str) -> str:
                return "https://auth.example.com/logout"

            def refresh_token(self, refresh_token: str) -> dict[str, Any]:
                return {"access_token": "new_token"}

            def validate_m2m_token(self, token: str) -> dict[str, Any]:
                return {"valid": True}

            def get_m2m_token(
                self, client_id: str = None, client_secret: str = None, scope: str = None
            ) -> dict[str, Any]:
                return {"access_token": "m2m_token"}

            def authorization_server_metadata(self) -> dict[str, Any]:
                return {
                    "issuer": "https://idp.example.com",
                    "authorization_endpoint": "https://idp.example.com/authorize",
                    "token_endpoint": "https://idp.example.com/token",
                    "jwks_uri": "https://idp.example.com/jwks",
                }

        # Act
        provider = TestProvider()

        # Assert - can call all methods
        assert provider.validate_token("token")["valid"] is True
        assert "keys" in provider.get_jwks()
        assert "access_token" in provider.exchange_code_for_token("code", "uri")
        assert "username" in provider.get_user_info("token")
        assert provider.get_auth_url("uri", "state").startswith("https://")
        assert provider.get_logout_url("uri").startswith("https://")
        assert "access_token" in provider.refresh_token("token")
        assert provider.validate_m2m_token("token")["valid"] is True
        assert "access_token" in provider.get_m2m_token()
        assert provider.authorization_server_metadata()["issuer"].startswith("https://")
        assert provider.is_delegated_user({"valid": True, "method": "test"}) is False

    @pytest.mark.asyncio
    async def test_provider_without_graph_recovery_preserves_claims(self):
        provider = TestProtectedResourceMetadataDefault()._make_provider()
        claims = {"valid": True, "method": "other", "groups": ["users"]}
        assert await provider.recover_bearer_groups(claims, "token") is None


class TestProtectedResourceMetadataDefault:
    """Tests for the default protected_resource_metadata() implementation."""

    def _make_provider(self, issuer: str = "https://idp.example.com"):
        from auth_server.providers.base import AuthProvider

        class _Provider(AuthProvider):
            def validate_token(self, token, **kwargs):
                return {}

            def get_jwks(self):
                return {}

            def exchange_code_for_token(self, code, redirect_uri):
                return {}

            def get_user_info(self, access_token):
                return {}

            def get_auth_url(self, redirect_uri, state, scope=None):
                return ""

            def get_logout_url(self, redirect_uri):
                return ""

            def refresh_token(self, refresh_token):
                return {}

            def validate_m2m_token(self, token):
                return {}

            def get_m2m_token(self, client_id=None, client_secret=None, scope=None):
                return {}

            def authorization_server_metadata(self):
                return {"issuer": issuer}

        return _Provider()

    def test_returns_rfc9728_required_fields(self):
        """PRM document includes resource, authorization_servers, scopes_supported, bearer_methods_supported."""
        provider = self._make_provider()

        document = provider.protected_resource_metadata(
            resource="https://gw.example.com",
            scopes_supported=["mcp-admin", "mcp-read"],
        )

        assert document["resource"] == "https://gw.example.com"
        assert document["authorization_servers"] == ["https://idp.example.com"]
        assert document["scopes_supported"] == ["mcp-admin", "mcp-read"]
        assert document["bearer_methods_supported"] == ["header"]

    def test_resource_documentation_optional(self):
        """resource_documentation is omitted when not provided, included when provided."""
        provider = self._make_provider()

        without = provider.protected_resource_metadata(
            resource="https://gw.example.com",
            scopes_supported=[],
        )
        assert "resource_documentation" not in without

        with_docs = provider.protected_resource_metadata(
            resource="https://gw.example.com",
            scopes_supported=[],
            resource_documentation="https://gw.example.com/docs/oauth",
        )
        assert with_docs["resource_documentation"] == "https://gw.example.com/docs/oauth"

    def test_scopes_supported_preserves_caller_order(self):
        """The route handler is responsible for sorting; the helper preserves order."""
        provider = self._make_provider()

        document = provider.protected_resource_metadata(
            resource="https://gw.example.com",
            scopes_supported=["zeta", "alpha", "mu"],
        )

        assert document["scopes_supported"] == ["zeta", "alpha", "mu"]

    def test_authorization_server_issuer_default_reads_metadata(self):
        """Default authorization_server_issuer() pulls from authorization_server_metadata()."""
        provider = self._make_provider(issuer="https://accounts.example.com/realm/x")

        assert provider.authorization_server_issuer() == "https://accounts.example.com/realm/x"

    def test_authorization_server_issuer_raises_when_missing(self):
        """If the AS metadata document lacks an issuer, default issuer() raises."""
        from auth_server.providers.base import AuthProvider

        class _NoIssuer(AuthProvider):
            def validate_token(self, token, **kwargs):
                return {}

            def get_jwks(self):
                return {}

            def exchange_code_for_token(self, code, redirect_uri):
                return {}

            def get_user_info(self, access_token):
                return {}

            def get_auth_url(self, redirect_uri, state, scope=None):
                return ""

            def get_logout_url(self, redirect_uri):
                return ""

            def refresh_token(self, refresh_token):
                return {}

            def validate_m2m_token(self, token):
                return {}

            def get_m2m_token(self, client_id=None, client_secret=None, scope=None):
                return {}

            def authorization_server_metadata(self):
                return {"authorization_endpoint": "x"}  # missing issuer

        with pytest.raises(ValueError, match="missing 'issuer'"):
            _NoIssuer().authorization_server_issuer()


class TestAuthProviderDocstrings:
    """Tests for documentation and interface contracts."""

    def test_validate_token_docstring(self):
        """Test validate_token method has proper documentation."""
        from auth_server.providers.base import AuthProvider

        # Act
        docstring = AuthProvider.validate_token.__doc__

        # Assert
        assert docstring is not None
        assert "validate" in docstring.lower()
        assert "token" in docstring.lower()

    def test_get_jwks_docstring(self):
        """Test get_jwks method has proper documentation."""
        from auth_server.providers.base import AuthProvider

        # Act
        docstring = AuthProvider.get_jwks.__doc__

        # Assert
        assert docstring is not None
        assert "jwks" in docstring.lower() or "key set" in docstring.lower()

    def test_exchange_code_for_token_docstring(self):
        """Test exchange_code_for_token method has proper documentation."""
        from auth_server.providers.base import AuthProvider

        # Act
        docstring = AuthProvider.exchange_code_for_token.__doc__

        # Assert
        assert docstring is not None
        assert "exchange" in docstring.lower() or "authorization" in docstring.lower()
        assert "code" in docstring.lower()

    def test_get_user_info_docstring(self):
        """Test get_user_info method has proper documentation."""
        from auth_server.providers.base import AuthProvider

        # Act
        docstring = AuthProvider.get_user_info.__doc__

        # Assert
        assert docstring is not None
        assert "user" in docstring.lower()
        assert "info" in docstring.lower()


class TestAuthProviderTypeHints:
    """Tests for type hints on abstract methods."""

    def test_validate_token_signature(self):
        """Test validate_token has correct type hints."""
        import inspect

        from auth_server.providers.base import AuthProvider

        # Act
        sig = inspect.signature(AuthProvider.validate_token)

        # Assert
        assert "token" in sig.parameters
        assert sig.parameters["token"].annotation is str
        # Return type should be Dict[str, Any] (or dict[str, Any] in Python 3.14+)
        return_str = str(sig.return_annotation).lower()
        assert "dict" in return_str

    def test_get_jwks_signature(self):
        """Test get_jwks has correct type hints."""
        import inspect

        from auth_server.providers.base import AuthProvider

        # Act
        sig = inspect.signature(AuthProvider.get_jwks)

        # Assert
        # Should return Dict[str, Any] (or dict[str, Any] in Python 3.14+)
        return_str = str(sig.return_annotation).lower()
        assert "dict" in return_str

    def test_exchange_code_for_token_signature(self):
        """Test exchange_code_for_token has correct type hints."""
        import inspect

        from auth_server.providers.base import AuthProvider

        # Act
        sig = inspect.signature(AuthProvider.exchange_code_for_token)

        # Assert
        assert "code" in sig.parameters
        assert "redirect_uri" in sig.parameters
        assert sig.parameters["code"].annotation is str
        assert sig.parameters["redirect_uri"].annotation is str
