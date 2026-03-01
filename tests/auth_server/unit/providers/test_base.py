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
        # Return type should be Dict[str, Any] (or dict[str, Any] in Python 3.12+)
        return_str = str(sig.return_annotation).lower()
        assert "dict" in return_str

    def test_get_jwks_signature(self):
        """Test get_jwks has correct type hints."""
        import inspect

        from auth_server.providers.base import AuthProvider

        # Act
        sig = inspect.signature(AuthProvider.get_jwks)

        # Assert
        # Should return Dict[str, Any] (or dict[str, Any] in Python 3.12+)
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
