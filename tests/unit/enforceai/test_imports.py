"""
Unit tests ensuring EnforceAI package skeleton is importable under pytest.
"""

import pytest


@pytest.mark.unit
class TestEnforceAIImports:
    """Test suite for EnforceAI package imports."""

    def test_imports(self):
        import auth_server.enforceai  # noqa: F401
        import auth_server.enforceai.auth  # noqa: F401
        import auth_server.enforceai.auth.credentials  # noqa: F401
        import auth_server.enforceai.auth.dependency  # noqa: F401
        import auth_server.enforceai.auth.resolver  # noqa: F401
        import auth_server.enforceai.config  # noqa: F401
        import auth_server.enforceai.crypto  # noqa: F401
        import auth_server.enforceai.crypto.keyring  # noqa: F401
        import auth_server.enforceai.db  # noqa: F401
        import auth_server.enforceai.db.connection  # noqa: F401
        import auth_server.enforceai.db.data_layer  # noqa: F401
        import auth_server.enforceai.db.migrations  # noqa: F401
        import auth_server.enforceai.errors  # noqa: F401
        import auth_server.enforceai.identity  # noqa: F401
        import auth_server.enforceai.logging  # noqa: F401
        import auth_server.enforceai.models  # noqa: F401
        import auth_server.enforceai.models.agent  # noqa: F401
        import auth_server.enforceai.models.api_key  # noqa: F401
        import auth_server.enforceai.models.audit  # noqa: F401
        import auth_server.enforceai.models.revocation  # noqa: F401
        import auth_server.enforceai.providers  # noqa: F401
        import auth_server.enforceai.providers.api_key  # noqa: F401
        import auth_server.enforceai.providers.gateway_token  # noqa: F401
        import auth_server.enforceai.providers.oidc  # noqa: F401
        import auth_server.enforceai.secrets  # noqa: F401
        import auth_server.enforceai.secrets.pepper  # noqa: F401
        import auth_server.enforceai.tokens  # noqa: F401
        import auth_server.enforceai.tokens.claims  # noqa: F401
        import auth_server.enforceai.tokens.mint  # noqa: F401
        import auth_server.enforceai.tokens.verify  # noqa: F401
        import auth_server.enforceai.oidc  # noqa: F401
        import auth_server.enforceai.oidc.claims  # noqa: F401
        import auth_server.enforceai.oidc.jwks  # noqa: F401
        import auth_server.enforceai.oidc.models  # noqa: F401
        import auth_server.enforceai.oidc.verify  # noqa: F401
        import auth_server.enforceai.fgac  # noqa: F401
        import auth_server.enforceai.fgac.catalog  # noqa: F401
        import auth_server.enforceai.fgac.evaluate  # noqa: F401
        import auth_server.enforceai.stores  # noqa: F401
        import auth_server.enforceai.stores.interfaces  # noqa: F401
        import auth_server.enforceai.stores.sqlite  # noqa: F401
        import auth_server.enforceai.stores.sqlite.agent_store  # noqa: F401
        import auth_server.enforceai.stores.sqlite.api_key_store  # noqa: F401
        import auth_server.enforceai.stores.sqlite.audit_store  # noqa: F401
        import auth_server.enforceai.stores.sqlite.revocation_store  # noqa: F401
