"""
Unit tests ensuring EnforceAI package skeleton is importable under pytest.
"""

import pytest


@pytest.mark.unit
class TestEnforceAIImports:
    """Test suite for EnforceAI package imports."""

    def test_imports(self):
        import auth_server.enforceai  # noqa: F401
        import auth_server.enforceai.config  # noqa: F401
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
        import auth_server.enforceai.stores  # noqa: F401
        import auth_server.enforceai.stores.interfaces  # noqa: F401
        import auth_server.enforceai.stores.sqlite  # noqa: F401
        import auth_server.enforceai.stores.sqlite.agent_store  # noqa: F401
        import auth_server.enforceai.stores.sqlite.api_key_store  # noqa: F401
        import auth_server.enforceai.stores.sqlite.audit_store  # noqa: F401
        import auth_server.enforceai.stores.sqlite.revocation_store  # noqa: F401
