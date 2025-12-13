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
        import auth_server.enforceai.errors  # noqa: F401
        import auth_server.enforceai.identity  # noqa: F401
        import auth_server.enforceai.logging  # noqa: F401
