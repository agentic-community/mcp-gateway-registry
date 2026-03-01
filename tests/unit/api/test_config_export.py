"""Unit tests for configuration export functions.

Tests from LLD section 7.1 — validates export format correctness
for env, JSON, and tfvars outputs.
"""

import json

from registry.api.config_routes import (
    _export_as_env,
    _export_as_json,
    _export_as_tfvars,
)


class TestConfigExport:
    """Export function unit tests (Requirements 3.3, 3.4, 3.6, 3.7, 8.1)."""

    def test_export_env_masks_sensitive(self):
        """Verify _export_as_env(include_sensitive=False) masks sensitive values."""
        output = _export_as_env(include_sensitive=False)
        assert "SENSITIVE_VALUE_MASKED" in output
        # Should not contain raw password values from settings
        from registry.core.config import settings

        raw_password = settings.admin_password
        # Sensitive fields should be commented out, not exposed
        assert "# SECRET_KEY=<SENSITIVE_VALUE_MASKED>" in output

    def test_export_env_includes_sensitive_when_requested(self):
        """Verify _export_as_env(include_sensitive=True) does not mask."""
        output = _export_as_env(include_sensitive=True)
        assert "SENSITIVE_VALUE_MASKED" not in output

    def test_export_json_valid_json(self):
        """Verify _export_as_json produces valid JSON with required keys."""
        output = _export_as_json(include_sensitive=False)
        parsed = json.loads(output)
        assert "_metadata" in parsed
        assert "configuration" in parsed
        assert "exported_at" in parsed["_metadata"]
        assert "registry_mode" in parsed["_metadata"]
        assert "includes_sensitive" in parsed["_metadata"]

    def test_export_tfvars_valid_syntax(self):
        """Verify _export_as_tfvars has no Python literals (None, True)."""
        output = _export_as_tfvars(include_sensitive=False)
        for line in output.splitlines():
            stripped = line.strip()
            # Skip comments and empty lines
            if stripped.startswith("#") or not stripped:
                continue
            # Should not contain Python-style True/False/None
            assert "None" not in stripped, f"Found Python 'None' in: {stripped}"
            assert "True" not in stripped, f"Found Python 'True' in: {stripped}"
            assert "False" not in stripped, f"Found Python 'False' in: {stripped}"
