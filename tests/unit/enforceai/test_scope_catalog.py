"""
Unit tests for the FGAC scope catalog loader and validator.
"""

from pathlib import Path

import pytest

import auth_server.enforceai.fgac.catalog as catalog_module
from auth_server.enforceai.fgac.catalog import (
    clear_scope_catalog_cache,
    default_scopes_catalog_path,
    load_scope_catalog,
)


@pytest.mark.unit
class TestScopeCatalog:
    def test_loads_real_scopes_catalog(self) -> None:
        clear_scope_catalog_cache()

        catalog = load_scope_catalog()

        assert catalog.path == default_scopes_catalog_path().resolve()
        assert "registry-users-lob1" in catalog.group_mappings
        assert "mcp-servers-unrestricted/read" in catalog.scopes

        unrestricted_read = catalog.scopes["mcp-servers-unrestricted/read"]
        assert any(rule.server == "*" for rule in unrestricted_read.server_permissions)

    def test_rejects_malformed_yaml(
        self,
        tmp_path: Path,
    ) -> None:
        clear_scope_catalog_cache()

        path = tmp_path / "scopes.yml"
        path.write_text("UI-Scopes: [", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_scope_catalog(path=path)

    def test_rejects_invalid_schema_missing_required_keys(
        self,
        tmp_path: Path,
    ) -> None:
        clear_scope_catalog_cache()

        path = tmp_path / "scopes.yml"
        path.write_text("UI-Scopes: {}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="missing required key: group_mappings"):
            load_scope_catalog(path=path)

    def test_cache_reuses_loaded_catalog(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clear_scope_catalog_cache()

        read_calls: int = 0
        original_read_text = catalog_module._read_text

        def _counting_read_text(path: Path) -> str:
            nonlocal read_calls
            read_calls += 1
            return original_read_text(path)

        monkeypatch.setattr(
            catalog_module,
            "_read_text",
            _counting_read_text,
        )

        path = default_scopes_catalog_path()
        first = load_scope_catalog(path=path)
        second = load_scope_catalog(path=path)

        assert first is second
        assert read_calls == 1

