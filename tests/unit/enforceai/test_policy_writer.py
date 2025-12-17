"""
Unit tests for scope catalog mutation helpers (policy writer).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auth_server.enforceai.fgac.policy_writer import (
    PolicyConflictError,
    PolicyNotFoundError,
    PolicyPreconditionFailedError,
    write_scope_catalog_scope,
)
from auth_server.enforceai.fgac.catalog import (
    clear_scope_catalog_cache,
    load_scope_catalog,
)


def _write_catalog(
    *,
    path: Path,
    group_mappings: str = "group_mappings: {}",
    scope_block: str = "",
) -> Path:
    content = "\n".join(
        [
            "UI-Scopes: {}",
            group_mappings,
            scope_block.strip(),
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def _read_text(
    path: Path,
) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.unit
class TestPolicyWriter:
    def test_create_scope_writes_and_validates(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_catalog(
            path=tmp_path / "scopes.yml",
            scope_block="\n".join(
                [
                    "existing-scope:",
                    "  - server: mcpgw",
                    "    methods: [tools/list]",
                    "",
                ]
            ),
        )

        result = write_scope_catalog_scope(
            path=path,
            scope_name="new-scope",
            entries=[
                {
                    "server": "mcpgw",
                    "methods": ["tools/list", "tools/call"],
                    "tools": ["tool-a"],
                }
            ],
            mode="create",
        )

        assert result.path == path.resolve()
        assert result.etag

        clear_scope_catalog_cache()
        catalog = load_scope_catalog(path=path)
        assert "existing-scope" in catalog.scopes
        assert "new-scope" in catalog.scopes

    def test_create_scope_conflicts_when_already_exists(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_catalog(
            path=tmp_path / "scopes.yml",
            scope_block="\n".join(
                [
                    "existing-scope:",
                    "  - server: mcpgw",
                    "    methods: [tools/list]",
                    "",
                ]
            ),
        )

        with pytest.raises(PolicyConflictError, match="already exists"):
            write_scope_catalog_scope(
                path=path,
                scope_name="existing-scope",
                entries=[],
                mode="create",
            )

    def test_replace_scope_requires_etag_match(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_catalog(
            path=tmp_path / "scopes.yml",
            scope_block="\n".join(
                [
                    "scope-a:",
                    "  - server: mcpgw",
                    "    methods: [tools/list, tools/call]",
                    "    tools: [tool-a]",
                    "",
                ]
            ),
        )

        with pytest.raises(PolicyPreconditionFailedError, match="ETag mismatch"):
            write_scope_catalog_scope(
                path=path,
                scope_name="scope-a",
                entries=[],
                mode="replace",
                if_match="deadbeef",
            )

    def test_replace_scope_not_found(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_catalog(path=tmp_path / "scopes.yml")

        with pytest.raises(PolicyNotFoundError, match="Scope not found"):
            write_scope_catalog_scope(
                path=path,
                scope_name="missing",
                entries=[],
                mode="replace",
            )

    def test_delete_scope_conflicts_when_referenced_by_group_mappings(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_catalog(
            path=tmp_path / "scopes.yml",
            group_mappings="\n".join(
                [
                    "group_mappings:",
                    "  some-group: [referenced-scope]",
                ]
            ),
            scope_block="\n".join(
                [
                    "referenced-scope:",
                    "  - server: mcpgw",
                    "    methods: [tools/list]",
                    "",
                ]
            ),
        )

        original = _read_text(path)

        with pytest.raises(PolicyConflictError, match="referenced by group_mappings"):
            write_scope_catalog_scope(
                path=path,
                scope_name="referenced-scope",
                entries=[],
                mode="delete",
            )

        assert _read_text(path) == original

    def test_rejects_tools_call_without_tools_policy_and_rolls_back(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_catalog(path=tmp_path / "scopes.yml")
        original = _read_text(path)

        with pytest.raises(ValueError, match="tools/call"):
            write_scope_catalog_scope(
                path=path,
                scope_name="bad-scope",
                entries=[{"server": "mcpgw", "methods": ["tools/call"]}],
                mode="create",
            )

        assert _read_text(path) == original

