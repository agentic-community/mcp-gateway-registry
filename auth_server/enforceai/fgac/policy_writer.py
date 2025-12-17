from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from ..errors import (
    DependencyUnavailableError,
)
from .catalog import (
    clear_scope_catalog_cache,
    load_scope_catalog,
)

logger = logging.getLogger(__name__)

RESERVED_SCOPE_NAMES: frozenset[str] = frozenset({"UI-Scopes", "group_mappings"})


@dataclass(frozen=True)
class PolicyWriteResult:
    path: Path
    etag: str
    last_modified: Optional[str]


class PolicyConflictError(Exception):
    pass


class PolicyNotFoundError(Exception):
    pass


class PolicyPreconditionFailedError(Exception):
    pass


def _compute_etag(
    payload: bytes,
) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(
    path: Path,
) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DependencyUnavailableError(
            f"Failed to read scope catalog at {path}",
            public_message="Scope catalog unavailable",
        ) from exc


def _write_bytes(
    path: Path,
    payload: bytes,
) -> None:
    try:
        path.write_bytes(payload)
    except OSError as exc:
        raise DependencyUnavailableError(
            f"Failed to write scope catalog at {path}",
            public_message="Scope catalog unavailable",
        ) from exc


def _safe_dump_yaml(
    value: dict[str, Any],
) -> str:
    class NoAnchorDumper(yaml.SafeDumper):
        def ignore_aliases(self, data: object) -> bool:  # noqa: ANN001
            return True

    return yaml.dump(
        value,
        default_flow_style=False,
        sort_keys=False,
        Dumper=NoAnchorDumper,
    )


def _parse_yaml_mapping(
    raw: bytes,
    *,
    path: Path,
) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"Scope catalog at {path} must be UTF-8") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in scope catalog: {exc}") from exc

    if payload is None:
        return {}

    if not isinstance(payload, dict):
        raise ValueError("Scope catalog root must be a YAML mapping/object")

    return payload


def _format_last_modified(
    path: Path,
) -> Optional[str]:
    try:
        stat = path.stat()
    except OSError:
        return None

    try:
        from datetime import datetime, timezone

        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(
            microsecond=0
        )
        return modified.isoformat()
    except Exception:
        return None


def _require_matching_etag(
    *,
    current_etag: str,
    if_match: Optional[str],
) -> None:
    if if_match is None:
        return

    provided = if_match.strip()
    if not provided:
        return

    if provided != current_etag:
        raise PolicyPreconditionFailedError("ETag mismatch")


def _normalize_scope_name(
    scope_name: str,
) -> str:
    normalized = scope_name.strip()
    if not normalized:
        raise ValueError("scope_name must be a non-empty string")
    if normalized in RESERVED_SCOPE_NAMES:
        raise ValueError(f"scope_name is reserved: {normalized}")
    return normalized


def _validate_scope_entries(
    entries: list[dict[str, Any]],
) -> None:
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"scope entry {index} must be a mapping/object")

        if "server" in entry:
            methods = entry.get("methods")
            if not isinstance(methods, list):
                raise ValueError(f"scope[{index}].methods must be a list")

            methods_set = {str(item).strip() for item in methods if str(item).strip()}
            methods_lower = {item.lower() for item in methods_set}
            allows_tools_call = ("tools/call" in methods_lower) or (
                "*" in methods_lower or "all" in methods_lower
            )
            if allows_tools_call and "tools" not in entry:
                raise ValueError(
                    "Server permission includes tools/call but has no tools policy"
                )
            continue

        if "agents" in entry:
            agents = entry.get("agents")
            if not isinstance(agents, dict):
                raise ValueError(f"scope[{index}].agents must be a mapping/object")
            actions = agents.get("actions")
            if not isinstance(actions, list):
                raise ValueError(f"scope[{index}].agents.actions must be a list")
            continue

        raise ValueError(f"scope entry {index} must include 'server' or 'agents'")


def _replace_scope(
    *,
    root: dict[str, Any],
    scope_name: str,
    entries: list[dict[str, Any]],
    require_exists: bool,
    forbid_exists: bool,
) -> None:
    exists = scope_name in root
    if require_exists and not exists:
        raise PolicyNotFoundError(f"Scope not found: {scope_name}")
    if forbid_exists and exists:
        raise PolicyConflictError(f"Scope already exists: {scope_name}")

    root[scope_name] = entries


def _delete_scope(
    *,
    root: dict[str, Any],
    scope_name: str,
) -> None:
    if scope_name not in root:
        raise PolicyNotFoundError(f"Scope not found: {scope_name}")

    group_mappings = root.get("group_mappings") or {}
    if isinstance(group_mappings, dict):
        references: list[str] = []
        for group, mapped in group_mappings.items():
            if not isinstance(mapped, list):
                continue
            if scope_name in mapped:
                references.append(str(group))
        if references:
            display = ", ".join(references[:10])
            suffix = "..." if len(references) > 10 else ""
            raise PolicyConflictError(
                f"Scope is referenced by group_mappings: {display}{suffix}"
            )

    del root[scope_name]


def write_scope_catalog_scope(
    *,
    path: Path,
    scope_name: str,
    entries: list[dict[str, Any]],
    mode: str,
    if_match: Optional[str] = None,
) -> PolicyWriteResult:
    """Create/update/delete a single scope entry in the scopes.yml catalog.

    Args:
        path: Catalog file path.
        scope_name: Top-level scope key to mutate.
        entries: New value for the scope key (ignored for delete).
        mode: One of: "create", "replace", "delete".
        if_match: Optional ETag to enforce optimistic concurrency.

    Raises:
        DependencyUnavailableError: File read/write fails.
        ValueError: Invalid inputs or invalid resulting catalog.
        PolicyConflictError: Conflicts (already exists, referenced by mappings).
        PolicyNotFoundError: Missing scope for update/delete.
        PolicyPreconditionFailedError: If-Match ETag mismatch.
    """

    resolved_path = path.expanduser().resolve()
    normalized_name = _normalize_scope_name(scope_name)

    original_bytes = _read_bytes(resolved_path)
    original_etag = _compute_etag(original_bytes)
    _require_matching_etag(
        current_etag=original_etag,
        if_match=if_match,
    )

    root = _parse_yaml_mapping(original_bytes, path=resolved_path)
    if "UI-Scopes" not in root or "group_mappings" not in root:
        raise ValueError("Scope catalog missing required keys (UI-Scopes, group_mappings)")

    if mode not in {"create", "replace", "delete"}:
        raise ValueError("mode must be one of: create, replace, delete")

    backup_path = resolved_path.with_suffix(resolved_path.suffix + ".backup")
    _write_bytes(backup_path, original_bytes)

    try:
        if mode == "delete":
            _delete_scope(
                root=root,
                scope_name=normalized_name,
            )
        else:
            if not isinstance(entries, list):
                raise ValueError("entries must be a list")
            _validate_scope_entries(entries)
            _replace_scope(
                root=root,
                scope_name=normalized_name,
                entries=entries,
                require_exists=(mode == "replace"),
                forbid_exists=(mode == "create"),
            )

        rendered = _safe_dump_yaml(root).encode("utf-8")
        _write_bytes(resolved_path, rendered)

        clear_scope_catalog_cache()
        load_scope_catalog(path=resolved_path)

        new_bytes = _read_bytes(resolved_path)
        try:
            os.remove(backup_path)
        except OSError:
            logger.warning(f"Failed to remove backup file at {backup_path}")

        return PolicyWriteResult(
            path=resolved_path,
            etag=_compute_etag(new_bytes),
            last_modified=_format_last_modified(resolved_path),
        )
    except Exception:
        try:
            _write_bytes(resolved_path, original_bytes)
        finally:
            clear_scope_catalog_cache()
        raise
