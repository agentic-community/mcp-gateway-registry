#!/usr/bin/env python3
"""Backfill per-type custom-entity scopes for pre-existing types.

Types created before the per-type authorization feature have no minted UI-Scopes,
so their records became admin-only after the upgrade. This one-time migration
grants each EXISTING custom type's full scope set
(``list/create/modify/delete_<type>_entity: ["all"]``) to the ``mcp-registry-admin``
group and triggers an auth-server scope reload.

BEHAVIOR CHANGE (announced): this backfill grants the scope set to
``mcp-registry-admin`` ONLY. Existing types therefore become admin-only until an
admin explicitly grants scopes to other groups via the IAM UI. Records that were
previously visible to non-admins (public/group-restricted) are hidden from them
until such a grant is made -- this is the intended, stricter parity with
``list_service``.

The migration is IDEMPOTENT: minting is a per-key ``$set`` merge, so re-running
it simply re-writes the same scope values. Safe to run repeatedly.

Usage:
    # Dry run (default) - list the types that would be backfilled
    SECRET_KEY=... DOCUMENTDB_HOST=localhost \\
        uv run python scripts/backfill-custom-entity-scopes.py

    # Actually apply changes
    SECRET_KEY=... DOCUMENTDB_HOST=localhost \\
        uv run python scripts/backfill-custom-entity-scopes.py --apply

    # MongoDB CE (single-node) backend
    MCP_STORAGE_BACKEND=mongodb-ce DOCUMENTDB_HOST=localhost \\
        uv run python scripts/backfill-custom-entity-scopes.py --apply

Requires the registry package importable (run from the repo root via uv). The
storage backend and connection are read from the same environment the registry
uses (MCP_STORAGE_BACKEND, DOCUMENTDB_HOST, etc.).
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure both the repo root AND this script's own directory are importable.
# Running the script directly (``python scripts/backfill-custom-entity-scopes.py``)
# puts the ``scripts/`` directory on sys.path[0] but NOT the repo root, so
# ``import registry`` would fail. Conversely, when the script is loaded by path
# (e.g. by the pytest suite via importlib), NEITHER is on sys.path, so the
# sibling ``import _mongo_conn_args`` would fail with ModuleNotFoundError.
# Prepending both makes the script self-contained regardless of how it is run.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Shared Mongo/DocumentDB connection CLI args (sibling module in scripts/).
from _mongo_conn_args import (  # noqa: E402
    add_connection_args,
    apply_connection_overrides,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill per-type custom-entity scopes to mcp-registry-admin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (default)
    uv run python scripts/backfill-custom-entity-scopes.py

    # Apply changes
    uv run python scripts/backfill-custom-entity-scopes.py --apply
""",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (default is a dry run)",
    )
    add_connection_args(parser)
    return parser.parse_args()


async def _run_backfill(
    dry_run: bool,
) -> dict[str, int]:
    """Backfill scopes for every existing custom type.

    Args:
        dry_run: If True, only report which types would be backfilled.

    Returns:
        Summary dict with counts (types_found, types_minted).
    """
    from registry.repositories.factory import get_custom_entity_service
    from registry.services.scope_service import (
        ScopeMintError,
        mint_custom_type_scopes,
        trigger_auth_server_reload,
    )

    service = get_custom_entity_service()
    descriptors = await service.list_types()
    type_names = sorted(d.name for d in descriptors)

    logger.info("Found %d existing custom type(s): %s", len(type_names), type_names)

    if dry_run:
        logger.info("DRY RUN - no changes will be made")
        for name in type_names:
            logger.info("  Would mint scopes for type: %s", name)
        return {"types_found": len(type_names), "types_minted": 0}

    minted = 0
    for name in type_names:
        try:
            await mint_custom_type_scopes(name)
            minted += 1
            logger.info("  Minted scopes for type: %s", name)
        except ScopeMintError as e:
            logger.error("  Failed to mint scopes for type %s: %s", name, e)

    if minted:
        reloaded = await trigger_auth_server_reload()
        logger.info("Triggered auth-server scope reload: success=%s", reloaded)

    return {"types_found": len(type_names), "types_minted": minted}


async def main() -> None:
    """Main entry point for the backfill migration."""
    args = _parse_args()
    dry_run = not args.apply

    logger.info("=" * 60)
    logger.info("Custom-Entity Scope Backfill")
    logger.info("=" * 60)
    logger.info("Mode: %s", "DRY RUN" if dry_run else "APPLY CHANGES")
    logger.info("=" * 60)

    # Apply CLI connection overrides into the environment BEFORE the registry
    # config singleton is imported inside _run_backfill.
    apply_connection_overrides(args)

    result = await _run_backfill(dry_run)

    logger.info("=" * 60)
    logger.info("Backfill Summary:")
    logger.info("  Types found:  %d", result["types_found"])
    logger.info("  Types minted: %d", result["types_minted"])
    if dry_run:
        logger.info("  Note: dry run. Use --apply to make changes.")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
