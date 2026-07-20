#!/usr/bin/env python3
"""Backfill the list_skills discovery scope for the admin group.

Skills gained a per-asset ``list_skills`` DISCOVERY gate (parity with
``list_service`` / ``list_agents`` / ``list_<type>_entity``). Before this
feature there was no such gate: every user (including anonymous) could discover
public skills, and no scope document held a ``list_skills`` key. With the gate
enforced, a caller that does not hold ``list_skills`` (or ``["all"]``) sees ZERO
skills -- including public ones.

This one-time migration grants ``list_skills: ["all"]`` to the
``mcp-registry-admin`` group and triggers an auth-server scope reload.

BEHAVIOR CHANGE (announced): this backfill grants the scope to
``mcp-registry-admin`` ONLY. After upgrade, skills become admin-only until an
admin explicitly grants ``list_skills`` to other groups via the IAM UI ("User
Groups" -> UI Permissions -> List Skills). Skills that were previously visible
to non-admins (public/group-restricted) are hidden from them until such a grant
is made -- this is the intended, stricter parity with ``list_service``.

The migration is IDEMPOTENT: minting is a per-key ``$set`` merge, so re-running
it simply re-writes the same value. Safe to run repeatedly.

Usage:
    # Dry run (default) - report what would be granted, connection from env/.env
    uv run python scripts/backfill-skill-list-scope.py

    # Actually apply changes
    uv run python scripts/backfill-skill-list-scope.py --apply

The simplest place to run this is inside a running container (Docker Compose:
``docker compose exec registry ...``; EKS: ``kubectl exec deploy/registry -- ...``),
where the registry environment is already present. The script also adds the repo
root to sys.path itself, so it can be run from any working directory.

CONNECTION: by default the storage backend and connection are read from the same
environment the registry uses (STORAGE_BACKEND, DOCUMENTDB_HOST, etc., including
values in a local ``.env``). For deployments where those service names are not
resolvable from wherever you run the script (a host shell against a Dockerized
Mongo, a one-off task pointed at Amazon DocumentDB on ECS, an EKS admin pod), the
connection can be supplied via CLI args instead (see ``--host`` and friends).

    # Amazon DocumentDB on ECS (TLS, SCRAM-SHA-1 via --storage-backend documentdb)
    DOCUMENTDB_PASSWORD=... SECRET_KEY=... \\
        uv run python scripts/backfill-skill-list-scope.py --apply \\
        --storage-backend documentdb --host docdb.cluster-xxxx.us-east-1.docdb.amazonaws.com \\
        --username admin --tls --auth-server-url http://localhost:8888

    # MongoDB CE on EKS / single node reached directly (skip replica-set discovery)
    DOCUMENTDB_PASSWORD=... SECRET_KEY=... \\
        uv run python scripts/backfill-skill-list-scope.py --apply \\
        --host localhost --username admin --direct-connection

SECURITY: the password and SECRET_KEY are read from the environment only and are
never accepted as CLI args (argv is visible in the process list). Supply them via
``DOCUMENTDB_PASSWORD`` / ``SECRET_KEY`` (env vars, or a task/secret injection).
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure both the repo root AND this script's own directory are importable.
# Running the script directly (``python scripts/backfill-skill-list-scope.py``)
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

# The built-in registry-admin group that the discovery scope is granted to.
# Matches scope_service.ADMIN_GROUP_NAME; non-admins are granted list_skills
# explicitly via the IAM UI after upgrade.
ADMIN_GROUP_NAME: str = "mcp-registry-admin"

# The discovery scope granted for all skills. Name matches the canonical
# registry.auth.asset_permissions map entry ("skill", "list") -> list_skills.
LIST_SKILLS_SCOPE: str = "list_skills"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill the list_skills discovery scope to mcp-registry-admin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (default), connection from environment / .env
    uv run python scripts/backfill-skill-list-scope.py

    # Apply, connection from environment / .env
    uv run python scripts/backfill-skill-list-scope.py --apply

    # Apply with an explicit connection (password + SECRET_KEY stay in env)
    DOCUMENTDB_PASSWORD=... SECRET_KEY=... \\
        uv run python scripts/backfill-skill-list-scope.py --apply \\
        --host localhost --username admin --direct-connection
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
) -> dict[str, bool]:
    """Grant list_skills:["all"] to the admin group.

    Args:
        dry_run: If True, only report what would be granted.

    Returns:
        Summary dict with a ``granted`` flag.
    """
    from registry.repositories.factory import get_scope_repository
    from registry.services.scope_service import trigger_auth_server_reload

    scope_repo = get_scope_repository()

    if dry_run:
        logger.info("DRY RUN - no changes will be made")
        logger.info(
            "  Would grant %s: ['all'] to group '%s'",
            LIST_SKILLS_SCOPE,
            ADMIN_GROUP_NAME,
        )
        return {"granted": False}

    # Per-key $set merge (idempotent); does not round-trip the whole doc, so the
    # privileged-write guard is not involved. list_skills is read-only-prefixed
    # and never admin-conferring, so this is not a privileged write regardless.
    granted = await scope_repo.merge_ui_permissions(ADMIN_GROUP_NAME, {LIST_SKILLS_SCOPE: ["all"]})
    if granted:
        logger.info("  Granted %s: ['all'] to group '%s'", LIST_SKILLS_SCOPE, ADMIN_GROUP_NAME)
        reloaded = await trigger_auth_server_reload()
        logger.info("Triggered auth-server scope reload: success=%s", reloaded)
    else:
        logger.error(
            "  Failed to grant %s to group '%s' (group not found or not updated)",
            LIST_SKILLS_SCOPE,
            ADMIN_GROUP_NAME,
        )

    return {"granted": granted}


async def main() -> None:
    """Main entry point for the backfill migration."""
    args = _parse_args()
    dry_run = not args.apply

    logger.info("=" * 60)
    logger.info("Skill list-scope Backfill")
    logger.info("=" * 60)
    logger.info("Mode: %s", "DRY RUN" if dry_run else "APPLY CHANGES")
    logger.info("=" * 60)

    # Apply CLI connection overrides into the environment BEFORE the registry
    # config singleton is imported inside _run_backfill.
    apply_connection_overrides(args)

    result = await _run_backfill(dry_run)

    logger.info("=" * 60)
    logger.info("Backfill Summary:")
    logger.info("  Granted: %s", result["granted"])
    if dry_run:
        logger.info("  Note: dry run. Use --apply to make changes.")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
