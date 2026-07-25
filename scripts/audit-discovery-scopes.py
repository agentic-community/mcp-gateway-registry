#!/usr/bin/env python3
"""Audit groups for the list_agents / list_skills discovery scopes.

Skills and agents are gated on a per-family ``list_`` UI permission (parity with
``list_service``). A group that lacks the grant sees ZERO skills/agents of that
family, including public ones. This READ-ONLY audit lists every group, reports
which ones are missing ``list_agents`` and/or ``list_skills``, and prints the
exact commands to grant them. It never mutates anything itself, so it is safe to
run against any environment.

For each group missing a grant it prints a describe -> edit -> import recipe. A
group whose ``server_access`` contains a reserved wildcard server (``"*"`` /
``"all"``) cannot be re-imported (the import guard refuses it), so for those the
script prints the alternative one-key grant path via the IAM UI instead.

Auth/connection: this script does NOT talk to the registry directly. It shells
out to ``api/registry_management.py`` (``list-groups`` / ``describe-group``) so
all auth, token handling, and client logic stay in that one CLI. Pass the same
``--registry-url`` and ``--token-file`` you would give registry_management.py;
they are forwarded verbatim.

Usage:
    uv run python scripts/audit-discovery-scopes.py \\
        --registry-url http://localhost --token-file .token

    # Only check for a missing list_skills grant (ignore list_agents)
    uv run python scripts/audit-discovery-scopes.py \\
        --registry-url http://localhost --token-file .token --scope list_skills
"""

import argparse
import json
import logging
import subprocess  # nosec B404 - used only to invoke the in-repo registry_management.py CLI
import sys
from pathlib import Path

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# The discovery scopes this audit checks. Servers (list_service) are excluded:
# the server tab has always been gated, so its grants are not a new regression.
_DISCOVERY_SCOPES: tuple[str, ...] = ("list_agents", "list_skills")

# Reserved wildcard server names the import guard refuses in server_access.
# Must match registry.repositories.documentdb.scope_repository.
_RESERVED_SERVER_NAMES: frozenset[str] = frozenset({"all", "*"})

# The management CLI this audit drives. Resolved relative to the repo root
# (this file's parent's parent) so the audit works from any directory.
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_MGMT_CLI: Path = _REPO_ROOT / "api" / "registry_management.py"

# Subprocess timeout in seconds for each management CLI call.
_CLI_TIMEOUT: int = 60


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit groups for missing list_agents / list_skills discovery scopes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python scripts/audit-discovery-scopes.py \\
        --registry-url http://localhost --token-file .token
""",
    )
    parser.add_argument(
        "--registry-url",
        default=None,
        help="Registry base URL (or set REGISTRY_URL)",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help="Path to a file with an admin JWT (or set REGISTRY_TOKEN)",
    )
    parser.add_argument(
        "--scope",
        choices=list(_DISCOVERY_SCOPES),
        default=None,
        help="Only audit this one scope (default: both list_agents and list_skills)",
    )
    return parser.parse_args()


def _run_mgmt_cli(
    registry_url: str,
    token_file: str | None,
    subcommand_args: list[str],
) -> str:
    """Invoke api/registry_management.py and return its stdout.

    Centralizes the subprocess call so auth (via --token-file) and the registry
    URL live only in the management CLI. The command is a fixed list (no shell),
    with a timeout, per the repo subprocess-security rules.

    Args:
        registry_url: Registry base URL, forwarded as --registry-url.
        token_file: Token file path, forwarded as --token-file (may be None to
            let the CLI resolve auth from its own environment).
        subcommand_args: The subcommand and its args (e.g. ["list-groups",
            "--no-keycloak", "--json"]).

    Returns:
        The CLI's stdout as text.

    Raises:
        RuntimeError: If the CLI exits non-zero or times out.
    """
    cmd = [
        sys.executable,
        str(_MGMT_CLI),
        "--registry-url",
        registry_url,
    ]
    if token_file:
        cmd += ["--token-file", token_file]
    cmd += subcommand_args

    try:
        result = subprocess.run(  # nosec B603 - fixed in-repo CLI path, list form, no shell
            cmd,
            capture_output=True,
            text=True,
            timeout=_CLI_TIMEOUT,
            check=True,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"registry_management.py timed out: {' '.join(subcommand_args)}") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"registry_management.py failed ({' '.join(subcommand_args)}): {e.stderr.strip()}"
        ) from e

    return result.stdout


def _extract_json(
    text: str,
) -> dict:
    """Extract the JSON object from CLI stdout that may be prefixed with logs.

    The management CLI logs to stdout/stderr around its JSON payload, so locate
    the first ``{`` and parse from there.

    Args:
        text: Raw CLI stdout.

    Returns:
        The parsed JSON object, or an empty dict if none is found.
    """
    start = text.find("{")
    if start == -1:
        return {}
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return {}


def _fetch_groups(
    registry_url: str,
    token_file: str | None,
) -> dict[str, dict]:
    """List all scope groups via ``registry_management.py list-groups``.

    Args:
        registry_url: Registry base URL.
        token_file: Admin token file path (forwarded to the CLI).

    Returns:
        The ``scopes_groups`` mapping (group name -> summary incl. ``ui_scopes``).
    """
    out = _run_mgmt_cli(
        registry_url,
        token_file,
        ["list-groups", "--no-keycloak", "--json"],
    )
    return _extract_json(out).get("scopes_groups", {})


def _fetch_group_detail(
    registry_url: str,
    token_file: str | None,
    group_name: str,
) -> dict:
    """Get one group's full definition via ``registry_management.py describe-group``.

    Used for the server_access wildcard check.

    Args:
        registry_url: Registry base URL.
        token_file: Admin token file path (forwarded to the CLI).
        group_name: Group to describe.

    Returns:
        The full group definition, or an empty dict on error.
    """
    try:
        out = _run_mgmt_cli(
            registry_url,
            token_file,
            ["describe-group", "--name", group_name, "--json"],
        )
    except RuntimeError as e:
        logger.warning("Could not describe group '%s': %s", group_name, e)
        return {}
    return _extract_json(out)


def _server_access_has_wildcard(
    server_access: list[dict],
) -> bool:
    """Return True if any server_access rule names a reserved wildcard server.

    Such a group cannot be re-imported (the import guard refuses it), so the
    recipe for it must use the IAM UI rather than describe/edit/import.

    Args:
        server_access: The group's server_access list.

    Returns:
        True if a wildcard server rule is present.
    """
    for rule in server_access or []:
        if not isinstance(rule, dict):
            continue
        name = rule.get("server")
        if name and str(name).strip("/").lower() in _RESERVED_SERVER_NAMES:
            return True
    return False


def _missing_scopes(
    ui_scopes: dict,
    scopes_to_check: tuple[str, ...],
) -> list[str]:
    """Return the discovery scopes a group's ui_scopes is missing.

    A scope is "missing" only when the key is absent or its grant list is empty.
    A present non-empty grant (named resources or ``["all"]``) counts as granted.

    Args:
        ui_scopes: The group's ui_permissions/ui_scopes mapping.
        scopes_to_check: Which discovery scopes to check.

    Returns:
        The subset of scopes_to_check that are missing.
    """
    missing = []
    for scope in scopes_to_check:
        granted = (ui_scopes or {}).get(scope) or []
        if not granted:
            missing.append(scope)
    return missing


def _print_recipe(
    group_name: str,
    missing: list[str],
    has_wildcard: bool,
) -> None:
    """Print the fix commands for one group missing discovery scopes.

    Args:
        group_name: The group to fix.
        missing: The scopes it is missing.
        has_wildcard: Whether its server_access has a reserved wildcard (which
            blocks the describe/edit/import round-trip).
    """
    grant_json = ", ".join(f'"{s}": ["all"]' for s in missing)
    print(f"\n# Group '{group_name}' is missing: {', '.join(missing)}")

    if has_wildcard:
        print(
            "#   NOTE: this group's server_access contains a reserved wildcard "
            "('*'/'all'),\n"
            "#   so it cannot be re-imported (the import guard refuses it). Grant "
            "the scope\n"
            "#   via the IAM UI instead: Settings > IAM > Groups > "
            f"{group_name} > UI Permissions,\n"
            f"#   turn on the 'All' toggle for: {', '.join(missing)}."
        )
        return

    print(
        f"uv run python api/registry_management.py \\\n"
        f'  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \\\n'
        f"  describe-group --name {group_name} --json > {group_name}.json\n"
        f'# then add these keys under "ui_permissions" in {group_name}.json:  '
        f"{{ {grant_json} }}\n"
        f"uv run python api/registry_management.py \\\n"
        f'  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \\\n'
        f"  import-group --file {group_name}.json"
    )


def _audit(
    registry_url: str,
    token_file: str | None,
    scopes_to_check: tuple[str, ...],
) -> int:
    """Run the audit and print per-group fix recipes.

    Args:
        registry_url: Registry base URL.
        token_file: Admin token file path (forwarded to the management CLI).
        scopes_to_check: Which discovery scopes to audit.

    Returns:
        Process exit code (0 = ran successfully, regardless of findings).
    """
    groups = _fetch_groups(registry_url, token_file)
    logger.info("Found %d scope group(s)", len(groups))

    groups_missing = {}
    for name, summary in groups.items():
        ui_scopes = summary.get("ui_scopes") or summary.get("ui_permissions") or {}
        missing = _missing_scopes(ui_scopes, scopes_to_check)
        if missing:
            groups_missing[name] = missing

    print("=" * 60)
    print(f"Discovery-scope audit: {len(groups_missing)} of {len(groups)} groups need a grant")
    print("=" * 60)

    if not groups_missing:
        print("\nAll groups already hold the audited discovery scopes. Nothing to do.")
        return 0

    print(
        "\nSet these first, then run the printed commands:\n"
        '  export REGISTRY_URL="<your-registry-url>"\n'
        '  export TOKEN_FILE="<path-to-admin-token-file>"'
    )

    for name in sorted(groups_missing):
        detail = _fetch_group_detail(registry_url, token_file, name)
        has_wildcard = _server_access_has_wildcard(detail.get("server_access", []))
        _print_recipe(name, groups_missing[name], has_wildcard)

    print("\n" + "=" * 60)
    print("Review each command above before running it. This audit changed nothing.")
    print("=" * 60)
    return 0


def main() -> None:
    """Control flow: parse args, resolve config, run the audit."""
    import os

    args = _parse_args()

    registry_url = args.registry_url or os.getenv("REGISTRY_URL")
    if not registry_url:
        logger.error("Provide --registry-url or set REGISTRY_URL")
        sys.exit(1)

    if not _MGMT_CLI.exists():
        logger.error("Management CLI not found at %s", _MGMT_CLI)
        sys.exit(1)

    scopes_to_check = (args.scope,) if args.scope else _DISCOVERY_SCOPES

    try:
        exit_code = _audit(registry_url, args.token_file, scopes_to_check)
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
