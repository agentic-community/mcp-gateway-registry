"""CLI entry point for AgentCore auto-registration.

Provides ``sync`` and ``list`` subcommands via argparse.

Usage::

    python -m cli.agentcore.sync sync [options]
    python -m cli.agentcore.sync list [options]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .models import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_REGION,
    DEFAULT_REGISTRY_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_TOKEN_FILE,
    _load_token,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argparse setup
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with sync and list subcommands."""
    parser = argparse.ArgumentParser(
        prog="agentcore-sync",
        description=(
            "Discover and register AWS Bedrock AgentCore Gateways and "
            "Agent Runtimes with the MCP Gateway Registry."
        ),
        epilog=(
            "Environment variables:\n"
            "  AWS_REGION                    AWS region (default: us-east-1)\n"
            "  REGISTRY_URL                  Registry base URL\n"
            "  REGISTRY_TOKEN_FILE           Path to registry auth token file\n"
            "  AGENTCORE_ACCOUNTS            Comma-separated account IDs (cross-account)\n"
            "  AGENTCORE_ASSUME_ROLE_NAME    Role name to assume (default: AgentCoreSyncRole)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    # -- shared arguments --------------------------------------------------
    def add_common_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--region",
            default=os.environ.get("AWS_REGION", DEFAULT_REGION),
            help="AWS region (default: AWS_REGION env or us-east-1)",
        )
        sub.add_argument(
            "--registry-url",
            default=os.environ.get("REGISTRY_URL", DEFAULT_REGISTRY_URL),
            help="Registry base URL (default: REGISTRY_URL env or http://localhost)",
        )
        sub.add_argument(
            "--token-file",
            default=os.environ.get("REGISTRY_TOKEN_FILE", DEFAULT_TOKEN_FILE),
            help="Path to registry auth token file",
        )
        sub.add_argument(
            "--timeout",
            type=int,
            default=DEFAULT_TIMEOUT,
            help="AWS API call timeout in seconds (default: 30)",
        )
        sub.add_argument(
            "--gateways-only",
            action="store_true",
            help="Only process gateways",
        )
        sub.add_argument(
            "--runtimes-only",
            action="store_true",
            help="Only process runtimes",
        )
        sub.add_argument(
            "--output",
            choices=["text", "json"],
            default="text",
            help="Output format (default: text)",
        )
        sub.add_argument(
            "--debug",
            action="store_true",
            help="Enable DEBUG logging",
        )

    # -- cross-account arguments (shared by sync and list) ----------------
    def add_cross_account_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--accounts",
            default=os.environ.get("AGENTCORE_ACCOUNTS", ""),
            help=(
                "Comma-separated AWS account IDs to scan (cross-account). "
                "Requires a role in each account that the caller can assume. "
                "(default: current account only)"
            ),
        )
        sub.add_argument(
            "--assume-role-name",
            default=os.environ.get(
                "AGENTCORE_ASSUME_ROLE_NAME", "AgentCoreSyncRole"
            ),
            help=(
                "IAM role name to assume in each target account "
                "(default: AGENTCORE_ASSUME_ROLE_NAME env or AgentCoreSyncRole)"
            ),
        )

    # -- sync subcommand ---------------------------------------------------
    sync_parser = subparsers.add_parser(
        "sync",
        help="Discover and register AgentCore resources",
    )
    add_common_args(sync_parser)
    add_cross_account_args(sync_parser)
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without registering or persisting credentials",
    )
    sync_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing registrations",
    )
    sync_parser.add_argument(
        "--visibility",
        choices=["public", "internal", "group-restricted"],
        default="internal",
        help="Registration visibility (default: internal)",
    )
    sync_parser.add_argument(
        "--include-mcp-targets",
        action="store_true",
        help="Register mcpServer gateway targets as separate MCP Servers",
    )
    sync_parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST_PATH,
        help="Output path for token refresh manifest (default: token_refresh_manifest.json)",
    )

    # -- list subcommand ---------------------------------------------------
    list_parser = subparsers.add_parser(
        "list",
        help="Discover and display AgentCore resources without registering",
    )
    add_common_args(list_parser)
    add_cross_account_args(list_parser)

    return parser


# ---------------------------------------------------------------------------
# Cross-account helpers
# ---------------------------------------------------------------------------


def _parse_account_ids(accounts_str: str) -> list[str]:
    """Parse comma-separated account IDs, stripping whitespace."""
    if not accounts_str or not accounts_str.strip():
        return []
    return [a.strip() for a in accounts_str.split(",") if a.strip()]


def _assume_role_session(
    account_id: str,
    role_name: str,
    region: str,
) -> "boto3.Session":
    """Assume an IAM role in a target account and return a boto3 Session.

    Args:
        account_id: Target AWS account ID.
        role_name: IAM role name to assume in the target account.
        region: AWS region for the STS call.

    Returns:
        boto3.Session with temporary credentials from the assumed role.

    Raises:
        botocore.exceptions.ClientError: If AssumeRole fails.
    """
    import boto3

    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    logger.info(f"Assuming role {role_arn} for cross-account access...")

    sts = boto3.client("sts", region_name=region)
    response = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=f"agentcore-sync-{account_id}",
        DurationSeconds=3600,
    )
    creds = response["Credentials"]

    session = boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=region,
    )
    logger.info(f"Assumed role in account {account_id} successfully")
    return session


# ---------------------------------------------------------------------------
# cmd_sync
# ---------------------------------------------------------------------------


def cmd_sync(args: argparse.Namespace) -> int:
    """Execute the sync subcommand: discover, register, write manifest."""
    # Load registry token
    try:
        token = _load_token(args.token_file)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    # Late imports to keep argparse fast
    from .discovery import AgentCoreScanner
    from .registration import RegistrationBuilder, SyncOrchestrator

    # Add project root so api.registry_client is importable
    sys.path.insert(
        0,
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ),
    )
    from api.registry_client import RegistryClient

    registry = RegistryClient(registry_url=args.registry_url, token=token)

    # Determine accounts to scan
    account_ids = _parse_account_ids(getattr(args, "accounts", ""))
    role_name = getattr(args, "assume_role_name", "AgentCoreSyncRole")

    # Build list of (label, session_or_none) pairs
    # Empty list = current account only (session=None)
    account_sessions: list[tuple[str, object]] = []
    if account_ids:
        for acct in account_ids:
            try:
                session = _assume_role_session(acct, role_name, args.region)
                account_sessions.append((acct, session))
            except Exception as e:
                logger.error(f"Failed to assume role in account {acct}: {e}")
                if args.output == "json":
                    print(json.dumps({"error": f"AssumeRole failed for {acct}: {e}"}))
                return 1
    else:
        account_sessions.append(("current", None))

    # Run sync for each account
    for label, session in account_sessions:
        if len(account_sessions) > 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"Syncing account: {label}")
            logger.info(f"{'='*60}")

        scanner = AgentCoreScanner(
            region=args.region, timeout=args.timeout, session=session
        )
        builder = RegistrationBuilder(
            region=args.region, visibility=args.visibility, session=session
        )

        orchestrator = SyncOrchestrator(
            scanner=scanner,
            builder=builder,
            registry_client=registry,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            include_mcp_targets=args.include_mcp_targets,
            output_format=args.output,
            manifest_path=args.manifest,
        )

        # Scope filtering
        if not args.runtimes_only:
            orchestrator.sync_gateways()
        if not args.gateways_only:
            orchestrator.sync_runtimes()

        # Write token refresh manifest
        orchestrator.write_manifest()

        # Summary
        orchestrator.print_summary()

    return 0


# ---------------------------------------------------------------------------
# cmd_list
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    """Execute the list subcommand: discover and display resources."""
    from .discovery import AgentCoreScanner

    # Determine accounts to scan
    account_ids = _parse_account_ids(getattr(args, "accounts", ""))
    role_name = getattr(args, "assume_role_name", "AgentCoreSyncRole")

    account_sessions: list[tuple[str, object]] = []
    if account_ids:
        for acct in account_ids:
            try:
                session = _assume_role_session(acct, role_name, args.region)
                account_sessions.append((acct, session))
            except Exception as e:
                logger.error(f"Failed to assume role in account {acct}: {e}")
                return 1
    else:
        account_sessions.append(("current", None))

    all_gateways: list = []
    all_runtimes: list = []
    all_errors: list[str] = []

    for label, session in account_sessions:
        scanner = AgentCoreScanner(
            region=args.region, timeout=args.timeout, session=session
        )

        if not args.runtimes_only:
            try:
                gateways = scanner.scan_gateways()
                # Tag with account for multi-account output
                if len(account_sessions) > 1:
                    for gw in gateways:
                        gw["_account"] = label
                all_gateways.extend(gateways)
            except Exception as e:
                all_errors.append(f"[{label}] Gateway scan error: {e}")
                logger.error(f"Failed to scan gateways in {label}: {e}")

        if not args.gateways_only:
            try:
                runtimes = scanner.scan_runtimes()
                if len(account_sessions) > 1:
                    for rt in runtimes:
                        rt["_account"] = label
                all_runtimes.extend(runtimes)
            except Exception as e:
                all_errors.append(f"[{label}] Runtime scan error: {e}")
                logger.error(f"Failed to scan runtimes in {label}: {e}")

    if args.output == "json":
        print(
            json.dumps(
                {
                    "region": args.region,
                    "accounts": account_ids or ["current"],
                    "gateways": all_gateways,
                    "runtimes": all_runtimes,
                    "errors": all_errors,
                },
                indent=2,
                default=str,
            )
        )
    else:
        _print_list_text(all_gateways, all_runtimes, args.region, all_errors)

    return 0


def _print_list_text(
    gateways: list,
    runtimes: list,
    region: str,
    errors: list[str],
) -> None:
    """Print discovered resources in text format."""
    print(f"\nAgentCore Resources in {region}")
    print("=" * 70)

    if gateways:
        print(f"\nGateways ({len(gateways)}):")
        print("-" * 70)
        for gw in gateways:
            name = gw.get("name", gw.get("gatewayId", "unknown"))
            auth = gw.get("authorizerType", "unknown")
            status = gw.get("status", "unknown")
            targets = len(gw.get("targets", []))
            print(f"  {name:<30} auth={auth:<12} targets={targets}  [{status}]")
    else:
        print("\nNo gateways found.")

    if runtimes:
        print(f"\nRuntimes ({len(runtimes)}):")
        print("-" * 70)
        for rt in runtimes:
            name = rt.get("agentRuntimeName", rt.get("agentRuntimeId", "unknown"))
            protocol = rt.get("protocolConfiguration", {}).get(
                "serverProtocol", "unknown"
            )
            status = rt.get("status", "unknown")
            print(f"  {name:<30} protocol={protocol:<8} [{status}]")
    else:
        print("\nNo runtimes found.")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse args, configure logging, dispatch subcommand."""
    # Load .env before anything reads os.environ
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
    )

    logger.debug(f"CLI args: {args}")

    if args.command == "sync":
        return cmd_sync(args)
    elif args.command == "list":
        return cmd_list(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
