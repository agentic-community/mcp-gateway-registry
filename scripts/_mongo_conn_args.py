"""Shared Mongo/DocumentDB connection CLI args for the backfill scripts.

The backfill scripts (``backfill-skill-list-scope.py``,
``backfill-custom-entity-scopes.py``) normally read their storage backend and
connection from the registry environment / ``.env``. For deployments where those
service names are not resolvable from where the script runs (a host shell against
a Dockerized Mongo, a one-off ECS task pointed at Amazon DocumentDB, an EKS admin
pod), this module adds a shared set of ``--host`` / ``--username`` / ``--tls`` /
etc. args and translates them into the env the registry config singleton reads.

SECURITY: the password and SECRET_KEY are read from the environment ONLY and are
never accepted as CLI args (argv is world-readable via the process list). Supply
them via ``DOCUMENTDB_PASSWORD`` / ``SECRET_KEY``.

Usage from a script::

    import argparse
    from _mongo_conn_args import add_connection_args, apply_connection_overrides

    parser = argparse.ArgumentParser(...)
    add_connection_args(parser)
    args = parser.parse_args()
    apply_connection_overrides(args)   # BEFORE importing registry.*
"""

import argparse
import logging
import os
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Environment variable holding the Mongo password. Never a CLI arg (argv is
# world-readable via the process list).
PASSWORD_ENV_VAR: str = "DOCUMENTDB_PASSWORD"

# Backends that support SCRAM-SHA-256. Amazon DocumentDB v5.0 supports only
# SCRAM-SHA-1, matching registry.utils.mongodb_connection.
_SHA256_BACKENDS: frozenset[str] = frozenset({"mongodb-ce", "mongodb", "mongodb-atlas"})


def add_connection_args(
    parser: argparse.ArgumentParser,
) -> None:
    """Add the shared connection-override arguments to ``parser``.

    Args:
        parser: The argument parser to extend.
    """
    conn = parser.add_argument_group(
        "connection overrides",
        "Override the Mongo/DocumentDB connection. Omit to use the registry "
        "environment / .env. The password is read from the "
        f"{PASSWORD_ENV_VAR} environment variable, never a CLI arg.",
    )
    conn.add_argument(
        "--host",
        default=None,
        help="Mongo/DocumentDB host (enables CLI connection overrides when set)",
    )
    conn.add_argument(
        "--port",
        type=int,
        default=27017,
        help="Mongo/DocumentDB port (default: 27017)",
    )
    conn.add_argument(
        "--database",
        default="mcp_registry",
        help="Database name (default: mcp_registry)",
    )
    conn.add_argument(
        "--username",
        default=None,
        help=f"Username for SCRAM auth (password via ${PASSWORD_ENV_VAR})",
    )
    conn.add_argument(
        "--auth-source",
        default="admin",
        help="authSource database for SCRAM auth (default: admin)",
    )
    conn.add_argument(
        "--tls",
        action="store_true",
        help="Enable TLS (required for Amazon DocumentDB)",
    )
    conn.add_argument(
        "--direct-connection",
        action="store_true",
        help="Set directConnection=true (reach a single node without replica-set "
        "discovery; use when the advertised replica-set hostnames are not "
        "resolvable from where the script runs)",
    )
    conn.add_argument(
        "--storage-backend",
        default="mongodb-ce",
        choices=["mongodb-ce", "documentdb", "mongodb", "mongodb-atlas"],
        help="Storage backend, selects the SCRAM mechanism (documentdb -> "
        "SCRAM-SHA-1, others -> SCRAM-SHA-256). Default: mongodb-ce",
    )
    conn.add_argument(
        "--auth-server-url",
        default=None,
        help="Auth-server base URL for the post-grant scope reload "
        "(e.g. http://localhost:8888). Omit to use the registry environment.",
    )


def _build_connection_string(
    args: argparse.Namespace,
) -> str:
    """Build a Mongo connection string from the CLI connection overrides.

    The password comes from the environment (never argv). The SCRAM mechanism is
    derived from ``--storage-backend`` to match the registry's own logic.

    Args:
        args: Parsed CLI args with ``host`` set.

    Returns:
        A ``mongodb://`` connection string.
    """
    query: list[str] = []
    if args.username:
        password = os.environ.get(PASSWORD_ENV_VAR, "")
        credentials = f"{quote_plus(args.username)}:{quote_plus(password)}@"
        mechanism = "SCRAM-SHA-256" if args.storage_backend in _SHA256_BACKENDS else "SCRAM-SHA-1"
        query.append(f"authMechanism={mechanism}")
        query.append(f"authSource={args.auth_source}")
    else:
        credentials = ""

    if args.tls:
        query.append("tls=true")
    if args.direct_connection:
        query.append("directConnection=true")

    query_string = ("?" + "&".join(query)) if query else ""
    return f"mongodb://{credentials}{args.host}:{args.port}/{args.database}{query_string}"


def apply_connection_overrides(
    args: argparse.Namespace,
) -> None:
    """Translate CLI connection args into the env the registry config reads.

    Must run BEFORE any ``registry`` import, because ``registry.core.config``
    instantiates its settings singleton at import time. Setting these env vars
    first makes the singleton pick up the overrides. A no-op for the connection
    string when ``--host`` is not supplied (the registry environment / .env is
    used as-is), though ``--storage-backend`` / ``--auth-server-url`` are still
    applied when provided.

    Args:
        args: Parsed CLI args produced after :func:`add_connection_args`.
    """
    if args.storage_backend:
        os.environ["STORAGE_BACKEND"] = args.storage_backend
    if args.auth_server_url:
        os.environ["AUTH_SERVER_URL"] = args.auth_server_url

    if not args.host:
        return

    os.environ["MONGODB_CONNECTION_STRING"] = _build_connection_string(args)
    # Log the target WITHOUT credentials (host:port/db only).
    logger.info(
        "Using CLI connection override: %s:%s/%s (backend=%s, tls=%s, direct=%s)",
        args.host,
        args.port,
        args.database,
        args.storage_backend,
        args.tls,
        args.direct_connection,
    )
