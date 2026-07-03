"""Egress server-path + config helpers (shared by the egress consent routes).

These are the small pure helpers the egress consent flow needs: canonicalizing a
server path and deciding whether a server is wired for per-user OAuth egress.

"""

import logging

from registry.egress_auth.providers import resolve_provider

logger = logging.getLogger(__name__)


# Standard well-known prefix for the per-server PRM.
PRM_WELLKNOWN_PREFIX: str = "/.well-known/oauth-protected-resource"


def _normalize_server_path(server_path: str) -> str:
    """Return the canonical leading-slash server path (e.g. ``/github``)."""
    return server_path if server_path.startswith("/") else "/" + server_path


def is_server_egress_configured(server: dict | None) -> bool:
    """True iff the server has per-user OAuth egress wired (mode + provider config)."""
    if not server:
        return False
    if server.get("egress_auth_mode") != "oauth_user":
        return False
    eo = server.get("egress_oauth")
    if not eo:
        return False
    try:
        resolve_provider(eo)
    except ValueError:
        return False
    return True
