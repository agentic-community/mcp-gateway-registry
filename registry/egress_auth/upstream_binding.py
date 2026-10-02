"""Registered outbound destinations and write-time approvals for egress credentials.

This module is the single definition of:

- which linked version documents belong to a server (``linked_version``);
- the exact URL a request for one version and route mode is sent to
  (``selected_upstream``);
- the destination set a credential approves when it is written
  (``registered_destinations``).

nginx renders its routes from ``selected_upstream``/``linked_version``, the vend
checks the signed upstream against them, and consent snapshots
``registered_destinations``. Keeping all three here is what stops the route, the
check and the approval from drifting apart.

An approval is the exact outbound URL. It is deliberately NOT keyed on a version
document id: promoting a version swaps the active and inactive document ids
without changing any destination, so an id-keyed approval would be revoked by a
promotion that sends nothing anywhere new. A credential is usable for a URL only
if that URL was registered when the user approved it AND is still registered for
the version the request selected.
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def selected_upstream(server: dict, virtual_backend: bool) -> str:
    """Resolve the exact URL a request for this version and route is sent to."""
    from registry.core.endpoint_utils import get_endpoint_url_from_server_info

    proxy_url = server.get("proxy_pass_url")
    if not isinstance(proxy_url, str) or not proxy_url:
        raise ValueError("selected version has no proxy_pass_url")
    if not virtual_backend:
        return proxy_url
    endpoint = get_endpoint_url_from_server_info(server)
    parsed_proxy = urlparse(proxy_url)
    return f"{parsed_proxy.scheme}://{parsed_proxy.netloc}{urlparse(endpoint).path.rstrip('/')}"


async def linked_version(
    server: dict,
    server_path: str,
    version_id: str,
) -> dict | None:
    """Return the inactive version document ``version_id`` if it belongs to ``server``.

    None unless the id is listed on the active document, is namespaced under
    ``server_path``, and the stored document points back at this server.
    """
    from registry.repositories.factory import get_server_repository

    if (
        not isinstance(version_id, str)
        or not version_id.startswith(server_path + ":")
        or version_id not in (server.get("other_version_ids") or [])
    ):
        return None
    version = await get_server_repository().get(version_id)
    if (
        not version
        or version.get("path") != version_id
        or version.get("active_version_id") not in (None, server_path)
    ):
        return None
    return version


async def registered_versions(
    server: dict,
    server_path: str,
) -> list[tuple[str, dict]]:
    """The active document (id ``""``) followed by every linked inactive version.

    The one linkage policy for routing (nginx) and approvals (consent/PAT): a
    linked id that does not resolve to a document of this server is skipped and
    logged, never routed and never approved. Skipping gives up no safety -- the
    vend refuses an unlinked version id -- and keeps one stale link from making
    the whole server unconnectable while it stays routable.
    """
    versions: list[tuple[str, dict]] = [("", server)]
    for version_id in server.get("other_version_ids") or []:
        version = await linked_version(server, server_path, version_id)
        if version is None:
            # Debug, not warning: nginx re-renders every few seconds, so a single
            # stale link would otherwise flood the log; it is never routed anyway.
            logger.debug("Skipping unavailable linked version %r of %s", version_id, server_path)
            continue
        versions.append((version_id, version))
    return versions


async def registered_destinations(
    server: dict,
    server_path: str,
) -> list[str]:
    """Every exact URL a request to this server can currently be sent to (sorted).

    Snapshot on credential write. Sorted so the stored value is stable across
    replicas and diffable in an operator dump. A linked version without an
    endpoint is unroutable and contributes nothing.

    Raises:
        ValueError: the active version has no endpoint (nothing can be approved).
    """
    destinations: set[str] = set()
    for version_id, version in await registered_versions(server, server_path):
        for virtual_backend in (False, True):
            try:
                destinations.add(selected_upstream(version, virtual_backend))
            except ValueError:
                if not version_id:
                    raise
                logger.warning("Skipping linked version without endpoint of %s", server_path)
    return sorted(destinations)
