"""Destination binding for vaulted egress credentials.

The vend path cross-checks the mcp-proxy token's ``upstream_url`` claim against
the server's registered upstream set, but BOTH sides of that comparison read the
LIVE server record.

This module captures the destination set at credential-WRITE time; the
vend path requires the request's destination to be a member of that stored set.
A retarget is therefore a fail-closed MISS (re-consent for ``oauth_user``, the
terminal PAT-missing result for ``pat``). It mirrors the
existing ``StoredToken.client_id`` binding, which forces re-consent when the
operator rotates the provider OAuth app.

Values here are non-secret: they already exist in the server record.
"""

from urllib.parse import urlparse


def base_url(url: str) -> str:
    """scheme://host[:port] of a URL, lowercased -- the comparison surface.

    The mcp_proxy sub-path append is confined to the bound host, so every
    comparison here is on the BASE (scheme+host+port), not the post-append path.
    """
    p = urlparse(url)
    return f"{(p.scheme or '').lower()}://{(p.netloc or '').lower()}"


def registered_upstreams(server: dict) -> set[str]:
    """The legal upstream base-URL set for a server: proxy_pass_url u versions[*]."""
    bases: set[str] = set()
    if server.get("proxy_pass_url"):
        bases.add(base_url(server["proxy_pass_url"]))
    for ver in server.get("versions") or []:
        ppu = (
            ver.get("proxy_pass_url")
            if isinstance(ver, dict)
            else getattr(ver, "proxy_pass_url", None)
        )
        if ppu:
            bases.add(base_url(ppu))
    return bases


def bound_upstreams(server: dict) -> list[str]:
    """The destination set to persist on a credential write (sorted, deterministic).

    Sorted so the stored value is stable across replicas and diffable in an
    operator dump. An empty list means the server had no registered upstream at
    write time; it can never match a vend (the upstream claim is mandatory), so
    that case stays fail-closed.
    """
    return sorted(registered_upstreams(server))
