"""The Claude sandbox domain an MCP App page must declare behind the gateway.

An MCP App page can set ``_meta.ui.domain`` to ask for a stable sandbox origin.
Claude accepts only ``<hash>.claudemcpcontent.com``, where ``<hash>`` is the
first 32 hex characters of the SHA-256 of the connector URL the user added. A
backend computes that value from its own URL, so behind the gateway it is wrong
and Claude refuses to show the page. Claude's rule:
https://claude.com/docs/connectors/building/mcp-apps/troubleshooting

The auth server rewrites the value on a direct route's ``resources/read``
replies (``auth_server/mcp_apps_domain.py``). For a virtual server the registry
computes the values and writes them into the server's mapping file, because the
nginx Lua router has no SHA-256. Both use the functions here, so they cannot
disagree about the rule.
"""

import hashlib
import os

from .request_origin import origin_of_url

CLAUDE_UI_DOMAIN_SUFFIX: str = ".claudemcpcontent.com"
CLAUDE_UI_DOMAIN_HASH_CHARS: int = 32
# Deployment settings that may hold the gateway's public URL, in order of
# preference. Never the request's Host or forwarded headers. REGISTRY_URL comes
# first: MCP routes are served by nginx on the registry's host, while
# AUTH_SERVER_EXTERNAL_URL can name the auth server itself (the .env.example
# default is http://localhost:8888).
GATEWAY_PUBLIC_URL_ENV_VARS: tuple[str, ...] = ("REGISTRY_URL", "AUTH_SERVER_EXTERNAL_URL")


def claude_ui_domain(
    url: str,
) -> str:
    """Return the ui.domain value Claude expects for a connector URL."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return digest[:CLAUDE_UI_DOMAIN_HASH_CHARS] + CLAUDE_UI_DOMAIN_SUFFIX


def gateway_public_origins() -> list[str]:
    """Return the gateway's configured public origins, most preferred first.

    Returns:
        Distinct origins such as "https://gateway.example.com". Empty when
        neither setting holds an absolute http or https URL.
    """
    origins: list[str] = []
    for name in GATEWAY_PUBLIC_URL_ENV_VARS:
        origin = origin_of_url(os.environ.get(name, ""))
        if origin and origin not in origins:
            origins.append(origin)
    return origins
