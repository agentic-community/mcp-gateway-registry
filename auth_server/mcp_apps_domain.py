"""Rewrite an MCP Apps page's Claude sandbox domain to match the gateway URL.

An MCP App page can set ``_meta.ui.domain`` to ask for a stable sandbox origin.
Claude accepts only ``<hash>.claudemcpcontent.com``, where ``<hash>`` is the
first 32 hex characters of the SHA-256 of the connector URL the user added.
A backend computes that value from its own URL. Behind the gateway the user
connects to the gateway's URL instead, so Claude sees a mismatch and refuses
to show the page.

This module recomputes the value for the gateway URL on ``resources/read``
replies. It touches only values already in Claude's format, so other hosts'
domains pass through unchanged. Claude's rule:
https://claude.com/docs/connectors/building/mcp-apps/troubleshooting
"""

import hashlib
import json
import logging
import secrets
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from registry.common.request_origin import origin_of_url

logger = logging.getLogger(__name__)

CLAUDE_UI_DOMAIN_SUFFIX: str = ".claudemcpcontent.com"
CLAUDE_UI_DOMAIN_HASH_CHARS: int = 32
# Only current nginx templates set this marker and replace the URL header.
# The ordinary validation marker alone cannot authenticate these newer fields.
CONNECTOR_BINDING_HEADER: str = "X-Mcp-Connector-Binding"
CONNECTOR_URL_HEADER: str = "X-Mcp-Connector-URL"
RESOURCES_READ_METHOD: str = "resources/read"
SSE_DATA_PREFIX: str = "data:"


def _rewrite_payload(
    payload: Any,
    new_domain: str,
) -> int:
    """Replace Claude-format ui.domain values in one JSON-RPC reply, in place.

    Args:
        payload: A decoded JSON-RPC reply.
        new_domain: The domain Claude expects for the gateway URL.

    Returns:
        How many values were replaced.
    """
    if not isinstance(payload, dict):
        return 0
    result = payload.get("result")
    if not isinstance(result, dict):
        return 0
    contents = result.get("contents")
    if not isinstance(contents, list):
        return 0
    replaced = 0
    for item in contents:
        replaced += _rewrite_content_item(item, new_domain)
    return replaced


def _rewrite_content_item(
    item: Any,
    new_domain: str,
) -> int:
    """Replace one resource item's ui.domain when it is in Claude's format."""
    if not isinstance(item, dict):
        return 0
    meta = item.get("_meta")
    if not isinstance(meta, dict):
        return 0
    ui = meta.get("ui")
    if not isinstance(ui, dict):
        return 0
    domain = ui.get("domain")
    if not isinstance(domain, str) or not domain.endswith(CLAUDE_UI_DOMAIN_SUFFIX):
        return 0
    if domain == new_domain:
        return 0
    ui["domain"] = new_domain
    return 1


def _rewrite_json_body(
    body_text: str,
    new_domain: str,
) -> str | None:
    """Rewrite a plain JSON reply. Returns None when nothing changed."""
    payload = json.loads(body_text)
    if _rewrite_payload(payload, new_domain) == 0:
        return None
    return json.dumps(payload)


def _rewrite_sse_body(
    body_text: str,
    new_domain: str,
) -> str | None:
    """Rewrite each SSE data line that carries a reply. Returns None when nothing changed.

    The event framing is kept line for line, as the tools/list filter does. A
    data line that is not JSON is kept as it is and the other lines are still
    rewritten.
    """
    out: list[str] = []
    replaced = 0
    for line in body_text.split("\n"):
        stripped = line.lstrip()
        if not stripped.startswith(SSE_DATA_PREFIX):
            out.append(line)
            continue
        data = stripped[len(SSE_DATA_PREFIX) :].strip()
        if not data:
            out.append(line)
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            out.append(line)
            continue
        count = _rewrite_payload(payload, new_domain)
        if count == 0:
            out.append(line)
            continue
        replaced += count
        indent = line[: len(line) - len(stripped)]
        out.append(f"{indent}{SSE_DATA_PREFIX} {json.dumps(payload)}")
    if replaced == 0:
        return None
    return "\n".join(out)


def trusted_connector_url(
    headers: Mapping[str, str],
    marker_secret: str,
) -> str:
    """Read the exact public URL asserted by the selecting nginx location.

    The location joins its configured public origin with the original request
    URI before nginx rewrites the internal proxy path. Missing or old templates
    leave the page unchanged rather than guessing a connector URL.
    """
    supplied_marker = headers.get(CONNECTOR_BINDING_HEADER, "")
    if not marker_secret or not secrets.compare_digest(
        supplied_marker.encode("utf-8"), marker_secret.encode("utf-8")
    ):
        return ""
    url = headers.get(CONNECTOR_URL_HEADER, "")
    if not url or any(ord(char) <= 32 or ord(char) == 127 for char in url):
        return ""
    if not origin_of_url(url):
        return ""
    parsed = urlsplit(url)
    if parsed.fragment or "#" in url or not parsed.path.startswith("/"):
        return ""
    return url


def claude_ui_domain(
    url: str,
) -> str:
    """Return the ui.domain value Claude expects for a connector URL."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return digest[:CLAUDE_UI_DOMAIN_HASH_CHARS] + CLAUDE_UI_DOMAIN_SUFFIX


def rewrite_ui_domains(
    body_bytes: bytes,
    is_sse: bool,
    new_domain: str,
) -> bytes | None:
    """Rewrite Claude-format ui.domain values in a resources/read reply.

    Any problem leaves the reply to the caller unchanged: the value only
    decides whether Claude shows the page, so a reply this function cannot
    read is forwarded as the backend sent it.

    Args:
        body_bytes: The upstream reply body.
        is_sse: True when the reply is text/event-stream.
        new_domain: The domain Claude expects for the gateway URL.

    Returns:
        The rewritten body, or None when nothing was rewritten.
    """
    try:
        body_text = body_bytes.decode("utf-8")
        if is_sse:
            rewritten = _rewrite_sse_body(body_text, new_domain)
        else:
            rewritten = _rewrite_json_body(body_text, new_domain)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning(f"MCP Apps ui.domain rewrite skipped: reply is not readable ({exc})")
        return None
    if rewritten is None:
        return None
    return rewritten.encode("utf-8")
