"""Fetch and check OpenAPI specs before a target is stored (issue #2022).

Every check fails closed: a target is stored only after its URLs pass the SSRF
guard, its spec is fetched within the size cap, parses as OpenAPI 3.x without
YAML aliases, and converts into an acceptable set of tools.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import (
    UTC,
    datetime,
)
from typing import Any

import httpx
import yaml

from ..common.log_redaction import redact_url
from ..exceptions import UrlValidationError
from ..utils.url_guard import (
    PROXY_PROFILE,
    guarded_async_client,
    validate_url,
)
from .openapi_errors import (
    OpenApiServerError,
    unprocessable,
)

logger = logging.getLogger(__name__)


SPEC_FETCH_TIMEOUT_SECONDS: float = 15.0
MAX_TOOL_NAME_CHARS: int = 64
LOCAL_REF_PREFIX: str = "#"
MAX_REPORTED_REFS: int = 5
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
SPEC_ACCEPT_HEADER: str = (
    "application/json, application/yaml, application/x-yaml, text/yaml, */*;q=0.5"
)


@dataclass
class FetchedSpec:
    """A spec that passed every check before conversion."""

    raw: bytes
    content_type: str
    spec: dict[str, Any]
    sha256: str
    title: str
    version: str
    openapi_version: str


class _NoAliasSafeLoader(yaml.SafeLoader):
    """A SafeLoader that refuses YAML aliases.

    A small file built from nested aliases expands into gigabytes once anything
    walks it. Real OpenAPI specs reuse definitions with $ref, not YAML aliases.
    """

    def compose_node(self, parent, index):  # noqa: ANN001, ANN201 - PyYAML signature
        if self.check_event(yaml.AliasEvent):
            event = self.peek_event()
            raise yaml.composer.ComposerError(
                None, None, "YAML aliases are not allowed", event.start_mark
            )
        return super().compose_node(parent, index)


def validate_target_url(
    url: str,
    field_name: str,
) -> None:
    """Run the SSRF guard on a spec URL or base URL, as for any registered backend."""
    try:
        validate_url(url, profile=PROXY_PROFILE, reject_nginx_metacharacters=True)
    except UrlValidationError as e:
        logger.warning(f"OpenAPI target {field_name} blocked: {redact_url(url)}")
        raise unprocessable(
            "url_blocked",
            f"{field_name} is not allowed by the SSRF guard ({e.reason}). Use a public URL, "
            "or ask an operator to add the host to SSRF_ALLOWED_HOSTS or SSRF_ALLOWED_CIDRS.",
            {"field": field_name},
        ) from e


async def _read_capped(
    response: httpx.Response,
    max_bytes: int,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise unprocessable(
                "spec_too_large",
                f"the spec is larger than OPENAPI_MCP_SPEC_MAX_BYTES ({max_bytes} bytes)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _count_fetch(
    result: str,
) -> None:
    try:
        from ..observability.meters import openapi_mcp_spec_fetch_total

        openapi_mcp_spec_fetch_total.add(1, {"result": result})
    except Exception:  # noqa: BLE001 - metrics are optional
        logger.debug("openapi_mcp spec fetch counter unavailable", exc_info=True)


async def fetch_spec_bytes(
    spec_url: str,
    max_bytes: int,
) -> tuple[bytes, str]:
    """Fetch a spec through the SSRF guard: no credential, no redirects, size-capped."""
    try:
        raw, content_type = await _fetch(spec_url, max_bytes)
    except OpenApiServerError as e:
        _count_fetch(e.code)
        raise
    _count_fetch("ok")
    return raw, content_type


async def _fetch(
    spec_url: str,
    max_bytes: int,
) -> tuple[bytes, str]:
    try:
        async with guarded_async_client(
            profile=PROXY_PROFILE, timeout=SPEC_FETCH_TIMEOUT_SECONDS
        ) as client:
            async with client.stream(
                "GET", spec_url, headers={"Accept": SPEC_ACCEPT_HEADER}, follow_redirects=False
            ) as response:
                if response.status_code != 200:
                    raise OpenApiServerError(
                        502,
                        "spec_fetch_failed",
                        f"the spec host returned HTTP {response.status_code}. Redirects are not "
                        "followed: give the final URL of the spec.",
                    )
                raw = await _read_capped(response, max_bytes)
                return raw, response.headers.get("content-type", "")
    except UrlValidationError as e:
        raise unprocessable(
            "url_blocked",
            f"spec_url is not allowed by the SSRF guard ({e.reason})",
            {"field": "spec_url"},
        ) from e
    except httpx.HTTPError as e:
        logger.warning(f"Spec fetch failed for {redact_url(spec_url)}: {type(e).__name__}")
        raise OpenApiServerError(
            502, "spec_fetch_failed", f"could not fetch the spec ({type(e).__name__})"
        ) from e


def _decode_document(
    raw: bytes,
) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as e:
        raise unprocessable("spec_invalid", "the spec is not valid UTF-8") from e
    try:
        document = yaml.load(text, Loader=_NoAliasSafeLoader)  # nosec B506 - SafeLoader subclass
    except yaml.YAMLError as e:
        raise unprocessable(
            "spec_invalid", f"the spec is not valid JSON or YAML: {str(e)[:300]}"
        ) from e
    except RecursionError as e:
        raise unprocessable("spec_invalid", "the spec is nested too deeply") from e
    # YAML timestamps load as datetime objects; turn everything into plain JSON
    # types so the spec can be pushed to the container as JSON.
    return json.loads(json.dumps(document, default=str))


def _non_local_refs(
    document: object,
) -> list[str]:
    """Every $ref that is not a local "#/..." pointer, found without recursion.

    FastMCP refuses external references today, but that is the library's
    behavior, not ours: a later version could start fetching URLs or reading
    files named in a spec. Refusing them here keeps the guarantee in the registry.
    """
    found: list[str] = []
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            ref = node.get("$ref")
            if "$ref" in node and not (isinstance(ref, str) and ref.startswith(LOCAL_REF_PREFIX)):
                found.append(str(ref)[:200])
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


def parse_spec(
    raw: bytes,
    content_type: str = "",
) -> FetchedSpec:
    """Parse and check a spec. Accepts OpenAPI 3.x in JSON or YAML only."""
    document = _decode_document(raw)
    if not isinstance(document, dict):
        raise unprocessable("spec_invalid", "the spec is not a JSON or YAML object")
    external_refs = _non_local_refs(document)
    if external_refs:
        raise unprocessable(
            "spec_invalid",
            "the spec uses $ref values that are not local '#/...' pointers, such as "
            f"{external_refs[:MAX_REPORTED_REFS]}. Bundle the spec into one document "
            "(for example with 'redocly bundle') and add the target again",
            {"refs": external_refs[:MAX_REPORTED_REFS]},
        )
    if "swagger" in document:
        raise unprocessable(
            "spec_invalid",
            "the spec is Swagger 2.0; convert it to OpenAPI 3 (for example with "
            "https://converter.swagger.io) and add the target again",
        )
    version = str(document.get("openapi", ""))
    if not version.startswith("3."):
        raise unprocessable(
            "spec_invalid",
            "the document has no 'openapi: 3.x' field, so it is not an OpenAPI 3 spec",
        )
    if not isinstance(document.get("paths"), dict) or not document["paths"]:
        raise unprocessable("spec_invalid", "the spec has no paths, so it would produce no tools")
    info = document.get("info") if isinstance(document.get("info"), dict) else {}
    return FetchedSpec(
        raw=raw,
        content_type=content_type,
        spec=document,
        sha256=hashlib.sha256(raw).hexdigest(),
        title=str(info.get("title", ""))[:200],
        version=str(info.get("version", ""))[:50],
        openapi_version=version[:20],
    )


def check_tool_count(
    tools: list[str],
    tags: dict[str, int],
    max_tools: int,
) -> None:
    """Refuse a target that would produce more tools than the cap."""
    if len(tools) <= max_tools:
        return
    raise unprocessable(
        "too_many_tools",
        f"the spec would produce {len(tools)} tools; the limit is {max_tools} "
        "(OPENAPI_MCP_MAX_TOOLS_PER_TARGET). Set include_tags or include_operations "
        "to choose the operations you need.",
        {"count": len(tools), "limit": max_tools, "tags": tags},
    )


def check_tool_names(
    tools: list[str],
) -> None:
    """Refuse tool names that are too long, use odd characters, or collide."""
    too_long = [name for name in tools if len(name) > MAX_TOOL_NAME_CHARS]
    bad_chars = [name for name in tools if not TOOL_NAME_RE.fullmatch(name)]
    seen: set[str] = set()
    duplicates = sorted({name for name in tools if name in seen or seen.add(name)})
    problems = sorted(set(too_long) | set(bad_chars) | set(duplicates))
    if not problems:
        return
    raise unprocessable(
        "tool_name_invalid",
        f"these tool names are longer than {MAX_TOOL_NAME_CHARS} characters, contain characters "
        "other than letters, digits, '_', '.' or '-', or collide. Rename them with "
        "tool_name_overrides.",
        {"names": problems[:50]},
    )


def now_utc() -> datetime:
    return datetime.now(UTC)
