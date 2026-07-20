"""
A2A Agent API routes for MCP Gateway Registry.

This module provides REST API endpoints for agent registration, discovery,
and management following the A2A protocol specification.

Based on: docs/design/a2a-protocol-integration.md
"""

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Annotated, Any

import httpx
from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from ..audit import set_audit_action
from ..auth.asset_permissions import user_has_asset_permission
from ..auth.csrf import verify_csrf_token_flexible
from ..auth.dependencies import nginx_proxied_auth
from ..common.log_redaction import redact_headers, redact_mapping
from ..core.config import settings
from ..core.metrics import ASSET_ID_SUPPLIED_TOTAL
from ..exceptions import AssetIdConflictError, UrlValidationError
from ..repositories.factory import get_search_repository
from ..repositories.interfaces import SearchRepositoryBase
from ..schemas.agent_models import (
    REGISTRANT_ONLY_FIELDS,
    AgentBatchItem,
    AgentBatchRequest,
    AgentCard,
    AgentCardPatch,
    AgentInfo,
    AgentProvider,
    AgentRegistrationRequest,
    PullCardFieldChange,
    PullCardResponse,
)
from ..schemas.duplicate_check_models import (
    AgentDuplicateCheckRequest,
    DuplicateCheckResult,
)
from ..services._asset_id import (
    InvalidAssetIdError,
    check_caller_supplied_id_allowed,
    resolve_asset_id,
)
from ..services.agent_batch_service import (
    ConcurrentJobLimitError,
    agent_batch_service,
)
from ..services.agent_service import agent_service
from ..services.duplicate_check_service import get_duplicate_check_service
from ..services.lifecycle_events import (
    EnforcedStatusError,
    enforce_registration_status,
    fire_scan_complete_event,
    user_can_change_lifecycle_status,
)
from ..services.registration_gate_service import check_registration_gate
from ..services.webhook_service import send_registration_webhook
from ..utils.metadata import flatten_metadata_to_text
from ..utils.request_utils import get_client_ip
from ..utils.url_guard import PROXY_PROFILE, guarded_async_client, validate_agent_url
from ._etag_utils import (
    parse_if_match,
    updated_ms,
    weak_etag_for_timestamp,
)


def get_search_repo() -> SearchRepositoryBase:
    """Get search repository instance."""
    return get_search_repository()


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


router = APIRouter()


async def _perform_agent_security_scan_on_registration(
    path: str,
    agent_card: AgentCard,
    agent_card_dict: dict,
) -> bool:
    """Perform security scan on newly registered agent.

    Handles the complete security scan workflow including:
    - Running the security scan with configured analyzers
    - Adding security-pending tag if scan fails
    - Disabling agent if configured and scan fails
    - Updating search index with disabled state if agent disabled

    All scan failures are non-fatal and will be logged but not raised.

    Args:
        path: Agent path (e.g., /code-reviewer)
        agent_card: AgentCard Pydantic model instance
        agent_card_dict: Agent card as dictionary for scanning

    Returns:
        bool: True if agent should remain enabled, False if disabled due to scan
    """
    from ..services.agent_scanner import agent_scanner_service

    scan_config = agent_scanner_service.get_scan_config()
    if not (scan_config.enabled and scan_config.scan_on_registration):
        return True  # Agent remains enabled

    logger.info(f"Running A2A security scan for newly registered agent: {path}")

    try:
        # Run the security scan
        scan_result = await agent_scanner_service.scan_agent(
            agent_card=agent_card_dict,
            agent_path=path,
            analyzers=scan_config.analyzers,
            api_key=scan_config.llm_api_key,
            timeout=scan_config.scan_timeout_seconds,
        )

        # Handle unsafe agents
        if not scan_result.is_safe:
            logger.warning(
                f"Agent {path} failed security scan. "
                f"Critical: {scan_result.critical_issues}, High: {scan_result.high_severity}"
            )

            # Add security-pending tag if configured
            if scan_config.add_security_pending_tag:
                current_tags = agent_card.tags or []
                if "security-pending" not in current_tags:
                    current_tags.append("security-pending")
                    agent_card.tags = current_tags
                    # Issue #1033: update_agent (not register_agent) is the
                    # right call here. The agent was already inserted by the
                    # caller; calling register_agent again raises
                    # "path '...' already exists" which silently swallows
                    # the security-pending tag and leaves the agent without
                    # the warning it earned.
                    agent_info = await agent_service.get_agent_info(path)
                    if agent_info:
                        # update_agent takes a dict of fields to merge (it calls
                        # updates.get(...)); pass the dict directly, not an
                        # AgentCard instance (which has no .get and raises
                        # AttributeError). Only the changed field is needed.
                        await agent_service.update_agent(path, {"tags": current_tags})
                    logger.info(f"Added 'security-pending' tag to agent {path}")

            # Disable agent if configured
            if scan_config.block_unsafe_agents:
                await agent_service.toggle_agent(path, False)
                logger.warning(f"Disabled agent {path} due to failed security scan")

                # Update search index with disabled state.
                # Issue #1033 follow-up: index_agent expects an AgentCard
                # model (it reads .name / .description / .tags / .skills);
                # passing the raw dict here raised
                # `'dict' object has no attribute 'name'`. Use the model
                # we already have, which carries the security-pending tag
                # added a few lines above.
                search_repo = get_search_repository()
                await search_repo.index_agent(path, agent_card, is_enabled=False)
                # scan_complete webhook (Issue #1330): agent was disabled.
                fire_scan_complete_event(
                    agent_card.model_dump(),
                    scan_result,
                    auto_disabled=True,
                    registration_type="agent",
                )
                return False  # Agent disabled

        else:
            logger.info(f"Agent {path} passed security scan")

        # scan_complete webhook (Issue #1330): safe, or unsafe-but-not-blocked.
        fire_scan_complete_event(
            agent_card.model_dump(),
            scan_result,
            auto_disabled=False,
            registration_type="agent",
        )
        return True  # Agent remains enabled

    except Exception as e:
        logger.error(f"Failed to run security scan for agent {path}: {e}")
        # Non-fatal error - agent is registered but not scanned. Still notify
        # consumers so they are not left polling for a scan that never reports.
        fire_scan_complete_event(
            agent_card_dict,
            None,
            scan_error=f"{type(e).__name__}: {e}",
            registration_type="agent",
        )
        return True  # Agent remains enabled on scan error


class RatingRequest(BaseModel):
    rating: int


def _build_agent_health_urls(
    base_url: str,
) -> list[str]:
    """Build health check URLs for an A2A agent in priority order.

    Per the A2A spec, there is no /ping endpoint. Agent availability
    is determined by fetching the agent card at /.well-known/agent-card.json.
    Falls back to the registered URL for non-A2A agents.

    Args:
        base_url: The agent's registered URL (e.g., https://agent.example.com/a2a)

    Returns:
        List of URLs to try in order (agent card first, then registered URL)
    """
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    agent_card_url = f"{parsed.scheme}://{parsed.netloc}/.well-known/agent-card.json"
    return [agent_card_url, base_url]


async def _probe_agent_backend_health(
    path: str,
    agent_card: AgentCard,
) -> dict[str, Any]:
    """Probe an A2A agent's real backend and return its health result.

    Tries GET on the agent-card URL(s) first, then a HEAD ping fallback (any HTTP
    response means reachable). In reverse-proxy mode the advertised ``url`` is the
    gateway, so the registrant's backend lives in ``proxy_pass_url``; fall back to
    ``url`` for agents registered before the flag was on. Shared by the manual
    ``/health`` route and the enable/register flow so a freshly enabled agent gets
    a real status (not the default "unknown", which would block its nginx block).

    Args:
        path: Normalized agent path (for logging).
        agent_card: The agent whose backend is probed.

    Returns:
        Dict with keys: status ("healthy"/"unhealthy"), status_code, detail,
        response_time_ms, health_check_url, last_checked (datetime),
        last_checked_iso.
    """
    backend_url = getattr(agent_card, "proxy_pass_url", None) or agent_card.url
    base_url = str(backend_url).rstrip("/")
    health_urls = _build_agent_health_urls(base_url)
    timeout_seconds = max(1, settings.health_check_timeout_seconds)

    status_label = "unhealthy"
    detail = None
    status_code = None
    response_time_ms = None
    health_check_url = health_urls[0]

    for url in health_urls:
        health_check_url = url
        start_time = datetime.now(UTC)
        try:
            async with guarded_async_client(
                profile=PROXY_PROFILE, timeout=timeout_seconds
            ) as client:
                response = await client.get(url)
            status_code = response.status_code
            response_time_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)
            if response.status_code == 200:
                status_label = "healthy"
                detail = None
                logger.info(f"Agent health check for {path} succeeded via GET on {url}")
                break
            detail = f"Agent responded with HTTP {response.status_code}"
            logger.debug(f"Agent health check for {path} got HTTP {response.status_code} on {url}")
        except httpx.TimeoutException:
            detail = f"Health check timed out on {url}"
            logger.debug(f"Agent health check for {path} timed out on {url}")
        except httpx.HTTPError as exc:
            detail = f"Health check failed on {url}"
            logger.debug(f"Agent health check for {path} failed on {url}: {exc}")
        except Exception as exc:
            detail = f"Unexpected health check error on {url}"
            logger.debug(f"Agent health check for {path} unexpected error on {url}: {exc}")

    # Fallback: if GET-based checks failed, try HEAD on the base URL. A
    # non-connection-error response (even 401/403) means the server is reachable.
    if status_label == "unhealthy":
        logger.info(f"Agent {path} GET checks failed, falling back to HEAD ping on {base_url}")
        try:
            start_time = datetime.now(UTC)
            async with guarded_async_client(
                profile=PROXY_PROFILE, timeout=timeout_seconds
            ) as client:
                response = await client.head(base_url)
            status_code = response.status_code
            response_time_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)
            health_check_url = base_url
            status_label = "healthy"
            detail = f"Reachable via HEAD (HTTP {response.status_code})"
            logger.info(
                f"Agent health check for {path} succeeded via HEAD ping "
                f"(HTTP {response.status_code})"
            )
        except httpx.TimeoutException:
            logger.debug(f"Agent {path} HEAD ping timed out on {base_url}")
        except httpx.HTTPError as exc:
            logger.debug(f"Agent {path} HEAD ping failed on {base_url}: {exc}")
        except Exception as exc:
            logger.debug(f"Agent {path} HEAD ping unexpected error on {base_url}: {exc}")

    last_checked = datetime.now(UTC)
    return {
        "status": status_label,
        "status_code": status_code,
        "detail": detail,
        "response_time_ms": response_time_ms,
        "health_check_url": health_check_url,
        "last_checked": last_checked,
        "last_checked_iso": last_checked.isoformat(),
    }


async def _refresh_agent_health(
    path: str,
) -> None:
    """Probe an enabled A2A agent and persist its health so nginx will route it.

    A freshly registered/enabled agent defaults to health_status "unknown", which
    the nginx generator treats as not-healthy and therefore emits no proxy block.
    Nothing else flips it automatically (the periodic health loop only covers MCP
    servers), so an enabled agent would be silently unroutable until a manual
    ``/health`` call. Call this after enabling/registering-enabled so the block is
    generated. Best-effort: failures are logged, never raised.

    Args:
        path: Normalized agent path.
    """
    try:
        agent_card = await agent_service.get_agent_info(path)
        if not agent_card:
            return
        result = await _probe_agent_backend_health(path, agent_card)
        await agent_service.update_agent(
            path,
            {
                "health_status": result["status"],
                "last_health_check": result["last_checked"],
            },
        )
        logger.info(f"Agent '{path}' health refreshed on enable: {result['status']}")
    except Exception as e:
        logger.warning(f"Failed to refresh health for agent {path} on enable: {e}")


# A2A-spec fields the pull-card diff considers. Registry-extension fields
# (tags, ratings, visibility, trust_level, etc.) are deliberately excluded so a
# remote card can never overwrite registry-managed state.
A2A_SPEC_FIELDS: set[str] = {
    "protocol_version",
    "name",
    "description",
    "url",
    "version",
    "capabilities",
    "default_input_modes",
    "default_output_modes",
    "skills",
    "preferred_transport",
    "provider",
    "icon_url",
    "documentation_url",
    "security_schemes",
    "security",
    "supports_authenticated_extended_card",
}

# Remote A2A cards use camelCase (A2A spec); our model uses snake_case.
A2A_CAMEL_TO_SNAKE: dict[str, str] = {
    "protocolVersion": "protocol_version",
    "defaultInputModes": "default_input_modes",
    "defaultOutputModes": "default_output_modes",
    "preferredTransport": "preferred_transport",
    "iconUrl": "icon_url",
    "documentationUrl": "documentation_url",
    "securitySchemes": "security_schemes",
    "supportsAuthenticatedExtendedCard": "supports_authenticated_extended_card",
}

# Maximum size (bytes) we will read from a remote agent card. The card is
# attacker-influenced (the agent owner hosts it), so we cap the read to avoid
# a memory-exhaustion vector. 1 MiB is far larger than any real agent card.
MAX_REMOTE_CARD_BYTES: int = 1_048_576


async def _fetch_remote_agent_card(
    base_url: str,
) -> tuple[dict[str, Any], str]:
    """Fetch the remote A2A agent card from the well-known endpoint.

    Reads the response with a hard size cap (MAX_REMOTE_CARD_BYTES) before
    parsing, since the remote card is hosted by the agent owner and is not
    trusted input.

    Args:
        base_url: The agent's registered URL

    Returns:
        Tuple of (parsed card dict, URL that was fetched)

    Raises:
        HTTPException: 502 if fetch fails or the payload is too large
    """
    import json

    # Fail-closed pre-check: reject a caller-supplied base_url that targets a
    # private/metadata address or bad scheme before any outbound fetch. The
    # guarded client below additionally re-validates and pins at connect time.
    validate_agent_url(base_url)

    urls = _build_agent_health_urls(base_url)
    agent_card_url = urls[0]
    timeout_seconds = max(1, settings.health_check_timeout_seconds)

    try:
        async with guarded_async_client(profile=PROXY_PROFILE, timeout=timeout_seconds) as client:
            response = await client.get(agent_card_url)

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Remote agent returned HTTP {response.status_code} from {agent_card_url}",
            )

        # S1: enforce a size limit before parsing untrusted JSON.
        content = response.content
        if len(content) > MAX_REMOTE_CARD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Remote agent card from {agent_card_url} exceeds {MAX_REMOTE_CARD_BYTES} bytes"
                ),
            )

        remote_card = json.loads(content)
        if not isinstance(remote_card, dict):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Remote agent card from {agent_card_url} is not a JSON object",
            )
        return remote_card, agent_card_url

    except HTTPException:
        raise
    except UrlValidationError as exc:
        logger.warning("Agent card fetch blocked by SSRF guard for %s: %s", agent_card_url, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent URL failed SSRF validation and cannot be fetched",
        ) from exc
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Timeout fetching agent card from {agent_card_url}",
        )
    except httpx.HTTPError as exc:
        # Do not reflect the httpx error (leaks resolved hosts / internal
        # network detail); log it and return the URL only.
        logger.warning("Failed to fetch agent card from %s: %s", agent_card_url, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch agent card from {agent_card_url}",
        )
    except Exception as exc:
        logger.warning("Invalid response from %s: %s", agent_card_url, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Invalid response from {agent_card_url}",
        )


def _normalize_remote_card_keys(
    remote_card: dict[str, Any],
) -> dict[str, Any]:
    """Convert camelCase keys from a remote A2A card to snake_case.

    Only known A2A-spec fields are converted; unknown keys pass through unchanged.
    """
    normalized = {}
    for key, value in remote_card.items():
        snake_key = A2A_CAMEL_TO_SNAKE.get(key, key)
        normalized[snake_key] = value
    return normalized


def _compute_card_diff(
    current_agent: AgentCard,
    remote_card: dict[str, Any],
) -> list[PullCardFieldChange]:
    """Compute a field-by-field diff between the local agent and a remote card.

    Only A2A-spec fields are compared. Registry-extension fields are ignored,
    even if the remote card echoes them back.
    """
    current_dict = current_agent.model_dump()
    changes: list[PullCardFieldChange] = []

    for field_name in A2A_SPEC_FIELDS:
        if field_name not in remote_card:
            continue

        current_value = current_dict.get(field_name)
        remote_value = remote_card[field_name]

        # Normalize Pydantic-dumped sub-objects to match how A2A remotes serialize.
        # Remotes typically omit unset nullable fields (examples / input_modes /
        # output_modes / security on AgentSkill, etc.) while the local-side
        # current_dict = current_agent.model_dump() above includes them as
        # explicit nulls. The drop-nulls step below makes both sides comparable
        # so a skill list with any unset nullable field doesn't report a
        # spurious diff.
        def _drop_nulls(d: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in d.items() if v is not None}

        if field_name == "provider" and isinstance(current_value, dict):
            current_value = _drop_nulls(current_value)
            if isinstance(remote_value, dict):
                remote_value = _drop_nulls(remote_value)

        if field_name == "skills":
            if isinstance(current_value, list):
                current_value = [
                    _drop_nulls(s) if isinstance(s, dict) else s for s in current_value
                ]
                # Sort by stable key so a reordered-but-equal remote list does
                # not produce a spurious change.
                current_value = sorted(
                    current_value,
                    key=lambda s: (s.get("id") if isinstance(s, dict) else "") or "",
                )
            if isinstance(remote_value, list):
                remote_value = [_drop_nulls(s) if isinstance(s, dict) else s for s in remote_value]
                remote_value = sorted(
                    remote_value,
                    key=lambda s: (s.get("id") if isinstance(s, dict) else "") or "",
                )

        if field_name == "security_schemes" and isinstance(current_value, dict):
            current_value = {
                k: _drop_nulls(v) if isinstance(v, dict) else v for k, v in current_value.items()
            }
            if isinstance(remote_value, dict):
                remote_value = {
                    k: _drop_nulls(v) if isinstance(v, dict) else v for k, v in remote_value.items()
                }

        if current_value != remote_value:
            changes.append(
                PullCardFieldChange(
                    field=field_name,
                    current_value=current_value,
                    remote_value=remote_value,
                )
            )

    return changes


def _build_safe_card_updates(
    changes: list[PullCardFieldChange],
) -> dict[str, Any]:
    """Build the field-update dict to apply from a pull-card diff.

    Only A2A-spec fields appear in the diff (see _compute_card_diff), but we
    defensively reject anything in REGISTRANT_ONLY_FIELDS so the pull path can
    never overwrite registry-managed state. This keeps a single source of truth
    for "not client-writable" shared with the PATCH endpoint (S5).

    Raises:
        HTTPException: 400 if a registrant-only field is present in the diff.
    """
    updates: dict[str, Any] = {}
    for change in changes:
        if change.field in REGISTRANT_ONLY_FIELDS:
            # Should never happen: A2A_SPEC_FIELDS and REGISTRANT_ONLY_FIELDS are
            # disjoint. Fail loud rather than silently writing protected state.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Refusing to apply registrant-only field '{change.field}' from remote card",
            )
        updates[change.field] = change.remote_value
    return updates


def _normalize_path(
    path: str | None,
    agent_name: str | None = None,
) -> str:
    """
    Normalize agent path format.

    If path is None, derives it from agent_name by converting to lowercase
    and replacing spaces with hyphens.

    Args:
        path: Agent path to normalize, or None to auto-generate
        agent_name: Agent name used for auto-generating path if needed

    Returns:
        Normalized path string

    Raises:
        ValueError: If path is None and agent_name is not provided
    """
    if path is None:
        if not agent_name:
            raise ValueError("Path is required or agent_name must be provided for auto-generation")
        path = agent_name.lower().replace(" ", "-")

    if not path.startswith("/"):
        path = "/" + path

    if path.endswith("/") and len(path) > 1:
        path = path.rstrip("/")

    return path


def _normalize_tag_list(tags: str | list[str]) -> list[str]:
    """Normalize a tags field that may be a comma-separated string or a list."""
    if isinstance(tags, str):
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    return [str(tag).strip() for tag in tags if str(tag).strip()]


def _gateway_agent_url(
    agent_path: str,
) -> str:
    """Build the gateway-facing A2A URL that clients use to reach an agent.

    When A2A reverse-proxy mode is on, the registry advertises this URL as the
    agent's ``url`` (and keeps the registrant's real backend in
    ``proxy_pass_url``) so discovery routes callers through the gateway. Mirrors
    the nginx location route ``{ROOT_PATH}/agent/<path>/`` with a trailing slash
    so it matches the JSON-RPC prefix location.

    Args:
        agent_path: Normalized agent path with a leading slash (e.g. "/travel").

    Returns:
        Absolute gateway URL, e.g. "https://gateway.example.com/agent/travel/".
    """
    from ..core.nginx_service import AGENT_ROUTE_PREFIX

    base = settings.registry_url.rstrip("/")
    return f"{base}{AGENT_ROUTE_PREFIX}/{agent_path.strip('/')}/"


def _apply_a2a_reverse_proxy_split(
    agent_card: AgentCard,
    path: str,
) -> None:
    """Advertise the gateway url and keep the registrant backend in proxy_pass_url.

    In A2A reverse-proxy mode the stored ``url`` is the gateway-facing address and
    the registrant's real backend lives in ``proxy_pass_url``. This mutates the
    card in place so discovery routes callers through the gateway. It is idempotent
    and MUST run on every write path (register, PUT, PATCH), not just register, or
    an edit that changes ``url`` would desync the advertised url from the backend.

    Only applies when ``a2a_reverse_proxy_effective`` (flag AND with-gateway) and
    the agent speaks a2a; otherwise the card is left untouched (registry-only mode
    keeps url == backend, and non-a2a agents are never proxied). When the card's
    ``url`` already points at the gateway (an edit that did not touch the backend),
    the existing ``proxy_pass_url`` is preserved rather than overwritten with the
    gateway url.

    Args:
        agent_card: The card being written; mutated in place.
        path: Normalized agent path with a leading slash (e.g. "/travel").
    """
    if not settings.a2a_reverse_proxy_effective:
        return
    if (agent_card.supported_protocol or "").lower() != "a2a":
        return

    gateway_url = _gateway_agent_url(path)
    if agent_card.url == gateway_url:
        # url already gateway-facing (edit that did not change the backend); keep
        # the stored backend so we do not clobber it with the gateway url.
        return

    agent_card.proxy_pass_url = agent_card.url
    agent_card.url = gateway_url
    logger.info(
        f"A2A reverse-proxy: agent '{path}' advertised url set to gateway "
        f"({agent_card.url}); backend preserved in proxy_pass_url."
    )


def _weak_etag_for(agent_card: AgentCard) -> str:
    """Weak ETag derived from updated_at epoch milliseconds.

    Thin wrapper over the shared helper to preserve agent-specific call sites.
    """
    return weak_etag_for_timestamp(agent_card.updated_at, agent_card.registered_at)


def _parse_if_match(if_match: str | None) -> int | None:
    """Parse a weak If-Match header. Thin wrapper over the shared helper."""
    return parse_if_match(if_match)


def _agent_updated_ms(agent_card: AgentCard) -> int:
    """Epoch-ms of the card's updated_at (or registered_at fallback, else 0).

    Thin wrapper over the shared helper to preserve agent-specific call sites.
    """
    return updated_ms(agent_card.updated_at, agent_card.registered_at)


def _hash_items(items: list[AgentBatchItem]) -> str:
    """Stable SHA-256 over canonicalized items, for idempotency auditing."""
    payload = json.dumps(
        [item.model_dump(mode="json") for item in items],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_agent_permission(
    action: str,
    agent_name: str,
    user_context: dict[str, Any],
) -> None:
    """
    Check if user has permission for an agent operation.

    Takes a logical ACTION (list/get/create/modify/delete/toggle) and resolves
    it to the correct agent scope via the canonical asset-permission map. This
    replaces the previous raw-scope-string argument, which had let agent toggle
    and modify be gated on the SERVER scopes ``toggle_service`` / ``modify_service``
    (a cross-asset privilege bleed: a server grant leaked agent control, and the
    intended ``modify_agent`` grant was ignored). Passing an action keeps the
    enforced scope in lockstep with the family so this cannot recur.

    Args:
        action: Logical agent action (e.g. "modify", "toggle", "delete").
        agent_name: Name of the agent (used for the 403 detail and per-resource
            grant match).
        user_context: User context from auth.

    Raises:
        HTTPException: 403 if the user lacks the permission.
    """
    if not user_has_asset_permission("agent", action, agent_name, user_context):
        logger.warning(
            f"User {user_context['username']} attempted to {action} "
            f"agent {agent_name} without permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to {action} agent {agent_name}",
        )


def _check_agent_lifecycle_status_permission(
    existing_agent,
    merged_agent,
    user_context: dict[str, Any],
) -> None:
    """Gate agent lifecycle status changes on change_lifecycle_status (Issue #1330).

    No-op when the status is unchanged. Admins always pass; other users need the
    'change_lifecycle_status' permission for the agent.

    Raises:
        HTTPException: 403 if the user may not change lifecycle status.
    """
    old_status = (getattr(existing_agent, "status", None) or "active").lower()
    new_status = (getattr(merged_agent, "status", None) or "active").lower()
    if new_status == old_status:
        return

    if user_can_change_lifecycle_status(existing_agent.name, user_context):
        return

    logger.warning(
        f"User {user_context['username']} attempted to change lifecycle status "
        f"of agent {existing_agent.name} without change_lifecycle_status permission"
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            f"You do not have permission to change the lifecycle status of "
            f"{existing_agent.name}. This action requires the 'change_lifecycle_status' "
            f"permission, which is typically granted to admins."
        ),
    )


def _filter_agents_by_access(
    agents: list[AgentCard],
    user_context: dict[str, Any],
) -> list[AgentCard]:
    """
    Filter agents based on user access permissions.

    Args:
        agents: List of agent cards
        user_context: User context from auth

    Returns:
        Filtered list of agents user can access
    """
    accessible = []
    user_groups = set(user_context.get("groups", []))
    username = user_context["username"]
    is_admin = user_context.get("is_admin", False)

    # Get accessible agents from user context (UI-Scopes)
    accessible_agent_list = user_context.get("accessible_agents", [])
    logger.debug(f"User {username} accessible agents from UI-Scopes: {accessible_agent_list}")

    for agent in agents:
        if is_admin:
            accessible.append(agent)
            continue

        # Check if user has agent-level restrictions from UI-Scopes
        if "all" not in accessible_agent_list and agent.path not in accessible_agent_list:
            logger.debug(
                f"Agent {agent.path} filtered out: not in accessible agents {accessible_agent_list}"
            )
            continue

        if agent.visibility == "public":
            accessible.append(agent)
            continue

        if agent.visibility == "private":
            if agent.registered_by == username:
                accessible.append(agent)
            continue

        if agent.visibility == "group-restricted":
            agent_groups = set(agent.allowed_groups)
            if agent_groups & user_groups:
                accessible.append(agent)
            continue

    return accessible


@router.post(
    "/agents/check-duplicates",
    response_model=DuplicateCheckResult,
    summary="Check whether an agent registration would duplicate an existing one",
)
async def check_agent_duplicates(
    payload: AgentDuplicateCheckRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
) -> DuplicateCheckResult:
    """Advisory duplicate check for agent registrations.

    Always returns 200; the response shape signals matches via
    ``collision_with`` (exact-URL hit on the agent endpoint) and
    ``advisory_matches`` (similarity hits). The endpoint does not
    block registration — callers are free to proceed even when matches
    are returned.
    """
    ui_permissions = user_context.get("ui_permissions", {})
    publish_permissions = ui_permissions.get("publish_agent", [])
    if not publish_permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to register agents",
        )

    if not payload.name.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="name must not be blank",
        )

    service = get_duplicate_check_service()
    return await service.check(
        name=payload.name,
        description=payload.description,
        identity_url=payload.url,
        self_path=payload.self_path,
        user_context=user_context,
    )


@router.post("/agents/register")
async def register_agent(
    http_request: Request,
    request: AgentRegistrationRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """
    Register a new A2A agent in the registry.

    Requires publish_agent scope/permission.

    Args:
        request: Agent registration request data
        user_context: Authenticated user context

    Returns:
        201 with agent card and registration metadata

    Raises:
        HTTPException: 409 if path exists, 422 if validation fails, 403 if unauthorized
    """
    # Set audit action for agent registration
    set_audit_action(
        http_request,
        "create",
        "agent",
        resource_id=request.path,
        description=f"Register agent {request.name}",
    )

    ui_permissions = user_context.get("ui_permissions", {})
    publish_permissions = ui_permissions.get("publish_agent", [])

    if not publish_permissions:
        logger.warning(
            f"User {user_context['username']} attempted to register agent without permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to register agents",
        )

    logger.info(f"Agent registration request from user '{user_context['username']}'")
    logger.info(f"Name: {request.name}, Path: {request.path}, URL: {request.url}")

    path = _normalize_path(request.path, request.name)

    if await agent_service.get_agent_info(path):
        logger.error(f"Agent registration failed: path '{path}' already exists")
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": f"Agent with path '{path}' already exists",
                "suggestion": "Use a different path or update the existing agent",
            },
        )

    tag_list = _normalize_tag_list(request.tags)

    # Parse external_tags
    external_tag_list = []
    if request.external_tags:
        if isinstance(request.external_tags, str):
            external_tag_list = [
                tag.strip() for tag in request.external_tags.split(",") if tag.strip()
            ]
        elif isinstance(request.external_tags, list):
            external_tag_list = [tag.strip() for tag in request.external_tags if tag.strip()]

    # Convert provider dict to AgentProvider object if provided
    provider_obj = None
    if request.provider:
        provider_obj = AgentProvider(
            organization=request.provider.get("organization", ""),
            url=request.provider.get("url", ""),
        )

    # Parse source timestamps
    source_created_dt = None
    if request.source_created_at:
        try:
            source_created_dt = datetime.fromisoformat(
                request.source_created_at.replace("Z", "+00:00")
            )
        except ValueError:
            logger.warning(f"Invalid source_created_at format: {request.source_created_at}")

    source_updated_dt = None
    if request.source_updated_at:
        try:
            source_updated_dt = datetime.fromisoformat(
                request.source_updated_at.replace("Z", "+00:00")
            )
        except ValueError:
            logger.warning(f"Invalid source_updated_at format: {request.source_updated_at}")

    try:
        from ..utils.agent_validator import agent_validator

        # Build optional kwargs for fields that have defaults on AgentCard
        optional_card_kwargs: dict[str, Any] = {}
        if request.default_input_modes:
            optional_card_kwargs["default_input_modes"] = request.default_input_modes
        if request.default_output_modes:
            optional_card_kwargs["default_output_modes"] = request.default_output_modes
        if request.metadata:
            optional_card_kwargs["metadata"] = request.metadata
        # Build capabilities: merge explicit capabilities dict with streaming bool
        capabilities = dict(request.capabilities) if request.capabilities else {}
        if request.streaming and "streaming" not in capabilities:
            capabilities["streaming"] = request.streaming
        if capabilities:
            optional_card_kwargs["capabilities"] = capabilities

        # Enforced-status policy (Issue #1330). Use model_fields_set to tell an
        # omitted status (force to enforced) from an explicit one (reject on
        # mismatch). When the policy is unset, fall back to the request value.
        requested_status = request.status if "status" in request.model_fields_set else None
        try:
            effective_status = enforce_registration_status(requested_status, "agent")
        except EnforcedStatusError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        # Feature-flag gate (#1276): reject a caller-supplied id when the flag is
        # off (fail-closed). Charset/length are already validated by the request
        # model. Federation is not gated here (it does not use this route).
        try:
            check_caller_supplied_id_allowed(request.id, settings.allow_caller_supplied_asset_id)
        except InvalidAssetIdError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid asset id: {e}",
            )

        agent_card = AgentCard(
            protocol_version=request.protocol_version,
            name=request.name,
            description=request.description,
            url=request.url,
            path=path,
            id=resolve_asset_id(request.id),
            version=request.version,
            status=effective_status or request.status,
            provider=provider_obj,
            security_schemes=request.security_schemes or {},
            skills=request.skills or [],
            tags=tag_list,
            license=request.license,
            visibility=request.visibility,
            allowed_groups=request.allowed_groups,
            trust_level=request.trust_level,
            supported_protocol=request.supported_protocol,
            registered_by=user_context["username"],
            source_created_at=source_created_dt,
            source_updated_at=source_updated_dt,
            external_tags=external_tag_list,
            **optional_card_kwargs,
        )

        validation_result = await agent_validator.validate_agent_card(
            agent_card,
            verify_endpoint=True,
        )

        if not validation_result.is_valid:
            logger.error(f"Agent validation failed: {validation_result.errors}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Agent card validation failed",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                },
            )

        # A2A reverse-proxy mode: advertise the gateway-facing URL and keep the
        # registrant's real backend in proxy_pass_url. Done AFTER validation so
        # the endpoint health check above still probes the real backend, not the
        # gateway. Mirrors the MCP-server url/proxy_pass_url split; proxy_pass_url
        # is redacted from non-admin reads (registry/services/visibility.py) and
        # the nginx generator proxies to it. When the flag is off, the card is
        # stored exactly as registered (no proxy_pass_url) -- backwards compatible.
        #
        # Gated on a2a_reverse_proxy_effective (flag AND with-gateway), NOT the
        # raw flag: in registry-only mode there is no gateway to route through, so
        # we must NOT advertise a gateway url that would 503. url and
        # proxy_pass_url stay identical (backend) in that case. The same split is
        # re-applied on the PUT/PATCH update paths via this shared helper.
        _apply_a2a_reverse_proxy_split(agent_card, path)

    except ValueError as e:
        logger.error(f"Invalid agent card data: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid agent card: {str(e)}",
        )

    # Registration gate check (admission control, issue #809)
    gate_result = await check_registration_gate(
        asset_type="agent",
        operation="register",
        source_api="/api/agents/register",
        registration_payload=request.model_dump(mode="json"),
        raw_headers=http_request.scope.get("headers", []),
    )
    if not gate_result.allowed:
        logger.warning(
            f"Registration gate denied agent '{request.name}': {gate_result.error_message}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Registration denied by policy gate: {gate_result.error_message}",
        )

    try:
        success = await agent_service.register_agent(agent_card)
        if success and request.id is not None:
            ASSET_ID_SUPPLIED_TOTAL.labels(asset_type="agent").inc()
    except AssetIdConflictError as e:
        logger.warning(f"Agent registration id conflict: {e}")
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": f"Agent with id '{e.asset_id}' already exists",
                "suggestion": "Use a different id or omit it to auto-generate one",
            },
        )

    if not success:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Failed to save agent data",
                "suggestion": "Check server logs for details",
            },
        )

    is_enabled = await agent_service.is_agent_enabled(path)
    try:
        search_repo = get_search_repository()
        await search_repo.index_agent(path, agent_card, is_enabled)
    except Exception as e:
        logger.warning(f"Failed to update search index for {path}: {e}")

    logger.info(
        f"New agent registered: '{request.name}' at path '{path}' "
        f"by user '{user_context['username']}'"
    )

    # Agent security scanning if enabled
    agent_card_dict = agent_card.model_dump()
    is_enabled = await _perform_agent_security_scan_on_registration(
        path, agent_card, agent_card_dict
    )

    # If registration left the agent enabled, probe its backend and persist health
    # so the nginx reverse-proxy block is generated (default "unknown" would be
    # treated as not-healthy and skipped).
    if is_enabled:
        await _refresh_agent_health(path)

    # Best-effort ANS linking if ans_agent_id is provided
    if request.ans_agent_id and settings.ans_integration_enabled:
        try:
            from ..services.ans_service import link_ans_to_agent

            ans_result = await link_ans_to_agent(
                agent_path=path,
                ans_agent_id=request.ans_agent_id,
                username=user_context["username"],
            )
            if ans_result.get("success"):
                logger.info(f"ANS ID '{request.ans_agent_id}' linked to agent '{path}'")
            else:
                logger.warning(
                    f"Failed to link ANS ID '{request.ans_agent_id}' to agent '{path}': "
                    f"{ans_result.get('message', 'Unknown error')}"
                )
        except Exception as e:
            logger.warning(
                f"ANS linking failed for agent '{path}' with ANS ID '{request.ans_agent_id}': {e}"
            )

    # Registration webhook (Issue #742)
    asyncio.create_task(
        send_registration_webhook(
            event_type="registration",
            registration_type="agent",
            card_data=agent_card.model_dump(mode="json"),
            performed_by=user_context["username"],
        )
    )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "message": "Agent registered successfully",
            "agent": {
                "name": agent_card.name,
                "path": agent_card.path,
                "url": str(agent_card.url),
                "num_skills": len(agent_card.skills),
                "registered_at": (
                    agent_card.registered_at.isoformat() if agent_card.registered_at else None
                ),
                "is_enabled": is_enabled,
            },
        },
    )


@router.get("/agents")
async def list_agents(
    request: Request,
    query: str | None = Query(
        None,
        description="Lexical substring search across agent name, description, tags, skill names, and metadata",
    ),
    enabled_only: bool = Query(False, description="Show only enabled agents"),
    visibility: str | None = Query(None, description="Filter by visibility"),
    allowed_groups: str | None = Query(
        None,
        alias="allowed_groups",
        description="Filter by allowed_groups (comma-separated). Returns only group-restricted agents whose allowed_groups intersect with the given values.",
    ),
    limit: int = Query(20, ge=1, le=2000, description="Number of agents to return (max 2000)"),
    offset: int = Query(0, ge=0, description="Number of agents to skip"),
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    List all agents filtered by user permissions with pagination.

    Uses lexical (substring) search, not hybrid/semantic. For vector-based
    search, use POST /api/search/semantic instead.

    Args:
        query: Lexical substring filter across name, description, tags, skill names, and metadata
        enabled_only: Only return enabled agents
        visibility: Filter by visibility level
        limit: Number of agents to return (1-500, default 20)
        offset: Number of agents to skip (default 0)
        user_context: Authenticated user context

    Returns:
        Paginated list of agent info objects with metadata
    """
    # Set audit action for agent list
    set_audit_action(request, "list", "agent", description="List all agents")

    logger.debug(
        f"list_agents called: limit={limit}, offset={offset}, "
        f"query={query!r}, enabled_only={enabled_only}, visibility={visibility}"
    )

    # Diagnostics: log at DEBUG with sensitive values redacted. Raw headers carry
    # Authorization/Cookie and the user_context can carry credential material, so
    # neither is ever emitted verbatim or at INFO.
    logger.info(f"[GET_AGENTS_ENTRY] GET /api/agents called from {get_client_ip(request)}")
    logger.debug(
        f"[GET_AGENTS_ENTRY] Request headers (redacted): {redact_headers(request.headers)}"
    )

    if user_context:
        logger.debug(f"[GET_AGENTS_DEBUG] user_context (redacted): {redact_mapping(user_context)}")
        logger.debug(f"[GET_AGENTS_DEBUG] Username: {user_context.get('username', 'NOT PRESENT')}")
        logger.debug(f"[GET_AGENTS_DEBUG] Scopes: {user_context.get('scopes', 'NOT PRESENT')}")
        logger.debug(
            f"[GET_AGENTS_DEBUG] Auth method: {user_context.get('auth_method', 'NOT PRESENT')}"
        )

    # Determine if user has unrestricted access (no agents will be filtered out)
    is_admin = user_context.get("is_admin", False) if user_context else False
    accessible_agent_list = user_context.get("accessible_agents", []) if user_context else []
    has_field_filters = bool(query or enabled_only or visibility or allowed_groups)
    # Admins skip all filtering. Non-admin users with "all" in accessible_agents
    # still need _filter_agents_by_access to enforce group-restricted visibility.
    is_unrestricted = is_admin

    # Dual-path pagination:
    # - Fast path: DB-level skip/limit for unrestricted users without field filters
    # - Fallback: full fetch + Python filter + slice for restricted users or when field filters active
    if is_unrestricted and not has_field_filters:
        # FAST PATH: DB-level pagination — correct because no agents are filtered out
        # and no field filters need a full scan for accurate total_count
        all_agents, db_total = await agent_service.get_agents_paginated(skip=offset, limit=limit)
        accessible_agents = all_agents
    else:
        # FALLBACK PATH: full fetch needed
        all_agents = await agent_service.get_all_agents()
        if is_unrestricted:
            accessible_agents = all_agents
        else:
            accessible_agents = _filter_agents_by_access(all_agents, user_context)

    # Bulk-load security-scan summaries once (path -> {scan_failed, severity counts})
    # so each card colours its shield icon from the list payload instead of fetching
    # /agents/{path}/security-scan on mount (N+1 over the page).
    from ..services.agent_scanner import agent_scanner_service

    scan_summaries = await agent_scanner_service.get_scan_summaries()

    filtered_agents = []
    search_query = query.lower() if query else ""

    for agent in accessible_agents:
        agent_is_enabled = getattr(agent, "is_enabled", False)

        if enabled_only and not agent_is_enabled:
            continue

        if visibility and agent.visibility != visibility:
            continue

        if allowed_groups:
            requested_groups = {g.strip() for g in allowed_groups.split(",") if g.strip()}
            agent_groups = set(getattr(agent, "allowed_groups", []))
            if not requested_groups.intersection(agent_groups):
                continue

        metadata_text = flatten_metadata_to_text(agent.metadata) if agent.metadata else ""
        searchable_text = (
            f"{agent.name.lower()} {agent.description.lower()} "
            f"{' '.join(agent.tags)} {' '.join([s.name for s in agent.skills])} "
            f"{metadata_text.lower()}"
        )

        if not search_query or search_query in searchable_text:
            # Extract streaming capability from agent capabilities dict
            streaming = agent.capabilities.get("streaming", False) if agent.capabilities else False

            # Extract provider organization name (provider is AgentProvider object)
            provider_name = agent.provider.organization if agent.provider else None

            agent_info = AgentInfo(
                name=agent.name,
                description=agent.description,
                path=agent.path,
                url=str(agent.url),
                tags=agent.tags,
                skills=[s.name for s in agent.skills],
                num_skills=len(agent.skills),
                num_stars=agent.num_stars,
                rating_details=agent.rating_details,
                security_scan=scan_summaries.get(agent.path),
                is_enabled=agent_is_enabled,
                provider=provider_name,
                streaming=streaming,
                trust_level=agent.trust_level,
                sync_metadata=agent.sync_metadata,
                record_kind=getattr(agent, "record_kind", None),
                ard_source_url=getattr(agent, "ard_source_url", None),
                ans_metadata=agent.ans_metadata,
                registered_by=agent.registered_by,
                status=agent.status if hasattr(agent, "status") and agent.status else "active",
                provider_organization=agent.provider.organization if agent.provider else None,
                provider_url=agent.provider.url if agent.provider else None,
                source_created_at=agent.source_created_at.isoformat()
                if agent.source_created_at
                else None,
                source_updated_at=agent.source_updated_at.isoformat()
                if agent.source_updated_at
                else None,
                registered_at=agent.registered_at.isoformat() if agent.registered_at else None,
                updated_at=agent.updated_at.isoformat() if agent.updated_at else None,
                health_status=agent.health_status or "unknown",
                last_health_check=agent.last_health_check.isoformat()
                if agent.last_health_check
                else None,
                visibility=getattr(agent, "visibility", "public"),
                allowed_groups=getattr(agent, "allowed_groups", []),
                supported_protocol=getattr(agent, "supported_protocol", None),
                metadata=agent.metadata if agent.metadata else {},
            )
            filtered_agents.append(agent_info)

    # Compute pagination metadata
    if is_unrestricted and not has_field_filters:
        # Fast path: total from DB, agents already paginated
        total_count = db_total
        page_agents = filtered_agents
    else:
        # Fallback path: slice the fully-filtered list
        total_count = len(filtered_agents)
        page_agents = filtered_agents[offset : offset + limit]

    has_next = (offset + limit) < total_count

    logger.info(
        f"User {user_context['username']} listed {len(page_agents)} agents "
        f"(total: {total_count}, offset: {offset}, limit: {limit})"
    )

    return {
        "agents": [agent.model_dump() for agent in page_agents],
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "has_next": has_next,
    }


# IMPORTANT: Specific routes with path suffixes (/health, /rate, /rating, /toggle)
# must come BEFORE catch-all {path:path} routes to prevent FastAPI from matching them incorrectly


@router.post("/agents/{path:path}/health")
async def check_agent_health(
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Perform a health check against an A2A agent.

    Per the A2A spec, there is no /ping endpoint. Agent availability is
    determined by fetching the agent card from /.well-known/agent-card.json
    on the agent's host. Falls back to the registered URL if the agent card
    endpoint is not available.
    """
    path = _normalize_path(path)

    agent_card = await agent_service.get_agent_info(path)
    if not agent_card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    accessible = _filter_agents_by_access([agent_card], user_context)
    if not accessible:
        logger.warning(
            f"User {user_context['username']} attempted to health check agent {path} without permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this agent",
        )

    if not await agent_service.is_agent_enabled(path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot perform health check on a disabled agent",
        )

    probe = await _probe_agent_backend_health(path, agent_card)
    status_label = probe["status"]

    # Persist health status to MongoDB
    try:
        await agent_service.update_agent(
            path,
            {
                "health_status": status_label,
                "last_health_check": probe["last_checked"],
            },
        )
    except Exception as e:
        logger.warning(f"Failed to persist health status for agent {path}: {e}")

    logger.info(
        f"Agent health check for {path} completed with status {status_label} "
        f"(last URL tried: {probe['health_check_url']})"
    )

    return {
        "agent_path": path,
        "health_check_url": probe["health_check_url"],
        "status": status_label,
        "status_code": probe["status_code"],
        "detail": probe["detail"],
        "response_time_ms": probe["response_time_ms"],
        "last_checked_iso": probe["last_checked_iso"],
    }


@router.post("/agents/{path:path}/rate")
async def rate_agent(
    request: Request,
    path: str,
    rating_request: RatingRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Save integer ratings to agent card."""
    # Set audit action for agent rating
    set_audit_action(
        request,
        "rate",
        "agent",
        resource_id=path,
        description=f"Rate agent with {rating_request.rating}",
    )

    path = _normalize_path(path)

    agent_card = await agent_service.get_agent_info(path)
    if not agent_card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    accessible = _filter_agents_by_access([agent_card], user_context)
    if not accessible:
        logger.warning(
            f"User {user_context['username']} attempted to rate agent {path} without permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this agent",
        )

    try:
        avg_rating = await agent_service.update_rating(
            path, user_context["username"], rating_request.rating
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error updating rating: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save rating",
        )

    return {
        "message": "Rating added successfully",
        "average_rating": avg_rating,
    }


@router.get("/agents/{path:path}/rating")
async def get_agent_rating(
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Get agent rating information."""
    path = _normalize_path(path)

    agent_card = await agent_service.get_agent_info(path)
    if not agent_card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    accessible = _filter_agents_by_access([agent_card], user_context)
    if not accessible:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this agent",
        )

    return {
        "num_stars": agent_card.num_stars,
        "rating_details": agent_card.rating_details,
    }


@router.post("/agents/{path:path}/toggle")
async def toggle_agent(
    request: Request,
    path: str,
    enabled: bool,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
):
    """
    Enable or disable an agent.

    Requires toggle_service permission for the agent.

    Args:
        path: Agent path
        enabled: New enabled state
        user_context: Authenticated user context

    Returns:
        Updated agent status

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized
    """
    # Set audit action for agent toggle
    set_audit_action(
        request, "toggle", "agent", resource_id=path, description=f"Toggle agent to {enabled}"
    )

    path = _normalize_path(path)

    agent_card = await agent_service.get_agent_info(path)
    if not agent_card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    _check_agent_permission("toggle", agent_card.name, user_context)

    # Per-resource access check for non-admins, mirroring the server toggle
    # (POST /api/servers/toggle). Having toggle_service permission is not
    # enough; the caller must also have access to this specific agent (in
    # accessible_agents, or be its owner).
    if not user_context.get("is_admin", False):
        accessible_agents = user_context.get("accessible_agents", [])
        owns_agent = agent_card.registered_by == user_context.get("username")
        if "all" not in accessible_agents and path not in accessible_agents and not owns_agent:
            logger.warning(
                f"User {user_context.get('username')} attempted to toggle agent "
                f"{path} without access"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this agent",
            )

    success = await agent_service.toggle_agent(path, enabled)

    if not success:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Failed to toggle agent state"},
        )

    # On enable, probe the backend and persist health so the nginx reverse-proxy
    # block is actually generated: a freshly enabled agent defaults to "unknown",
    # which the generator treats as not-healthy and skips. Re-mark nginx dirty so
    # the (debounced) reload regenerates with the refreshed status.
    if enabled:
        await _refresh_agent_health(path)
        from ..core.nginx_service import nginx_reload_scheduler

        nginx_reload_scheduler.mark_dirty()

    try:
        search_repo = get_search_repository()
        await search_repo.index_agent(path, agent_card, enabled)
    except Exception as e:
        logger.warning(f"Failed to update search index for {path}: {e}")

    logger.info(
        f"Agent '{agent_card.name}' ({path}) toggled to {enabled} by user "
        f"'{user_context['username']}'"
    )

    return {
        "message": f"Agent {'enabled' if enabled else 'disabled'} successfully",
        "path": path,
        "is_enabled": enabled,
    }


@router.get("/agents/{path:path}/security-scan")
async def get_agent_security_scan(
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """
    Get security scan results for an A2A agent.

    Returns the latest security scan results for the specified agent,
    including threat analysis, severity levels, and detailed findings
    from YARA, specification validation, and heuristic analyzers.

    **Authentication:** JWT Bearer token or session cookie
    **Authorization:** Requires admin privileges or access to the agent

    **Path Parameters:**
    - `path` (required): Agent path (e.g., /code-reviewer)

    **Response:**
    Returns security scan results with analysis_results and findings.

    **Example:**
    ```bash
    curl -X GET http://localhost/api/agents/code-reviewer/security-scan \\
      --cookie-jar .cookies --cookie .cookies
    ```
    """
    if not path.startswith("/"):
        path = "/" + path

    # Check if agent exists
    agent_info = await agent_service.get_agent_info(path)
    if not agent_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    # Authorization: scan results expose security findings for the agent, so
    # restrict to users who can see the agent (admins always can). Without this
    # a non-admin could read scan results for a private/group agent. Reuse the
    # agent_info we already fetched via the *_from_doc helper so the visibility
    # check does not trigger a second identical agent lookup.
    if not user_context["is_admin"]:
        from ..services.visibility import user_can_access_agent_from_doc

        if not user_can_access_agent_from_doc(agent_info.model_dump(), user_context):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this agent",
            )

    # Get scan results
    from ..services.agent_scanner import agent_scanner_service

    scan_result = await agent_scanner_service.get_scan_result(path)
    if not scan_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No security scan results found for agent '{path}'. "
            "The agent may not have been scanned yet.",
        )

    return scan_result


@router.post("/agents/{path:path}/rescan")
async def rescan_agent(
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """
    Trigger a manual security scan for an A2A agent.

    Initiates a new security scan for the specified agent and returns
    the results. This endpoint is useful for re-scanning agents after
    updates or for on-demand security assessments.

    **Authentication:** JWT Bearer token or session cookie
    **Authorization:** Requires admin privileges

    **Path Parameters:**
    - `path` (required): Agent path (e.g., /code-reviewer)

    **Response:**
    Returns the newly generated security scan results.

    **Example:**
    ```bash
    curl -X POST http://localhost/api/agents/code-reviewer/rescan \\
      --cookie-jar .cookies --cookie .cookies
    ```
    """
    # Only admins can trigger manual scans
    if not user_context["is_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can trigger security scans",
        )

    if not path.startswith("/"):
        path = "/" + path

    # Check if agent exists
    agent_info = await agent_service.get_agent_info(path)
    if not agent_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    # Get agent card from agent info
    agent_card_dict = agent_info.model_dump()

    logger.info(
        f"Manual security scan requested by user '{user_context.get('username')}' "
        f"for agent '{path}'"
    )

    try:
        # Trigger security scan
        from ..services.agent_scanner import agent_scanner_service

        scan_result = await agent_scanner_service.scan_agent(
            agent_card=agent_card_dict,
            agent_path=path,
            analyzers=None,  # Use default analyzers from config
            api_key=None,  # Use default API key from config
            timeout=None,  # Use default timeout from config
        )

        # Return the full scan result including raw_output for detailed findings
        return scan_result.model_dump(mode="json")

    except Exception as e:
        logger.exception(f"Manual security scan failed for agent '{path}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security scan failed",
        )


@router.post(
    "/agents/batch",
    status_code=202,
    responses={
        202: {
            "description": "Batch accepted (or idempotent replay).",
            "headers": {
                "X-Idempotent-Replay": {
                    "description": (
                        "Present and equal to 'true' when the returned job_id comes "
                        "from a prior submission with the same idempotency_key. The "
                        "new request body was not run."
                    ),
                    "schema": {"type": "string"},
                },
            },
        },
        413: {"description": "Request body or item count exceeds configured limits."},
        429: {"description": "Submitter has too many concurrent batch jobs."},
    },
)
async def submit_agent_batch(
    http_request: Request,
    body: AgentBatchRequest,
    response: Response,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Submit an asynchronous batch of agent register/patch/replace/delete ops.

    Each item is re-authorized and processed independently by the batch worker;
    one failing item never aborts the job. Poll GET /api/agents/batch/{job_id}
    for progress and per-item results.

    Returns:
        202 with {job_id, status_url}.

    Raises:
        HTTPException: 413 if the body or item count is too large, 429 if the
            submitter already has the maximum number of active jobs.
    """
    # Defence in depth: reject oversize payloads before deep processing. nginx
    # sits in front in production but local/dev deployments may not cap size.
    content_length = int(http_request.headers.get("content-length") or 0)
    if content_length and content_length > settings.batch_max_request_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"request body exceeds {settings.batch_max_request_bytes} bytes",
        )

    if len(body.items) > settings.batch_max_operations_per_job:
        raise HTTPException(
            status_code=413,
            detail=f"batch exceeds max {settings.batch_max_operations_per_job} items",
        )

    try:
        job, replayed = await agent_batch_service.submit(
            body,
            submitted_by=user_context["username"],
            submitted_body_hash=_hash_items(body.items),
            submitter_is_admin=user_context.get("is_admin", False),
            submitter_ui_permissions=user_context.get("ui_permissions", {}),
            request_id=http_request.headers.get("x-request-id"),
        )
    except ConcurrentJobLimitError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e

    if replayed:
        response.headers["X-Idempotent-Replay"] = "true"

    return {"job_id": job.job_id, "status_url": f"/api/agents/batch/{job.job_id}"}


@router.get("/agents/batch/{job_id}")
async def get_agent_batch_job(
    job_id: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Fetch the current state and per-item results of a batch job.

    Caller must be the submitter or an admin.

    Raises:
        HTTPException: 403 if not the submitter/admin, 404 if unknown job_id.
    """
    job = await agent_batch_service.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    if not user_context.get("is_admin", False) and job.submitted_by != user_context["username"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view batch jobs you submitted",
        )
    return job.model_dump(
        mode="json",
        exclude={"submitter_ui_permissions", "submitter_is_admin", "submitted_body_hash"},
    )


@router.post("/agents/{path:path}/pull-card", response_model=PullCardResponse)
async def pull_agent_card(
    request: Request,
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    dry_run: bool = Query(True, description="Preview changes without applying"),
):
    """Pull the latest A2A agent card from the remote endpoint.

    Fetches /.well-known/agent-card.json from the agent's host and compares
    it with the local record. In dry-run mode (default), returns the diff
    without applying changes. In overwrite mode, applies A2A-spec fields
    while preserving registry-specific metadata.

    Note: a successful remote fetch always refreshes `health_status` and
    `last_health_check` on the local record regardless of `dry_run`, since
    the fetch itself is the health signal. Other than that side effect,
    dry-run mode performs no writes.

    CSRF: not enforced here, matching the agent PUT/PATCH/DELETE endpoints,
    which also rely on bearer-token auth rather than verify_csrf_token_flexible.

    Args:
        path: Agent path
        dry_run: If true, preview only (apart from the health-fields refresh
            documented above). If false, apply A2A-spec changes alongside the
            health refresh in a single write.
        user_context: Authenticated user context

    Returns:
        PullCardResponse with diff and optionally updated agent

    Raises:
        HTTPException: 400/403/404/502 depending on condition
    """
    set_audit_action(
        request,
        "pull_card" if not dry_run else "pull_card_preview",
        "agent",
        resource_id=path,
        description=f"Pull agent card {'(dry-run)' if dry_run else '(apply)'}",
    )

    path = _normalize_path(path)

    # 1. Check agent exists
    existing_agent = await agent_service.get_agent_info(path)
    if not existing_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    # 2. Check permissions (modify_service + owner or admin)
    _check_agent_permission("modify", existing_agent.name, user_context)

    if not user_context["is_admin"] and existing_agent.registered_by != user_context["username"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only pull card updates for agents you registered",
        )

    # 3. Block federated/read-only agents
    sync_metadata = existing_agent.sync_metadata or {}
    if sync_metadata.get("is_federated") or sync_metadata.get("is_read_only"):
        source_peer = sync_metadata.get("source_peer_id", "unknown peer registry")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Agent '{path}' is synced from {source_peer} and cannot be updated locally.",
        )

    # 4. Check agent has a valid URL and is A2A protocol
    if not existing_agent.url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent has no registered URL to fetch card from",
        )

    if existing_agent.supported_protocol and existing_agent.supported_protocol != "a2a":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pull card is only supported for A2A protocol agents",
        )

    # 5. Fetch remote card
    base_url = str(existing_agent.url).rstrip("/")
    remote_card_raw, remote_card_url = await _fetch_remote_agent_card(base_url)

    # 6. Normalize camelCase keys to snake_case
    remote_card = _normalize_remote_card_keys(remote_card_raw)

    # 7. Compute diff (A2A-spec fields only)
    changes = _compute_card_diff(existing_agent, remote_card)
    has_changes = len(changes) > 0

    # S3: a change to the agent's URL could indicate a redirect/takeover, so log
    #     it explicitly even though the operator also sees it in the dry-run diff.
    for change in changes:
        if change.field == "url":
            logger.warning(
                f"pull-card: agent {path} URL would change from "
                f"'{change.current_value}' to '{change.remote_value}' "
                f"(requested by '{user_context['username']}')"
            )
            # Fail closed on a remote card that tries to point the stored agent
            # URL at an internal/metadata target. Reject in BOTH preview and
            # apply modes so a poisoned card is surfaced during dry-run and can
            # never be persisted. update_agent re-validates on the write path;
            # this earlier check also covers the preview, which does not persist
            # the URL and therefore would not otherwise validate it.
            try:
                validate_agent_url(str(change.remote_value))
            except UrlValidationError as exc:
                logger.warning(
                    "pull-card: rejecting URL change for agent %s -> %s: %s",
                    path,
                    change.remote_value,
                    exc,
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "Remote agent card URL failed SSRF validation and was "
                        "rejected (private/internal addresses are not allowed)"
                    ),
                ) from exc

    # R4: single structured log line per pull-card op so adoption/outcomes can be
    #     scraped without a dedicated metric. The audit trail also records this.
    logger.info(
        "pull_card op=%s path=%s user=%s has_changes=%s change_count=%d",
        "preview" if dry_run else "apply",
        path,
        user_context["username"],
        has_changes,
        len(changes),
    )

    # 8. Build the update dict. A successful fetch means the agent is healthy,
    #    so health fields are always part of the write. When applying, the safe
    #    A2A-field updates (S5) are merged into the SAME write (P1) so there is a
    #    single DB write + single re-index per request instead of two.
    health_now = datetime.now(UTC)
    updates: dict[str, Any] = {
        "health_status": "healthy",
        "last_health_check": health_now,
    }

    applied = False
    if not dry_run and has_changes:
        updates.update(_build_safe_card_updates(changes))

    # 9. Persist. In dry-run mode this is just the health side-effect; in apply
    #    mode it is health + A2A fields in one call.
    try:
        updated_agent = await agent_service.update_agent(path, updates)
    except UrlValidationError as e:
        # A poisoned URL that reached the write path is a client-side rejection,
        # not a server error: surface it as 400 and never persist it.
        logger.warning(f"pull-card: URL rejected on write for agent {path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Remote agent card URL failed SSRF validation and was rejected "
                "(private/internal addresses are not allowed)"
            ),
        ) from e
    except Exception as e:
        # In dry-run mode a failed health write should not fail the preview.
        if dry_run or not has_changes:
            logger.warning(f"Failed to update health status for agent {path}: {e}")
            updated_agent = existing_agent
        else:
            logger.error(f"Failed to apply card changes to agent {path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to apply card changes: {e}",
            )
    else:
        if not dry_run and has_changes:
            is_enabled = await agent_service.is_agent_enabled(path)
            try:
                search_repo = get_search_repository()
                await search_repo.index_agent(path, updated_agent, is_enabled)
            except Exception as e:
                logger.warning(f"Failed to update search index for {path}: {e}")
            applied = True
            logger.info(
                f"Applied {len(changes)} A2A card changes to agent {path} "
                f"by user '{user_context['username']}'"
            )

    return PullCardResponse(
        agent_path=path,
        dry_run=dry_run,
        remote_card_url=remote_card_url,
        changes=changes,
        has_changes=has_changes,
        applied=applied,
        health_status="healthy",
        remote_card=remote_card_raw,
    )


@router.get("/agents/{path:path}")
async def get_agent(
    request: Request,
    path: str,
    response: Response,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """
    Get a single agent by path.

    Public agents are visible without special permissions.
    Private and group-restricted agents require authorization.

    Args:
        request: HTTP request object
        path: Agent path
        user_context: Authenticated user context

    Returns:
        Complete agent card

    Raises:
        HTTPException: 404 if not found, 403 if not authorized
    """
    path = _normalize_path(path)

    # Set audit action for agent read
    set_audit_action(request, "read", "agent", resource_id=path, description=f"Read agent {path}")

    agent_card = await agent_service.get_agent_info(path)
    if not agent_card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    accessible = _filter_agents_by_access([agent_card], user_context)

    if not accessible:
        logger.warning(
            f"User {user_context['username']} attempted to access agent {path} without permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this agent",
        )

    response.headers["ETag"] = _weak_etag_for(agent_card)
    agent_dict = agent_card.model_dump()
    # Hide the internal backend (proxy_pass_url) from non-admins in with-gateway
    # mode; admins and registry-only mode see it. Mirrors MCP-server redaction.
    from ..services.visibility import (
        redact_agent_backend_fields,
        should_redact_backend_urls,
    )

    if should_redact_backend_urls(user_context):
        redact_agent_backend_fields(agent_dict)
    return agent_dict


@router.put("/agents/{path:path}")
async def update_agent(
    http_request: Request,
    path: str,
    request: AgentRegistrationRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """
    Update an existing agent card.

    Requires modify_service permission for the agent.
    User must be agent owner or admin.

    Args:
        path: Agent path
        request: Updated agent data
        user_context: Authenticated user context

    Returns:
        Updated agent card

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized
    """
    # Set audit action for agent update
    set_audit_action(
        http_request,
        "update",
        "agent",
        resource_id=path,
        description=f"Update agent {request.name}",
    )

    path = _normalize_path(path)

    existing_agent = await agent_service.get_agent_info(path)
    if not existing_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    _check_agent_permission("modify", existing_agent.name, user_context)

    if not user_context["is_admin"] and existing_agent.registered_by != user_context["username"]:
        logger.warning(
            f"User {user_context['username']} attempted to update agent {path} "
            f"owned by {existing_agent.registered_by}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update agents you registered",
        )

    tag_list = _normalize_tag_list(request.tags)

    try:
        # Build optional kwargs for fields that have defaults on AgentCard
        update_optional_kwargs: dict[str, Any] = {}
        if request.default_input_modes:
            update_optional_kwargs["default_input_modes"] = request.default_input_modes
        if request.default_output_modes:
            update_optional_kwargs["default_output_modes"] = request.default_output_modes

        updated_agent = AgentCard(
            protocol_version=request.protocol_version,
            name=request.name,
            description=request.description,
            url=request.url,
            path=path,
            version=request.version,
            provider=request.provider,
            security_schemes=request.security_schemes or {},
            skills=request.skills or [],
            tags=tag_list,
            license=request.license,
            visibility=request.visibility,
            allowed_groups=request.allowed_groups,
            trust_level=request.trust_level,
            supported_protocol=request.supported_protocol,
            registered_by=existing_agent.registered_by,
            registered_at=existing_agent.registered_at,
            is_enabled=existing_agent.is_enabled,
            num_stars=existing_agent.num_stars,
            metadata=request.metadata if request.metadata else existing_agent.metadata,
            capabilities=request.capabilities
            if request.capabilities
            else existing_agent.capabilities,
            ans_metadata=existing_agent.ans_metadata,
            health_status=existing_agent.health_status,
            last_health_check=existing_agent.last_health_check,
            rating_details=existing_agent.rating_details,
            sync_metadata=existing_agent.sync_metadata,
            status=request.status if request.status else existing_agent.status,
            **update_optional_kwargs,
        )

        # Lifecycle status change requires change_lifecycle_status (Issue #1330).
        _check_agent_lifecycle_status_permission(existing_agent, updated_agent, user_context)

        from ..utils.agent_validator import agent_validator

        validation_result = await agent_validator.validate_agent_card(
            updated_agent,
            verify_endpoint=False,
        )

        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Agent card validation failed",
                    "errors": validation_result.errors,
                },
            )

        # Re-apply the reverse-proxy url/proxy_pass_url split (validation above
        # probed the real backend); without this a PUT that changes url would
        # desync the advertised gateway url from the stored backend.
        _apply_a2a_reverse_proxy_split(updated_agent, path)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid agent card: {str(e)}",
        )

    # Registration gate check for update (admission control, issue #809)
    gate_result = await check_registration_gate(
        asset_type="agent",
        operation="update",
        source_api=f"/api/agents/{path}",
        registration_payload=request.model_dump(mode="json"),
        raw_headers=http_request.scope.get("headers", []),
    )
    if not gate_result.allowed:
        logger.warning(
            f"Registration gate denied agent update '{request.name}': {gate_result.error_message}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Registration denied by policy gate: {gate_result.error_message}",
        )

    # update_agent takes a dict of fields to merge (it calls updates.get(...)),
    # so pass the model dump, not the AgentCard instance.
    success = await agent_service.update_agent(path, updated_agent.model_dump())

    if not success:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Failed to save updated agent data"},
        )

    is_enabled = await agent_service.is_agent_enabled(path)
    try:
        search_repo = get_search_repository()
        await search_repo.index_agent(path, updated_agent, is_enabled)
    except Exception as e:
        logger.warning(f"Failed to update search index for {path}: {e}")

    logger.info(
        f"Agent '{updated_agent.name}' ({path}) updated by user '{user_context['username']}'"
    )

    return updated_agent.model_dump()


@router.patch("/agents/{path:path}")
async def patch_agent(
    http_request: Request,
    path: str,
    patch_body: AgentCardPatch,
    response: Response,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
):
    """Apply an RFC 7396 JSON Merge Patch to an agent card.

    Only the fields explicitly supplied are changed; required fields stay
    required. Registrant-only fields are rejected by the AgentCardPatch
    validator before this handler runs.

    Concurrency: if `If-Match` is supplied the request is rejected with 412
    when the stored updated_at no longer matches. If absent, PATCH is
    last-write-wins (same race window as PUT); clients that care must send
    If-Match.

    Returns:
        200 with the full updated card and a fresh ETag header.

    Raises:
        HTTPException: 400 empty/malformed, 403 unauthorized/federated,
            404 not found, 412 precondition failed, 422 validation.
    """
    set_audit_action(
        http_request,
        "update",
        "agent",
        resource_id=path,
        description=f"Patch agent {path}",
    )

    path = _normalize_path(path)

    existing_agent = await agent_service.get_agent_info(path)
    if not existing_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    # Federated read-only guard (parity with DELETE)
    sync_metadata = existing_agent.sync_metadata or {}
    if sync_metadata.get("is_federated") or sync_metadata.get("is_read_only"):
        source_peer = sync_metadata.get("source_peer_id", "unknown peer registry")
        logger.warning(
            f"User {user_context['username']} attempted to patch federated agent {path} "
            f"from {source_peer}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Agent '{path}' is synced from {source_peer} and cannot be patched locally. "
            f"Patch this agent at its source registry, or remove the peer federation.",
        )

    # Authorization (parity with PUT)
    _check_agent_permission("modify", existing_agent.name, user_context)
    if not user_context["is_admin"] and existing_agent.registered_by != user_context["username"]:
        logger.warning(
            f"User {user_context['username']} attempted to patch agent {path} "
            f"owned by {existing_agent.registered_by}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only patch agents you registered",
        )

    # Optimistic concurrency
    client_ts = _parse_if_match(if_match)
    if client_ts is not None:
        server_ts = _agent_updated_ms(existing_agent)
        if client_ts != server_ts:
            logger.warning(
                "patch_agent if_match_mismatch path=%s user=%s client_ts=%d server_ts=%d",
                path,
                user_context["username"],
                client_ts,
                server_ts,
            )
            raise HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail="If-Match does not match current agent version",
            )

    # Merge supplied fields onto the existing card
    patch_dict = patch_body.model_dump(exclude_unset=True, by_alias=False)
    if not patch_dict:
        raise HTTPException(status_code=400, detail="Empty patch body")

    merged_dict = {**existing_agent.model_dump(), **patch_dict}
    try:
        merged_agent = AgentCard(**merged_dict)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors(),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid agent card: {str(e)}",
        ) from e

    # Defence in depth: re-pin server-managed fields from the existing card.
    for field in REGISTRANT_ONLY_FIELDS:
        setattr(merged_agent, field, getattr(existing_agent, field))

    # Lifecycle status change requires change_lifecycle_status (Issue #1330).
    _check_agent_lifecycle_status_permission(existing_agent, merged_agent, user_context)

    from ..utils.agent_validator import agent_validator

    validation_result = await agent_validator.validate_agent_card(
        merged_agent,
        verify_endpoint=False,
    )
    if not validation_result.is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Agent card validation failed",
                "errors": validation_result.errors,
            },
        )

    # Re-apply the reverse-proxy url/proxy_pass_url split so a PATCH of url keeps
    # the advertised gateway url in sync with the stored backend (idempotent when
    # url already points at the gateway).
    _apply_a2a_reverse_proxy_split(merged_agent, path)

    # Registration gate (parity with PUT)
    gate_result = await check_registration_gate(
        asset_type="agent",
        operation="update",
        source_api=f"/api/agents/{path}",
        registration_payload=merged_agent.model_dump(mode="json"),
        raw_headers=http_request.scope.get("headers", []),
    )
    if not gate_result.allowed:
        logger.warning(
            f"Registration gate denied agent patch '{merged_agent.name}': "
            f"{gate_result.error_message}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Registration denied by policy gate: {gate_result.error_message}",
        )

    # Persist (update_agent sets updated_at and re-indexes search internally)
    await agent_service.update_agent(path, merged_agent.model_dump())

    updated = await agent_service.get_agent_info(path)

    asyncio.create_task(
        send_registration_webhook(
            event_type="update",
            registration_type="agent",
            card_data=updated.model_dump(mode="json"),
            performed_by=user_context.get("username"),
        )
    )

    logger.info(
        "patch_agent success path=%s user=%s if_match_supplied=%s",
        path,
        user_context["username"],
        if_match is not None,
    )

    response.headers["ETag"] = _weak_etag_for(updated)
    return updated.model_dump()


@router.delete("/agents/{path:path}")
async def delete_agent(
    request: Request,
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """
    Delete an agent from the registry.

    Requires the delete_agent UI permission AND (admin OR agent ownership) --
    uniform with server/skill/custom-entity delete. Admins bypass both checks.

    Args:
        path: Agent path
        user_context: Authenticated user context

    Returns:
        204 No Content

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized
    """
    # Set audit action for agent deletion
    set_audit_action(
        request, "delete", "agent", resource_id=path, description=f"Delete agent at {path}"
    )

    path = _normalize_path(path)

    existing_agent = await agent_service.get_agent_info(path)
    if not existing_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    # Block deletion of federated (read-only) agents from peer registries
    sync_metadata = existing_agent.sync_metadata or {}
    if sync_metadata.get("is_federated") or sync_metadata.get("is_read_only"):
        source_peer = sync_metadata.get("source_peer_id", "unknown peer registry")
        logger.warning(
            f"User {user_context['username']} attempted to delete federated agent {path} "
            f"from {source_peer}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Agent '{path}' is synced from {source_peer} and cannot be deleted locally. "
            f"Delete this agent from its source registry, or remove the peer federation.",
        )

    # Strict dual gate, uniform with server/skill/custom-entity delete: the caller
    # must hold the delete_agent scope AND be an admin or the agent's owner.
    # Previously ownership alone sufficed (scope OR owner), letting an owner delete
    # without the delete_agent grant -- the lone outlier among the asset families.
    # The scope half routes through the canonical helper keyed on the agent NAME
    # (matching modify/toggle and the batch path); the ownership check below
    # (skipped for admins) supplies the AND-owner half.
    _check_agent_permission("delete", existing_agent.name, user_context)

    if (
        not user_context.get("is_admin", False)
        and existing_agent.registered_by != user_context["username"]
    ):
        logger.warning(
            f"User {user_context['username']} attempted to delete agent {path} "
            f"owned by {existing_agent.registered_by}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete agents you registered",
        )

    success = await agent_service.remove_agent(path)

    if not success:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Failed to delete agent"},
        )

    try:
        search_repo = get_search_repository()
        await search_repo.remove_entity(path)
    except Exception as e:
        logger.warning(f"Failed to remove search index for {path}: {e}")

    logger.info(f"Agent at path '{path}' deleted by user '{user_context['username']}'")

    asyncio.create_task(
        send_registration_webhook(
            event_type="deletion",
            registration_type="agent",
            card_data=existing_agent.model_dump(mode="json"),
            performed_by=user_context.get("username"),
        )
    )

    return JSONResponse(
        status_code=status.HTTP_204_NO_CONTENT,
        content=None,
    )


@router.post("/agents/discover")
async def discover_agents_by_skills(
    skills: list[str],
    tags: list[str] | None = None,
    max_results: int = Query(10, ge=1, le=100),
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Discover agents by required skills.

    Returns agents that have the specified skills, ranked by relevance.

    Args:
        skills: Required skill names or IDs
        tags: Optional tag filters
        max_results: Maximum number of results
        user_context: Authenticated user context

    Returns:
        List of matching agents with relevance scores

    Raises:
        HTTPException: 400 if no skills provided
    """
    if not skills:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one skill must be specified",
        )

    logger.info(f"User {user_context['username']} discovering agents with skills: {skills}")

    all_agents = await agent_service.get_all_agents()
    accessible_agents = _filter_agents_by_access(all_agents, user_context)

    matched_agents = []
    required_skills = set(s.lower() for s in skills)
    required_tags = set(t.lower() for t in tags) if tags else set()

    for agent in accessible_agents:
        if not getattr(agent, "is_enabled", False):
            continue

        agent_skills = set(skill.id.lower() for skill in agent.skills) | set(
            skill.name.lower() for skill in agent.skills
        )

        skill_matches = required_skills & agent_skills
        if not skill_matches:
            continue

        agent_tags = set(t.lower() for t in agent.tags)
        tag_matches = required_tags & agent_tags if required_tags else set()

        skill_match_score = len(skill_matches) / len(required_skills)
        tag_match_score = len(tag_matches) / len(required_tags) if required_tags else 0.0

        trust_boost = {
            "unverified": 0.0,
            "community": 0.2,
            "verified": 0.5,
            "trusted": 1.0,
        }.get(agent.trust_level, 0.0)

        relevance_score = 0.6 * skill_match_score + 0.2 * tag_match_score + 0.2 * trust_boost

        # Extract streaming capability -- check capabilities dict first, fall back to
        # top-level field for agents registered before the capabilities dict change
        streaming = (
            agent.capabilities.get("streaming", False)
            if agent.capabilities
            else getattr(agent, "streaming", False)
        )

        # Extract provider organization name (provider is AgentProvider object)
        provider_name = agent.provider.organization if agent.provider else None

        agent_info = AgentInfo(
            name=agent.name,
            description=agent.description,
            path=agent.path,
            url=str(agent.url),
            tags=agent.tags,
            skills=[s.name for s in agent.skills],
            num_skills=len(agent.skills),
            num_stars=agent.num_stars,
            is_enabled=True,
            provider=provider_name,
            streaming=streaming,
            trust_level=agent.trust_level,
            visibility=getattr(agent, "visibility", "public"),
            supported_protocol=getattr(agent, "supported_protocol", None),
            metadata=agent.metadata if agent.metadata else {},
        )

        matched_agents.append(
            {
                **agent_info.model_dump(),
                "relevance_score": round(relevance_score, 2),
                "matched_skills": list(skill_matches),
            }
        )

    matched_agents.sort(key=lambda x: x["relevance_score"], reverse=True)
    matched_agents = matched_agents[:max_results]

    logger.info(f"Found {len(matched_agents)} agents matching skills: {skills}")

    return {
        "agents": matched_agents,
        "query": {
            "skills": skills,
            "tags": tags,
        },
    }


@router.post("/agents/discover/semantic")
async def discover_agents_semantic(
    query: str,
    max_results: int = Query(10, ge=1, le=100),
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    search_repo: SearchRepositoryBase = Depends(get_search_repo),
):
    """
    Discover agents using natural language semantic search.

    Uses search repository (DocumentDB) to find agents matching the query intent.

    Args:
        query: Natural language query describing needed capabilities
        max_results: Maximum number of results
        user_context: Authenticated user context
        search_repo: Search repository dependency

    Returns:
        List of matching agents with relevance scores

    Raises:
        HTTPException: 400 if query is empty
    """
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    logger.info(f"User {user_context['username']} semantic search for agents: {query}")

    from ..services.visibility import (
        redact_agent_backend_fields,
        should_redact_backend_urls,
    )

    try:
        search_results = await search_repo.search(
            query=query,
            entity_types=["a2a_agent"],
            max_results=max_results,
        )

        # Extract agents from search results
        results = search_results.get("agents", [])

        all_agents = await agent_service.get_all_agents()
        agent_map = {agent.path: agent for agent in all_agents}

        # Non-admins get the gateway-facing url only; the internal backend
        # (proxy_pass_url) is stripped so discovery never leaks it. Mirrors the
        # MCP-server semantic-search redaction.
        redact_backend = should_redact_backend_urls(user_context)

        accessible_results = []
        for result in results:
            agent_card = agent_map.get(result.get("path"))
            if not agent_card:
                continue

            if not _filter_agents_by_access([agent_card], user_context):
                continue

            # Return full agent card with relevance score
            agent_data = agent_card.model_dump()
            if redact_backend:
                redact_agent_backend_fields(agent_data)
            agent_data["relevance_score"] = result.get("relevance_score", 0.0)

            accessible_results.append(agent_data)

        logger.info(f"Semantic search returned {len(accessible_results)} agents for query: {query}")

        # Increment semantic search counter (fail-silent)
        from ..repositories.stats_repository import increment_search_counter

        await increment_search_counter()

        return {
            "agents": accessible_results,
            "query": query,
        }

    except Exception as e:
        logger.error(f"Error in semantic agent search: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Semantic search failed",
        )
