"""One FastMCP hub per OpenAPI MCP server, rebuilt and swapped on every change.

FastMCP has no unmount call, so the container never edits a running hub. Each
push carries a server's full state; ``apply_server_state`` builds a new hub from
it, swaps it in, and then closes the old one. A push whose generation is not
newer than the running hub's is ignored, so out-of-order or repeated pushes
from several registry replicas are harmless.
"""

import asyncio
import logging
import time
from dataclasses import (
    dataclass,
    field,
)
from typing import Any

import httpx2
from conversion import (
    ConversionError,
    convert_target,
)
from fastmcp import FastMCP
from guarded_transport import (
    GuardedHttpx2Transport,
    PerUserAuth,
)
from models import (
    HubServerState,
    HubTargetState,
    ServerStatus,
    TargetStatus,
)
from settings import ContainerSettings

logger = logging.getLogger(__name__)


MCP_PATH: str = "/mcp"
# How long a replaced hub keeps serving requests that started before the swap.
# Closing it at once cut off in-flight calls, including the registry's security
# scan that every change starts, and a cut-off scan disables the server.
RETIRED_HUB_GRACE_SECONDS: float = 30.0


@dataclass
class HubEntry:
    """A running hub and everything that has to be closed with it."""

    generation: int
    app: Any = None
    lifespan_task: asyncio.Task | None = None
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    clients: list[httpx2.AsyncClient] = field(default_factory=list)
    status: dict[str, TargetStatus] = field(default_factory=dict)


_hubs: dict[str, HubEntry] = {}
_locks: dict[str, asyncio.Lock] = {}
_retiring: dict[asyncio.Task, HubEntry] = {}


def _lock_for(
    slug: str,
) -> asyncio.Lock:
    lock = _locks.get(slug)
    if lock is None:
        lock = asyncio.Lock()
        _locks[slug] = lock
    return lock


def _auth_client_kwargs(
    target: HubTargetState,
) -> dict[str, Any]:
    """Client headers, params or auth for the target's one auth scheme."""
    if target.auth_credential_mode == "per_user":
        return {}
    credential = target.auth_credential.get_secret_value() if target.auth_credential else ""
    if target.auth_scheme == "api_key_header" and target.auth_header_name:
        return {"headers": {target.auth_header_name: credential}}
    if target.auth_scheme == "api_key_query" and target.auth_query_param:
        return {"params": {target.auth_query_param: credential}}
    if target.auth_scheme == "bearer":
        return {"headers": {"Authorization": f"Bearer {credential}"}}
    if target.auth_scheme == "basic":
        return {"auth": httpx2.BasicAuth(target.auth_username or "", credential)}
    return {}


def _backend_client(
    target: HubTargetState,
    settings: ContainerSettings,
) -> httpx2.AsyncClient:
    """An httpx2 client for one target, routed through the SSRF guard."""
    per_user = None
    if target.auth_credential_mode == "per_user":
        per_user = PerUserAuth(
            target.name, target.auth_scheme, target.auth_header_name, target.base_url
        )
    return httpx2.AsyncClient(
        base_url=target.base_url,
        timeout=settings.openapi_mcp_backend_timeout_seconds,
        transport=GuardedHttpx2Transport(
            max_response_bytes=settings.openapi_mcp_max_response_bytes, per_user=per_user
        ),
        follow_redirects=False,
        **_auth_client_kwargs(target),
    )


async def _mount_target(
    hub: FastMCP,
    target: HubTargetState,
    settings: ContainerSettings,
    entry: HubEntry,
) -> None:
    """Convert one target and mount it. A failure marks only this target as error."""
    client = _backend_client(target, settings)
    try:
        sub, names = await convert_target(
            target.spec, target, client, settings.openapi_mcp_conversion_timeout_seconds
        )
    except ConversionError as e:
        await client.aclose()
        logger.warning(f"Target '{target.name}' failed to convert: {e}")
        entry.status[target.name] = TargetStatus(status="error", error=str(e))
        return
    hub.mount(sub, namespace=target.name)
    entry.clients.append(client)
    entry.status[target.name] = TargetStatus(
        status="ok", tools=[f"{target.name}_{name}" for name in names]
    )


async def _run_lifespan(
    app: Any,
    ready: asyncio.Event,
    stop: asyncio.Event,
    errors: list[BaseException],
) -> None:
    """Hold a hub's session manager open until told to stop.

    The lifespan is entered and exited in this one task. FastMCP sets context
    variables on entry, and resetting them from another task (the next push, or
    shutdown) raises.
    """
    try:
        async with app.router.lifespan_context(app):
            ready.set()
            await stop.wait()
    except Exception as e:
        errors.append(e)
        logger.exception("A hub's session manager failed")
    finally:
        ready.set()


async def _start_lifespan(
    entry: HubEntry,
) -> None:
    """Start the hub's session manager in its own task and wait until it is ready."""
    ready = asyncio.Event()
    errors: list[BaseException] = []
    entry.lifespan_task = asyncio.create_task(_run_lifespan(entry.app, ready, entry.stop, errors))
    await ready.wait()
    if errors:
        raise RuntimeError(f"hub session manager failed to start: {type(errors[0]).__name__}")


async def _build_hub(
    state: HubServerState,
    settings: ContainerSettings,
) -> HubEntry:
    """Build a hub for every target in the state and start its session manager."""
    start_time = time.time()
    hub = FastMCP(
        name=state.name,
        instructions=state.description or None,
        version=settings.build_version,
        strict_input_validation=True,
    )
    entry = HubEntry(generation=state.generation)
    for target in state.targets:
        await _mount_target(hub, target, settings, entry)

    # Starting each hub's session manager when the hub is built lets several
    # hubs created at runtime share one ASGI server (prototype finding 3).
    entry.app = hub.http_app(path=MCP_PATH, stateless_http=True)
    await _start_lifespan(entry)
    elapsed = time.time() - start_time
    summary = {name: s.status for name, s in entry.status.items()}
    logger.info(
        f"Hub '{state.slug}' generation {state.generation} built in {elapsed:.2f}s: {summary}"
    )
    return entry


async def _close_entry(
    entry: HubEntry,
) -> None:
    """Stop the hub's session manager and close its backend clients."""
    entry.stop.set()
    if entry.lifespan_task is not None:
        await entry.lifespan_task
    for client in entry.clients:
        await client.aclose()


def _retire_later(
    entry: HubEntry,
) -> None:
    """Close a replaced hub after the grace period, so in-flight requests finish."""

    async def _close_after_grace() -> None:
        await asyncio.sleep(RETIRED_HUB_GRACE_SECONDS)
        await _close_entry(entry)

    task = asyncio.create_task(_close_after_grace())
    _retiring[task] = entry
    task.add_done_callback(lambda done: _retiring.pop(done, None))


async def apply_server_state(
    state: HubServerState,
    settings: ContainerSettings,
) -> HubEntry:
    """Build a fresh hub for this server, swap it in, then close the old one."""
    async with _lock_for(state.slug):
        current = _hubs.get(state.slug)
        if current is not None and current.generation >= state.generation:
            logger.info(
                f"Ignoring push for '{state.slug}' generation {state.generation}; "
                f"running generation is {current.generation}"
            )
            return current
        new_entry = await _build_hub(state, settings)
        _hubs[state.slug] = new_entry
        if current is not None:
            _retire_later(current)
        return new_entry


async def delete_server(
    slug: str,
) -> bool:
    """Remove a hub. Returns False when no hub had that slug."""
    async with _lock_for(slug):
        entry = _hubs.pop(slug, None)
        if entry is None:
            return False
        await _close_entry(entry)
        logger.info(f"Hub '{slug}' deleted")
        return True


def get_hub(
    slug: str,
) -> HubEntry | None:
    return _hubs.get(slug)


def status_snapshot() -> dict[str, ServerStatus]:
    """Generation and per-target status of every running hub."""
    return {
        slug: ServerStatus(generation=entry.generation, targets=entry.status)
        for slug, entry in _hubs.items()
    }


def hub_count() -> int:
    return len(_hubs)


async def close_all() -> None:
    """Close every hub at shutdown, including replaced hubs still in their grace period."""
    for slug in list(_hubs):
        await delete_server(slug)
    for task, entry in list(_retiring.items()):
        task.cancel()
        await _close_entry(entry)
    _retiring.clear()
