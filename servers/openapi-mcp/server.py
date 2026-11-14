"""openapi-mcp: serve REST APIs described by OpenAPI 3.x specs as MCP servers.

The registry owns all state and pushes it here. This process keeps one FastMCP
hub per OpenAPI MCP server, in memory only:

- ``/s/<slug>/mcp``: the MCP endpoint of one server. It has no auth of its own;
  the gateway's ``auth_request`` guards it, and only nginx and the registry can
  reach this container.
- ``/internal/*``: the registry's control API, each call carrying a fresh sync
  token (``registry/auth/openapi_mcp_sync.py``).
- ``/health``: liveness, no auth.
"""

import argparse
import json
import logging
import time
from contextlib import asynccontextmanager

import hubs
import uvicorn
from conversion import (
    ConversionError,
    preview,
)
from guarded_transport import USER_CREDENTIALS_SCOPE_KEY
from logging_setup import setup_container_logging
from models import (
    ApplyResponse,
    HubServerState,
    PreviewRequest,
    StatusResponse,
)
from pydantic import ValidationError
from settings import (
    ContainerSettings,
    GuardSettings,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    Response,
)
from starlette.routing import Route

from registry.auth.openapi_mcp_sync import (
    ENVELOPE_HEADER,
    JtiCache,
    SyncTokenError,
    verify_credential_envelope,
    verify_sync_token,
)
from registry.common.secret_key import validate_signing_secret
from registry.utils import url_guard

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


SLUG_PREFIX: str = "/s/"
# Only these inbound headers reach a hub. FastMCP's OpenAPI tools copy every
# inbound header onto the backend request except a short exclusion list, so
# without this allowlist the gateway's identity, forwarding and routing headers
# would be sent to third-party APIs on every call.
MCP_REQUEST_HEADER_ALLOWLIST: frozenset[bytes] = frozenset(
    {
        b"content-type",
        b"content-length",
        b"accept",
        b"mcp-protocol-version",
        b"mcp-session-id",
        b"mcp-method",
        b"mcp-name",
        b"last-event-id",
    }
)

settings = ContainerSettings()
_seen_jtis = JtiCache()
_started_at: float = time.time()


def _error(
    status_code: int,
    code: str,
    detail: str,
) -> JSONResponse:
    return JSONResponse({"error": code, "detail": detail}, status_code=status_code)


def _check_sync_token(
    request: Request,
    purpose: str,
) -> JSONResponse | None:
    """Return a 401 response when the sync token is missing or invalid, else None."""
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return _error(401, "unauthorized", "missing bearer sync token")
    try:
        verify_sync_token(token, settings.openapi_mcp_sync_secret.get_secret_value(), _seen_jtis)
    except SyncTokenError as e:
        logger.warning(f"Rejected {purpose} call: {e}")
        return _error(401, "unauthorized", "invalid sync token")
    return None


async def _put_server(
    request: Request,
) -> Response:
    if (denied := _check_sync_token(request, "put_server")) is not None:
        return denied
    slug = request.path_params["slug"]
    try:
        state = HubServerState.model_validate(await request.json())
    except (ValidationError, ValueError) as e:
        return _error(400, "invalid_state", f"server state failed validation: {type(e).__name__}")
    if state.slug != slug:
        return _error(400, "invalid_state", "slug in the path and the body differ")
    entry = await hubs.apply_server_state(state, settings)
    body = ApplyResponse(slug=slug, generation=entry.generation, targets=entry.status)
    return JSONResponse(body.model_dump())


async def _delete_server(
    request: Request,
) -> Response:
    if (denied := _check_sync_token(request, "delete_server")) is not None:
        return denied
    found = await hubs.delete_server(request.path_params["slug"])
    if not found:
        return _error(404, "not_found", "no hub with that slug")
    return Response(status_code=204)


async def _preview(
    request: Request,
) -> Response:
    if (denied := _check_sync_token(request, "preview")) is not None:
        return denied
    try:
        body = PreviewRequest.model_validate(await request.json())
    except (ValidationError, ValueError) as e:
        return _error(
            400, "invalid_request", f"preview request failed validation: {type(e).__name__}"
        )
    try:
        result = await preview(body, settings.openapi_mcp_conversion_timeout_seconds)
    except ConversionError as e:
        logger.warning(f"Preview of target '{body.target_name}' failed: {e}")
        return _error(422, "conversion_failed", str(e))
    return JSONResponse(result.model_dump())


async def _status(
    request: Request,
) -> Response:
    if (denied := _check_sync_token(request, "status")) is not None:
        return denied
    body = StatusResponse(servers=hubs.status_snapshot(), started_at=_started_at)
    return JSONResponse(body.model_dump())


async def _health(
    request: Request,
) -> Response:
    return JSONResponse({"status": "ok", "servers": hubs.hub_count(), "started_at": _started_at})


@asynccontextmanager
async def _lifespan(
    app: Starlette,
):
    yield
    await hubs.close_all()


control_app = Starlette(
    routes=[
        Route("/internal/servers/{slug}", _put_server, methods=["PUT"]),
        Route("/internal/servers/{slug}", _delete_server, methods=["DELETE"]),
        Route("/internal/preview", _preview, methods=["POST"]),
        Route("/internal/status", _status, methods=["GET"]),
        Route("/health", _health, methods=["GET"]),
    ],
    lifespan=_lifespan,
)


async def _not_found(
    scope,
    receive,
    send,
) -> None:
    response = _error(404, "not_found", "no OpenAPI MCP server at this path")
    await response(scope, receive, send)


def _user_credentials(
    scope,
    slug: str,
) -> dict | None:
    """Verified per-user keys from the gateway's envelope; {} when there is none; None when invalid."""
    wanted = ENVELOPE_HEADER.lower().encode()
    raw = next((v for k, v in scope.get("headers", []) if k.lower() == wanted), None)
    if raw is None:
        return {}
    try:
        return verify_credential_envelope(
            raw.decode("latin-1"),
            settings.openapi_mcp_sync_secret.get_secret_value(),
            slug,
            _seen_jtis,
        )
    except SyncTokenError as e:
        logger.warning(f"Rejected credential envelope for '{slug}': {e}")
        return None


async def app(
    scope,
    receive,
    send,
) -> None:
    """Send /s/<slug>/... to that hub with only allowlisted headers; the rest to the control app."""
    path = scope.get("path", "")
    if scope["type"] != "http" or not path.startswith(SLUG_PREFIX):
        await control_app(scope, receive, send)
        return
    slug, _, rest = path[len(SLUG_PREFIX) :].partition("/")
    entry = hubs.get_hub(slug)
    if entry is None:
        await _not_found(scope, receive, send)
        return
    credentials = _user_credentials(scope, slug)
    if credentials is None:
        await _error(401, "unauthorized", "invalid credential envelope")(scope, receive, send)
        return
    # The envelope header is not in the allowlist, so FastMCP never sees it and
    # cannot forward it; the verified keys travel on the scope instead.
    headers = [
        (k, v) for k, v in scope.get("headers", []) if k.lower() in MCP_REQUEST_HEADER_ALLOWLIST
    ]
    child_scope = dict(
        scope,
        path="/" + rest,
        raw_path=("/" + rest).encode(),
        root_path=scope.get("root_path", "") + SLUG_PREFIX + slug,
        headers=headers,
    )
    child_scope[USER_CREDENTIALS_SCOPE_KEY] = credentials
    await entry.app(child_scope, receive, send)


def _configure_guard() -> None:
    """Point url_guard at the guard-only settings, so it never loads the registry config."""
    url_guard.settings = GuardSettings()


def _validate_startup() -> None:
    """Fail closed on a missing or weak sync secret."""
    validate_signing_secret(
        settings.openapi_mcp_sync_secret.get_secret_value(),
        "OPENAPI_MCP_SYNC_SECRET",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve REST APIs from OpenAPI specs as MCP servers, with state pushed by the registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    export OPENAPI_MCP_SYNC_SECRET=...   # same value as the registry's
    uv run python server.py --port 9876
""",
    )
    parser.add_argument(
        "--host", default=settings.host, help=f"Bind host, or HOST (default: {settings.host})"
    )
    parser.add_argument(
        "--port", type=int, default=settings.port, help=f"Port, or PORT (default: {settings.port})"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_file = setup_container_logging(settings)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
    logger.info(f"Logging to the console and {log_file or 'no file'}")
    _validate_startup()
    _configure_guard()
    config = settings.model_dump(exclude={"openapi_mcp_sync_secret"})
    config.update({"host": args.host, "port": args.port})
    logger.info(f"Configuration:\n{json.dumps(config, indent=2, default=str)}")
    uvicorn.run(app, host=args.host, port=args.port, lifespan="on")


if __name__ == "__main__":
    main()
