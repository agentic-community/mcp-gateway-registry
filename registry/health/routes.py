import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status

from ..auth.dependencies import nginx_proxied_auth, resolve_session_from_cookie
from ..auth.session_cookie import request_session_cookie
from .service import health_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/health_status")
async def websocket_endpoint(websocket: WebSocket):
    """High-performance WebSocket endpoint for real-time health status updates with authentication."""
    connection_added = False
    try:
        # The browser sends its cookies with the same-origin WebSocket handshake.
        # Validate the session before accepting the connection; a session cookie
        # the handshake carries more than once is refused like a missing one.
        # The session cookie is the only credential read here: a session in the
        # URL would land in access logs, so the query string is never consulted.
        # Closing before accept refuses the handshake: the client gets HTTP 403,
        # and the 1008 code below never reaches it as a close frame.
        logger.info(f"WebSocket connection attempt from {websocket.client}")
        session_cookie = request_session_cookie(websocket)

        logger.debug(f"WebSocket session cookie found: {bool(session_cookie)}")

        if session_cookie:
            session_data = await resolve_session_from_cookie(session_cookie)
            if session_data and session_data.get("username"):
                logger.info(
                    f"WebSocket connection from authenticated user: {session_data['username']}"
                )
            else:
                logger.warning("WebSocket authentication failed")
                await websocket.close(code=1008, reason="Authentication failed")
                return
        else:
            logger.warning(
                f"WebSocket connection without valid session cookie from {websocket.client}"
            )
            await websocket.close(code=1008, reason="Authentication required")
            return

        # Accept connection after successful authentication
        connection_added = await health_service.add_websocket_connection(websocket)
        if not connection_added:
            return  # Connection rejected (server at capacity)

        # Keep connection open and handle client messages
        while True:
            # We don't expect messages from client, but keep alive
            # Add timeout to prevent hanging on slow clients
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except TimeoutError:
                # No client message within the window. Starlette manages
                # protocol-level keepalive, so just loop again and keep
                # waiting (Starlette's WebSocket exposes no ping() method).
                continue

    except WebSocketDisconnect:
        logger.debug(f"WebSocket client disconnected: {websocket.client}")
    except Exception as e:
        logger.warning(f"WebSocket error for {websocket.client}: {e}")
    finally:
        if connection_added:
            await health_service.remove_websocket_connection(websocket)


@router.get("/ws/health_status")
async def health_status_http(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """HTTP endpoint that returns the same health status data as the WebSocket endpoint.

    This handles cases where health checks are done via HTTP GET instead of WebSocket.
    Requires authentication, matching the WebSocket sibling (which validates the
    session cookie) so the full server inventory/health is not exposed anonymously.
    """
    return await health_service.get_all_health_status()


@router.get("/ws/stats")
async def websocket_stats(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Get WebSocket performance statistics for monitoring (admin only)."""
    if not user_context.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator permissions are required for this operation",
        )
    return health_service.get_websocket_stats()
