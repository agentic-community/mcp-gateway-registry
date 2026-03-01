"""
FastAPI middleware for audit logging.

This module provides middleware that captures request/response
envelope and identity context for every API request, creating
structured audit records.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, List, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .models import (
    Action,
    Authorization,
    Identity,
    RegistryApiAccessRecord,
    Request as AuditRequest,
    Response as AuditResponse,
)
from .service import AuditLogger
from ..utils.request_utils import get_client_ip

logger = logging.getLogger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware that captures request/response data for audit logging.

    Creates structured audit records for every API request, including
    identity context, request/response details, and optional action context.

    Attributes:
        audit_logger: The AuditLogger service for writing events
        exclude_paths: List of paths to exclude from logging
        log_health_checks: Whether to log health check requests
        log_static_assets: Whether to log static asset requests
    """

    def __init__(
        self,
        app: ASGIApp,
        audit_logger: AuditLogger,
        exclude_paths: Optional[List[str]] = None,
        log_health_checks: bool = False,
        log_static_assets: bool = False,
    ):
        """
        Initialize the AuditMiddleware.

        Args:
            app: The ASGI application
            audit_logger: AuditLogger service instance
            exclude_paths: List of paths to exclude from audit logging
            log_health_checks: Whether to log health check endpoints (default: False)
            log_static_assets: Whether to log static asset requests (default: False)
        """
        super().__init__(app)
        self.audit_logger = audit_logger
        self.exclude_paths = exclude_paths or []
        self.log_health_checks = log_health_checks
        self.log_static_assets = log_static_assets

    def _should_log(self, path: str) -> bool:
        """
        Determine if a request should be logged.

        Args:
            path: The request path

        Returns:
            True if the request should be logged, False otherwise
        """
        # Check explicit exclusions
        if path in self.exclude_paths:
            return False

        # Check health check endpoints
        if not self.log_health_checks and "/health" in path.lower():
            return False

        # Check static assets
        if not self.log_static_assets:
            if path.startswith("/static"):
                return False
            if path.startswith("/favicon"):
                return False
            # Common static file extensions
            static_extensions = (
                ".css",
                ".js",
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".ico",
                ".svg",
                ".woff",
                ".woff2",
                ".ttf",
            )
            if path.endswith(static_extensions):
                return False

        return True

    def _get_credential_type(self, request: Request) -> str:
        """
        Determine the type of credential used for authentication.

        Args:
            request: The FastAPI request object

        Returns:
            Credential type: 'session_cookie', 'bearer_token', or 'none'
        """
        from ..core.config import settings

        # Check for session cookie (use configured cookie name)
        if request.cookies.get(settings.session_cookie_name):
            return "session_cookie"

        # Check for bearer token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return "bearer_token"

        return "none"

    def _get_credential_hint(self, request: Request) -> Optional[str]:
        """
        Extract credential hint for audit logging.

        The hint will be masked by the Identity model validator.

        Args:
            request: The FastAPI request object

        Returns:
            Raw credential value (will be masked), or None
        """
        from ..core.config import settings

        # Check for session cookie (use configured cookie name)
        session = request.cookies.get(settings.session_cookie_name)
        if session:
            return session  # Will be masked by validator

        # Check for bearer token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]  # Will be masked by validator

        return None

    def _extract_identity(self, request: Request) -> Identity:
        """
        Extract identity information from the request.

        Looks for user context in request.state (set by auth dependency)
        or falls back to anonymous identity.

        Args:
            request: The FastAPI request object

        Returns:
            Identity model with user information
        """
        # Try to get user context from request state (set by auth dependency)
        user_context = getattr(request.state, "user_context", None)

        if user_context and isinstance(user_context, dict):
            return Identity(
                username=user_context.get("username", "anonymous"),
                auth_method=user_context.get("auth_method", "anonymous"),
                provider=user_context.get("provider"),
                groups=user_context.get("groups", []),
                scopes=user_context.get("scopes", []),
                is_admin=user_context.get("is_admin", False),
                credential_type=self._get_credential_type(request),
                credential_hint=self._get_credential_hint(request),
            )

        # Fallback to anonymous identity
        return Identity(
            username="anonymous",
            auth_method="anonymous",
            credential_type=self._get_credential_type(request),
            credential_hint=self._get_credential_hint(request),
        )

    def _extract_action(self, request: Request) -> Optional[Action]:
        """
        Extract action context from the request.

        Route handlers can set audit_action in request.state to provide
        semantic context about the operation being performed.

        Args:
            request: The FastAPI request object

        Returns:
            Action model if audit_action is set, None otherwise
        """
        audit_action = getattr(request.state, "audit_action", None)

        if audit_action and isinstance(audit_action, dict):
            return Action(
                operation=audit_action.get("operation", "unknown"),
                resource_type=audit_action.get("resource_type", "unknown"),
                resource_id=audit_action.get("resource_id"),
                description=audit_action.get("description"),
            )

        return None

    def _extract_authorization(self, request: Request) -> Optional[Authorization]:
        """
        Extract authorization decision from the request.

        Route handlers can set audit_authorization in request.state to
        record the authorization decision for the request.

        Args:
            request: The FastAPI request object

        Returns:
            Authorization model if audit_authorization is set, None otherwise
        """
        audit_auth = getattr(request.state, "audit_authorization", None)

        if audit_auth and isinstance(audit_auth, dict):
            return Authorization(
                decision=audit_auth.get("decision", "NOT_REQUIRED"),
                required_permission=audit_auth.get("required_permission"),
                evaluated_scopes=audit_auth.get("evaluated_scopes", []),
            )

        return None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and create an audit record.

        Args:
            request: The FastAPI request object
            call_next: The next middleware/handler in the chain

        Returns:
            The response from the next handler
        """
        # Check if this request should be logged
        if not self._should_log(request.url.path):
            return await call_next(request)

        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        correlation_id = request.headers.get("X-Correlation-ID")

        # Start timing
        start_time = time.perf_counter()

        # Process the request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Extract client IP (validated against spoofed/malformed headers)
        client_ip = get_client_ip(request)

        # Get content length from request headers (may be None)
        request_content_length = None
        if "content-length" in request.headers:
            try:
                request_content_length = int(request.headers["content-length"])
            except (ValueError, TypeError):
                pass

        # Get content length from response headers (may be None)
        response_content_length = None
        if "content-length" in response.headers:
            try:
                response_content_length = int(response.headers["content-length"])
            except (ValueError, TypeError):
                pass

        # Build the audit record
        try:
            record = RegistryApiAccessRecord(
                timestamp=datetime.now(timezone.utc),
                request_id=request_id,
                correlation_id=correlation_id,
                identity=self._extract_identity(request),
                request=AuditRequest(
                    method=request.method,
                    path=request.url.path,
                    query_params=dict(request.query_params),
                    client_ip=client_ip,
                    forwarded_for=request.headers.get("X-Forwarded-For"),
                    user_agent=request.headers.get("User-Agent"),
                    content_length=request_content_length,
                ),
                response=AuditResponse(
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                    content_length=response_content_length,
                ),
                action=self._extract_action(request),
                authorization=self._extract_authorization(request),
            )

            # Log the event asynchronously
            await self.audit_logger.log_event(record)

        except Exception as e:
            # Don't let audit logging failures break the request
            logger.error(f"Failed to create audit record: {e}")

        return response


def add_audit_middleware(
    app,
    audit_logger: AuditLogger,
    exclude_paths: Optional[List[str]] = None,
    log_health_checks: bool = False,
    log_static_assets: bool = False,
) -> None:
    """
    Convenience function to add audit middleware to a FastAPI app.

    Args:
        app: FastAPI application instance
        audit_logger: AuditLogger service instance
        exclude_paths: List of paths to exclude from audit logging
        log_health_checks: Whether to log health check endpoints
        log_static_assets: Whether to log static asset requests
    """
    app.add_middleware(
        AuditMiddleware,
        audit_logger=audit_logger,
        exclude_paths=exclude_paths,
        log_health_checks=log_health_checks,
        log_static_assets=log_static_assets,
    )
    logger.info("Audit middleware added to application")
