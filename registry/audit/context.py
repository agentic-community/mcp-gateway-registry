"""
Audit context utilities for route handlers.

This module provides helper functions for setting audit action context
in route handlers, which is then captured by the AuditMiddleware.
"""

from typing import Optional

from fastapi import Request


def set_audit_action(
    request: Request,
    operation: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
    """
    Set audit action context on the request for the AuditMiddleware.

    This function should be called at the beginning of route handlers
    to provide semantic context about the operation being performed.

    Args:
        request: The FastAPI request object
        operation: The operation type (create, read, update, delete, list, toggle, rate, login, logout, search)
        resource_type: The resource type (server, agent, auth, federation, health, search, scope, user, group)
        resource_id: Optional identifier of the resource being acted upon
        description: Optional human-readable description of the action

    Example:
        @router.post("/servers")
        async def create_server(request: Request, ...):
            set_audit_action(request, "create", "server", description="Register new MCP server")
            ...
    """
    request.state.audit_action = {
        "operation": operation,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "description": description,
    }


def set_audit_authorization(
    request: Request,
    decision: str,
    required_permission: Optional[str] = None,
    evaluated_scopes: Optional[list] = None,
) -> None:
    """
    Set authorization decision context on the request for the AuditMiddleware.

    This function can be called by authorization dependencies to record
    the authorization decision for audit purposes.

    Args:
        request: The FastAPI request object
        decision: The authorization decision (ALLOW, DENY, NOT_REQUIRED)
        required_permission: The permission that was required
        evaluated_scopes: List of scopes that were evaluated

    Example:
        def check_permission(request: Request, user_context: dict):
            if user_context.get("is_admin"):
                set_audit_authorization(request, "ALLOW", "admin", user_context.get("scopes", []))
            else:
                set_audit_authorization(request, "DENY", "admin", user_context.get("scopes", []))
                raise HTTPException(status_code=403)
    """
    request.state.audit_authorization = {
        "decision": decision,
        "required_permission": required_permission,
        "evaluated_scopes": evaluated_scopes or [],
    }
