"""Distinct audit record for an operation that grants administrator-tier access.

Shared by every route that can set a principal's groups: IdP users, M2M
clients (generic, Okta and Auth0) and the user-group fallback store.
Each of these routes calls :func:`audit_admin_grant` after its own ``set_audit_action``.
"""

import logging

from fastapi import Request

from ..audit.context import set_audit_action
from ..services import admin_safety
from ..services.admin_safety import AdminSafetyError

logger = logging.getLogger(__name__)


async def audit_admin_grant(
    request: Request,
    identity: str,
    desired_groups: list[str] | None,
    operation: str,
    resource_type: str = "user",
) -> None:
    """Replace the request's audit action with an admin-grant record when warranted.

    Records a dedicated event whenever ``desired_groups`` confers admin, so the
    grant is independently traceable. Call it after the mutation succeeds and
    after the route's own ``set_audit_action``, which this overrides.

    Never fails the operation: an audit-derivation error is logged and
    swallowed (the mutation and its own audit entry are unaffected).

    Args:
        request: The request whose audit action to set.
        identity: The username or client_id whose groups were set.
        desired_groups: The group set the principal now holds.
        operation: The audit operation (``create`` or ``update``).
        resource_type: The audit resource type of ``identity``.
    """
    try:
        if await admin_safety.desired_groups_grant_admin(desired_groups):
            logger.warning(
                "admin_grant %s=%s operation=%s: account granted "
                "administrator-tier group membership",
                resource_type,
                identity,
                operation,
            )
            set_audit_action(
                request,
                operation=operation,
                resource_type=resource_type,
                resource_id=identity,
                description=f"Granted administrator-tier group membership to '{identity}'",
                metadata={"admin_grant": True},
            )
    except AdminSafetyError as exc:
        # Could not determine whether this is an admin grant. Do not block the
        # (already-completed) mutation, but log loudly so the gap is visible.
        logger.error(
            "admin_grant audit could not be evaluated for %s=%s: %s",
            resource_type,
            identity,
            exc.detail,
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception("admin_grant audit evaluation failed for %s=%s", resource_type, identity)
