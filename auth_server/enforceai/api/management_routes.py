from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
)

from ..auth.dependency import (
    EnforceAIManagementContext,
    get_enforceai_management_context,
)
from .management_common import (
    _require_admin,
)
from .admin_scopes_routes import (
    router as admin_scopes_router,
)
from .admin_users_routes import (
    router as admin_users_router,
)
from .agents_routes import (
    router as agents_router,
)
from .api_keys_routes import (
    router as api_keys_router,
)
from .audit_routes import (
    router as audit_router,
)
from .egress_allowlist_routes import (
    router as egress_allowlist_router,
)
from .scope_catalog_routes import (
    router as scope_catalog_router,
)
from .tokens_routes import (
    router as tokens_router,
)
from .upstream_credentials_routes import (
    router as upstream_credentials_router,
)
from .upstream_oauth_routes import (
    router as upstream_oauth_router,
)
from .upstream_oauth_provider_routes import (
    router as upstream_oauth_provider_router,
)

router = APIRouter(
    prefix="/enforceai",
    tags=["enforceai-management"],
)
router.include_router(admin_scopes_router)
router.include_router(admin_users_router)
router.include_router(agents_router)
router.include_router(api_keys_router)
router.include_router(tokens_router)
router.include_router(scope_catalog_router)
router.include_router(audit_router)
router.include_router(egress_allowlist_router)
router.include_router(upstream_oauth_provider_router)
router.include_router(upstream_credentials_router)
router.include_router(upstream_oauth_router)


@router.get("/admin/ping")
async def admin_ping(
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
) -> dict[str, bool]:
    _require_admin(context)
    return {"ok": True}
