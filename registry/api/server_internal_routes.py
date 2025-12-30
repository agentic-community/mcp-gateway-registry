from fastapi import (
    APIRouter,
)

from .server_internal_groups_routes import (
    internal_add_server_to_groups,
    internal_create_group,
    internal_delete_group,
    internal_list_groups,
    internal_remove_server_from_groups,
    router as internal_groups_router,
)
from .server_internal_services_routes import (
    internal_healthcheck,
    internal_list_services,
    internal_register_service,
    internal_remove_service,
    internal_toggle_service,
    router as internal_services_router,
)

router = APIRouter()
router.include_router(internal_services_router)
router.include_router(internal_groups_router)
