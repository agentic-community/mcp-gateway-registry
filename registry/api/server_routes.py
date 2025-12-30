from fastapi import (
    APIRouter,
)

from .server_external_routes import (
    router as external_router,
)
from .server_groups_routes import (
    router as groups_router,
)
from .server_internal_routes import (
    router as internal_router,
)
from .server_json_routes import (
    router as server_json_router,
)
from .server_tokens_routes import (
    router as tokens_router,
)
from .server_tools_routes import (
    router as tools_router,
)
from .server_ui_routes import (
    router as server_ui_router,
)

router = APIRouter()
router.include_router(internal_router)
router.include_router(external_router)
router.include_router(server_json_router)
router.include_router(server_ui_router)
router.include_router(groups_router)
router.include_router(tools_router)
router.include_router(tokens_router)
