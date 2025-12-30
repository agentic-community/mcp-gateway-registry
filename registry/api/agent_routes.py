"""A2A Agent API routes for MCP Gateway Registry.

This module is the public router entrypoint expected by `registry.main`.
Individual route groups live under `registry.api.agents.*`.
"""

from fastapi import (
    APIRouter,
)

from .agents.crud import (
    router as crud_router,
)
from .agents.discovery import (
    router as discovery_router,
)
from .agents.health import (
    router as health_router,
)
from .agents.listing import (
    router as listing_router,
)
from .agents.ratings import (
    router as ratings_router,
)
from .agents.registration import (
    router as registration_router,
)
from .agents.toggle import (
    router as toggle_router,
)

router = APIRouter()

router.include_router(registration_router)
router.include_router(listing_router)

# Specific subpaths must be registered before the `/{path:path}` catch-all routes.
router.include_router(health_router)
router.include_router(ratings_router)
router.include_router(toggle_router)
router.include_router(discovery_router)
router.include_router(crud_router)
