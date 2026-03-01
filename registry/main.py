#!/usr/bin/env python3
"""
MCP Gateway Registry - Modern FastAPI Application

A clean, domain-driven FastAPI app for managing MCP (Model Context Protocol) servers.
This main.py file serves as the application coordinator, importing and registering
domain routers while handling core app configuration.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, Cookie, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

# Import registry mode middleware
from registry.middleware.mode_filter import RegistryModeMiddleware

# Import domain routers
from registry.auth.routes import router as auth_router
from registry.api.server_routes import router as servers_router
from registry.api.search_routes import router as search_router
from registry.api.wellknown_routes import router as wellknown_router
from registry.api.registry_routes import router as registry_router
from registry.api.agent_routes import router as agent_router
from registry.api.management_routes import router as management_router
from registry.api.federation_routes import router as federation_router
from registry.api.federation_export_routes import router as federation_export_router
from registry.api.peer_management_routes import router as peer_management_router
from registry.api.skill_routes import router as skill_router
from registry.api.config_routes import router as config_router
from registry.api.virtual_server_routes import router as virtual_server_router
from registry.api.internal_routes import router as internal_router
from registry.health.routes import router as health_router
from registry.audit.routes import router as audit_router
from registry.api.system_routes import router as system_router, set_server_start_time

# Import auth dependencies
from registry.auth.dependencies import (
    get_ui_permissions_for_user,
    nginx_proxied_auth,
)

# Import services for initialization
from registry.services.server_service import server_service
from registry.services.agent_service import agent_service
from registry.repositories.factory import get_search_repository
from registry.health.service import health_service
from registry.core.nginx_service import nginx_service
from registry.services.peer_federation_service import get_peer_federation_service
from registry.services.peer_sync_scheduler import get_peer_sync_scheduler

# Import core configuration
from registry.core.config import (
    settings,
    RegistryMode,
    _validate_mode_combination,
    _print_config_warning_banner,
)
from registry.core.metrics import DEPLOYMENT_MODE_INFO

# Import audit logging
from registry.audit import AuditLogger, add_audit_middleware

# Import version
from registry.version import __version__

# Import datetime for uptime tracking
from datetime import datetime, timezone
from typing import Optional


# Server start time tracking moved to registry/api/system_routes.py


# Configure logging with file and console handlers
def setup_logging():
    """Configure logging to write to both file and console."""
    # Ensure log directory exists
    log_dir = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define log file path
    log_file = log_dir / "registry.log"

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s"
    )

    console_formatter = logging.Formatter(
        "%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s"
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return log_file


# Setup logging
log_file_path = setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"Logging configured. Writing to file: {log_file_path}")


def _log_startup_configuration() -> None:
    """Log startup configuration with clear formatting."""
    logger.info("=" * 60)
    logger.info("Registry starting with:")
    logger.info(f"  - DEPLOYMENT_MODE: {settings.deployment_mode.value}")
    logger.info(f"  - REGISTRY_MODE: {settings.registry_mode.value}")
    logger.info(f"  - Nginx updates: {'ENABLED' if settings.nginx_updates_enabled else 'DISABLED'}")

    # Log what's disabled based on registry mode
    if settings.registry_mode == RegistryMode.SKILLS_ONLY:
        logger.info("  - Running in skills-only mode:")
        logger.info("    - MCP servers API: DISABLED")
        logger.info("    - A2A agents API: DISABLED")
        logger.info("    - Federation API: DISABLED")
        logger.info("    - Skills API: ENABLED")
    elif settings.registry_mode == RegistryMode.MCP_SERVERS_ONLY:
        logger.info("  - Running in mcp-servers-only mode:")
        logger.info("    - MCP servers API: ENABLED")
        logger.info("    - A2A agents API: DISABLED")
        logger.info("    - Skills API: DISABLED")
        logger.info("    - Federation API: DISABLED")
    elif settings.registry_mode == RegistryMode.AGENTS_ONLY:
        logger.info("  - Running in agents-only mode:")
        logger.info("    - A2A agents API: ENABLED")
        logger.info("    - MCP servers API: DISABLED")
        logger.info("    - Skills API: DISABLED")
        logger.info("    - Federation API: DISABLED")

    logger.info("=" * 60)


def _initialize_deployment_metrics() -> None:
    """Initialize deployment mode Prometheus metrics."""
    DEPLOYMENT_MODE_INFO.labels(
        deployment_mode=settings.deployment_mode.value, registry_mode=settings.registry_mode.value
    ).set(1)


# Stats and deployment detection functions moved to registry/api/system_routes.py


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    # Record server start time for uptime tracking
    server_start_time = datetime.now(timezone.utc)
    set_server_start_time(server_start_time)
    logger.info(f"Server started at: {server_start_time.isoformat()}")

    logger.info("🚀 Starting MCP Gateway Registry...")

    # Validate and potentially correct mode combination
    original_deployment = settings.deployment_mode
    original_registry = settings.registry_mode

    corrected_deployment, corrected_registry, was_corrected = _validate_mode_combination(
        original_deployment, original_registry
    )

    if was_corrected:
        _print_config_warning_banner(
            original_deployment, original_registry, corrected_deployment, corrected_registry
        )
        # Update settings (use object.__setattr__ for frozen pydantic settings)
        object.__setattr__(settings, "deployment_mode", corrected_deployment)
        object.__setattr__(settings, "registry_mode", corrected_registry)

    # Log startup configuration
    _log_startup_configuration()

    # Initialize Prometheus metrics
    _initialize_deployment_metrics()

    # Initialize audit logger reference (middleware added at module level)
    audit_logger = getattr(app.state, "audit_logger", None)
    if audit_logger:
        logger.info(f"✅ Audit logging enabled. Writing to: {settings.audit_log_path}")

    try:
        # Load scopes configuration from repository
        logger.info("🔐 Loading scopes configuration from repository...")
        from registry.auth.dependencies import reload_scopes_from_repository

        await reload_scopes_from_repository()

        # Initialize services in order
        logger.info("📚 Loading server definitions and state...")
        await server_service.load_servers_and_state()

        # Get repository based on STORAGE_BACKEND configuration
        search_repo = get_search_repository()
        backend_name = "DocumentDB" if settings.storage_backend == "documentdb" else "FAISS"

        logger.info(f"🔍 Initializing {backend_name} search service...")
        await search_repo.initialize()

        logger.info(f"📊 Updating {backend_name} index with all registered services...")
        all_servers = await server_service.get_all_servers()
        for service_path, server_info in all_servers.items():
            is_enabled = await server_service.is_service_enabled(service_path)
            try:
                await search_repo.index_server(service_path, server_info, is_enabled)
                logger.debug(f"Updated {backend_name} index for service: {service_path}")
            except Exception as e:
                logger.error(
                    f"Failed to update {backend_name} index for service {service_path}: {e}",
                    exc_info=True,
                )

        logger.info(f"✅ {backend_name} index updated with {len(all_servers)} services")

        logger.info("📋 Loading agent cards and state...")
        await agent_service.load_agents_and_state()

        logger.info(f"📊 Updating {backend_name} index with all registered agents...")
        all_agents = agent_service.list_agents()
        for agent_card in all_agents:
            is_enabled = agent_service.is_agent_enabled(agent_card.path)
            try:
                await search_repo.index_agent(agent_card.path, agent_card, is_enabled)
                logger.debug(f"Updated {backend_name} index for agent: {agent_card.path}")
            except Exception as e:
                logger.error(
                    f"Failed to update {backend_name} index for agent {agent_card.path}: {e}",
                    exc_info=True,
                )

        logger.info(f"✅ {backend_name} index updated with {len(all_agents)} agents")

        logger.info("🏥 Initializing health monitoring service...")
        await health_service.initialize()

        logger.info("🔗 Checking federation configuration...")
        from registry.repositories.factory import get_federation_config_repository

        try:
            # Load federation config
            federation_repo = get_federation_config_repository()
            federation_config = await federation_repo.get_config("default")

            if federation_config and federation_config.is_any_federation_enabled():
                logger.info(
                    f"Federation enabled for: {', '.join(federation_config.get_enabled_federations())}"
                )

                # Sync on startup if configured
                sync_on_startup = (
                    federation_config.anthropic.enabled
                    and federation_config.anthropic.sync_on_startup
                ) or (federation_config.asor.enabled and federation_config.asor.sync_on_startup)

                if sync_on_startup:
                    logger.info("🔄 Syncing servers from federated registries on startup...")
                    try:
                        from registry.services.federation.anthropic_client import (
                            AnthropicFederationClient,
                        )

                        # Sync Anthropic servers if enabled and sync_on_startup is true
                        if (
                            federation_config.anthropic.enabled
                            and federation_config.anthropic.sync_on_startup
                        ):
                            logger.info("Syncing from Anthropic MCP Registry...")
                            anthropic_client = AnthropicFederationClient(
                                endpoint=federation_config.anthropic.endpoint
                            )
                            servers = anthropic_client.fetch_all_servers(
                                federation_config.anthropic.servers
                            )

                            # Register servers
                            synced_count = 0
                            for server_data in servers:
                                try:
                                    server_path = server_data.get("path")
                                    if not server_path:
                                        continue

                                    # Register or update server
                                    success = await server_service.register_server(server_data)
                                    if not success:
                                        success = await server_service.update_server(
                                            server_path, server_data
                                        )

                                    if success:
                                        # Enable the server
                                        await server_service.toggle_service(server_path, True)
                                        synced_count += 1
                                        logger.info(
                                            f"Synced: {server_data.get('server_name', server_path)}"
                                        )
                                except Exception as e:
                                    logger.error(
                                        f"Failed to sync server {server_data.get('server_name', 'unknown')}: {e}"
                                    )

                            logger.info(f"✅ Synced {synced_count} servers from Anthropic")

                            # Run reconciliation after sync to remove stale servers
                            logger.info("🔄 Running reconciliation after startup sync...")
                            try:
                                from registry.services.federation_reconciliation import (
                                    reconcile_anthropic_servers,
                                )
                                from registry.repositories.factory import (
                                    get_server_repository,
                                )

                                server_repo = get_server_repository()
                                reconciliation_result = await reconcile_anthropic_servers(
                                    config=federation_config,
                                    server_service=server_service,
                                    server_repo=server_repo,
                                    nginx_service=None,
                                    skip_nginx_regen=True,
                                )

                                logger.info(
                                    f"✅ Reconciliation complete: "
                                    f"removed {reconciliation_result['removed_count']} stale servers, "
                                    f"expected {reconciliation_result['expected_count']}, "
                                    f"found {reconciliation_result['actual_count']} in DB"
                                )
                            except Exception as e:
                                logger.error(
                                    f"⚠️ Reconciliation failed (continuing with startup): {e}",
                                    exc_info=True,
                                )

                        # ASOR sync would go here if needed

                    except Exception as e:
                        logger.error(
                            f"⚠️ Federation sync failed (continuing with startup): {e}",
                            exc_info=True,
                        )
            else:
                logger.info("Federation is disabled or not configured")
        except Exception as e:
            logger.error(f"Failed to load federation config: {e}")
            logger.info("Continuing without federation")

        logger.info("Initializing peer federation service...")
        peer_federation_service = get_peer_federation_service()
        await peer_federation_service.load_peers_and_state()
        logger.info(f"Loaded {len(peer_federation_service.registered_peers)} peer registries")

        # Start peer sync scheduler for scheduled federation sync
        logger.info("Starting peer sync scheduler...")
        peer_sync_scheduler = get_peer_sync_scheduler()
        await peer_sync_scheduler.start()
        logger.info("Peer sync scheduler started")

        # Always generate nginx configuration at startup to ensure placeholders are replaced
        # In registry-only mode, generate base config without MCP server location blocks
        if settings.nginx_updates_enabled:
            logger.info("Generating initial Nginx configuration with MCP server locations...")
            enabled_service_paths = await server_service.get_enabled_services()
            enabled_servers = {}
            for path in enabled_service_paths:
                server_info = await server_service.get_server_info(path)
                if server_info:
                    enabled_servers[path] = server_info
            await nginx_service.generate_config_async(enabled_servers)
        else:
            logger.info("Generating base Nginx configuration (registry-only mode)...")
            # Generate base config with empty location blocks but substitute all placeholders
            await nginx_service.generate_config_async({}, force_base_config=True)

        logger.info("✅ All services initialized successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}", exc_info=True)
        raise

    # Application is ready
    yield

    # Shutdown tasks
    logger.info("🔄 Shutting down MCP Gateway Registry...")
    try:
        # Stop peer sync scheduler
        peer_sync_scheduler = get_peer_sync_scheduler()
        await peer_sync_scheduler.stop()

        # Shutdown audit logger if enabled
        if audit_logger is not None:
            logger.info("📝 Closing audit logger...")
            await audit_logger.close()

        # Shutdown services gracefully
        await health_service.shutdown()
        logger.info("✅ Shutdown completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}", exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="MCP Gateway Registry",
    description="A registry and management system for Model Context Protocol (MCP) servers",
    version=__version__,
    lifespan=lifespan,
    root_path=os.environ.get("ROOT_PATH", ""),  # Support path-based routing with ALB
    swagger_ui_parameters={
        "persistAuthorization": True,
    },
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "OAuth2 and session-based authentication endpoints",
        },
        {
            "name": "Server Management",
            "description": "MCP server registration and management. Requires JWT Bearer token authentication.",
        },
        {
            "name": "Agent Management",
            "description": "A2A agent registration and management. Requires JWT Bearer token authentication.",
        },
        {
            "name": "Management API",
            "description": "IAM and user management operations. Requires JWT Bearer token with admin permissions.",
        },
        {
            "name": "Semantic Search",
            "description": "Vector-based semantic search for agents. Requires JWT Bearer token authentication.",
        },
        {"name": "Health Monitoring", "description": "Service health check endpoints"},
        {
            "name": "Anthropic Registry API",
            "description": "Anthropic-compatible registry API (v0.1) for MCP server discovery",
        },
        {
            "name": "federation",
            "description": "Federation configuration and peer-to-peer registry synchronization APIs",
        },
        {
            "name": "peer-management",
            "description": "Peer registry management API for configuring and synchronizing with peer registries. Requires JWT Bearer token authentication.",
        },
        {
            "name": "Audit Logs",
            "description": "Audit log viewing and export endpoints. Requires admin permissions.",
        },
        {
            "name": "skills",
            "description": "Agent Skills registration and management. Requires JWT Bearer token authentication.",
        },
        {
            "name": "virtual-servers",
            "description": "Virtual MCP Server management. Aggregate tools from multiple backends into unified endpoints.",
        },
    ],
)

# Add CORS middleware for React development and Docker deployment
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost(:[0-9]+)?|.*\.compute.*\.amazonaws\.com(:[0-9]+)?)",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add registry mode middleware to filter endpoints based on REGISTRY_MODE
# This must be after CORS (to allow preflight) and before audit (to log blocked requests)
if settings.registry_mode != RegistryMode.FULL:
    logger.info(f"Adding registry mode middleware - mode: {settings.registry_mode.value}")
    app.add_middleware(RegistryModeMiddleware)

# Add audit middleware if enabled (must be added before app starts)
if settings.audit_log_enabled:
    logger.info("📝 Initializing audit logging...")

    # Get audit repository if MongoDB is enabled
    _audit_repository = None
    _mongodb_enabled = settings.audit_log_mongodb_enabled and settings.storage_backend in (
        "documentdb",
        "mongodb-ce",
    )
    if _mongodb_enabled:
        from registry.repositories.factory import get_audit_repository

        _audit_repository = get_audit_repository()
        if _audit_repository:
            logger.info("📊 MongoDB audit storage enabled")
        else:
            logger.warning("⚠️ MongoDB audit storage requested but repository unavailable")
            _mongodb_enabled = False

    _audit_logger = AuditLogger(
        log_dir=str(settings.audit_log_path),
        rotation_hours=settings.audit_log_rotation_hours,
        rotation_max_mb=settings.audit_log_rotation_max_mb,
        local_retention_hours=settings.audit_log_local_retention_hours,
        stream_name="registry-api-access",
        mongodb_enabled=_mongodb_enabled,
        audit_repository=_audit_repository,
    )
    # Store audit logger in app state for lifespan access
    app.state.audit_logger = _audit_logger

    # Add audit middleware to the app
    add_audit_middleware(
        app,
        audit_logger=_audit_logger,
        log_health_checks=settings.audit_log_health_checks,
        log_static_assets=settings.audit_log_static_assets,
    )

# Register API routers with /api prefix
app.include_router(system_router, tags=["System"])  # /api/version, /api/stats
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(servers_router, prefix="/api", tags=["Server Management"])
app.include_router(agent_router, prefix="/api", tags=["Agent Management"])
app.include_router(management_router, prefix="/api")
app.include_router(search_router, prefix="/api/search", tags=["Semantic Search"])
app.include_router(federation_router, prefix="/api", tags=["federation"])
app.include_router(skill_router, prefix="/api", tags=["skills"])
app.include_router(config_router, prefix="/api/config", tags=["config"])
app.include_router(virtual_server_router, prefix="/api", tags=["virtual-servers"])
app.include_router(internal_router, prefix="/api")
app.include_router(health_router, prefix="/api/health", tags=["Health Monitoring"])
app.include_router(federation_export_router)
app.include_router(peer_management_router)
app.include_router(audit_router, prefix="/api", tags=["Audit Logs"])

# Register Anthropic MCP Registry API (public API for MCP servers only)
app.include_router(registry_router, tags=["Anthropic Registry API"])

# Register well-known discovery router
app.include_router(wellknown_router, prefix="/.well-known", tags=["Discovery"])


# Customize OpenAPI schema to add security schemes
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token obtained from Keycloak OAuth2 authentication. "
            "Include in Authorization header as: `Authorization: Bearer <token>`",
        }
    }

    # Apply Bearer security to all endpoints except auth, health, and public discovery endpoints
    for path, path_item in openapi_schema["paths"].items():
        # Skip authentication, health check, and public discovery endpoints
        if path.startswith("/api/auth/") or path == "/health" or path.startswith("/.well-known/"):
            continue

        # Apply Bearer security to all methods in this path
        for method in path_item:
            if method in ["get", "post", "put", "delete", "patch"]:
                if "security" not in path_item[method]:
                    path_item[method]["security"] = [{"Bearer": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Add user info endpoint for React auth context
@app.get("/api/auth/me")
async def get_current_user(user_context: Dict[str, Any] = Depends(nginx_proxied_auth)):
    """Get current user information for React auth context"""
    # Get user's scopes
    user_scopes = user_context.get("scopes", [])

    # Get UI permissions for the user based on their scopes
    ui_permissions = await get_ui_permissions_for_user(user_scopes)

    # Return user info with scopes and UI permissions for token generation
    return {
        "username": user_context["username"],
        "auth_method": user_context.get("auth_method", "basic"),
        "provider": user_context.get("provider"),
        "scopes": user_scopes,
        "groups": user_context.get("groups", []),
        "can_modify_servers": user_context.get("can_modify_servers", False),
        "is_admin": user_context.get("is_admin", False),
        "ui_permissions": ui_permissions,
        "accessible_servers": user_context.get("accessible_servers", []),
        "accessible_services": user_context.get("accessible_services", []),
        "accessible_agents": user_context.get("accessible_agents", []),
    }


# Basic health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check for load balancers and monitoring."""
    return {
        "status": "healthy",
        "service": "mcp-gateway-registry",
        "deployment_mode": settings.deployment_mode.value,
        "registry_mode": settings.registry_mode.value,
        "nginx_updates_enabled": settings.nginx_updates_enabled,
    }


# Version endpoint for UI
# System endpoints (version, stats) moved to registry/api/system_routes.py


# Serve React static files
FRONTEND_BUILD_PATH = Path(__file__).parent.parent / "frontend" / "build"

# Cache the modified index.html content for path-based routing
# Read once at startup instead of on every request
_CACHED_INDEX_HTML: Optional[str] = None
_ROOT_PATH: str = os.environ.get("ROOT_PATH", "")


def _build_cached_index_html() -> Optional[str]:
    """Read index.html and inject <base> tag if ROOT_PATH is set.

    Returns:
        Modified HTML string if ROOT_PATH is set, None otherwise.
    """
    if not _ROOT_PATH:
        return None

    index_path = FRONTEND_BUILD_PATH / "index.html"
    if not index_path.exists():
        return None

    with open(index_path, "r") as f:
        html_content = f.read()

    # Inject <base> tag if not already present
    if "<base" not in html_content:
        base_href = _ROOT_PATH if _ROOT_PATH.endswith("/") else f"{_ROOT_PATH}/"
        base_tag = f'<base href="{base_href}">'
        html_content = html_content.replace("<head>", f"<head>\n    {base_tag}")

    return html_content


if FRONTEND_BUILD_PATH.exists():
    # Build the cached HTML at import time
    _CACHED_INDEX_HTML = _build_cached_index_html()
    # Mount static files - path depends on ROOT_PATH
    # When ROOT_PATH is set, FastAPI automatically handles the prefix for routes,
    # but we need to explicitly mount static files at the root level
    # The <base> tag in HTML will make browsers request /registry/static/*
    # which FastAPI will handle correctly with root_path
    app.mount("/static", StaticFiles(directory=FRONTEND_BUILD_PATH / "static"), name="static")

    # Serve React app for all other routes (SPA)
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """Serve React app for all non-API routes"""
        # Import here to avoid circular dependency
        from registry.constants import REGISTRY_CONSTANTS

        # Don't serve React for API routes, Anthropic registry API, health checks, well-known discovery endpoints, and static files
        anthropic_api_prefix = f"{REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION}/"
        if (
            full_path.startswith("api/")
            or full_path.startswith(anthropic_api_prefix)
            or full_path.startswith("health")
            or full_path.startswith(".well-known/")
            or full_path.startswith("static/")
        ):  # Let static files mount handle these
            raise HTTPException(status_code=404)

        if _CACHED_INDEX_HTML is not None:
            return HTMLResponse(content=_CACHED_INDEX_HTML)

        return FileResponse(FRONTEND_BUILD_PATH / "index.html")
else:
    logger.warning(
        "React build directory not found. Serve React app separately during development."
    )

    # Serve legacy templates and static files during development
    from fastapi.templating import Jinja2Templates

    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    templates = Jinja2Templates(directory=settings.templates_dir)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "registry.main:app",
        host=os.getenv("REGISTRY_HOST", "127.0.0.1"),  # nosec B104
        port=7860,
        reload=True,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
