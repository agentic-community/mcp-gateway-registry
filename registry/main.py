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

# Import datetime for uptime tracking
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from registry.api.agent_routes import router as agent_router
from registry.api.ans_routes import router as ans_router
from registry.api.auth0_m2m_routes import router as auth0_m2m_router
from registry.api.config_routes import router as config_router
from registry.api.federation_export_routes import router as federation_export_router
from registry.api.federation_routes import router as federation_router
from registry.api.internal_routes import router as internal_router
from registry.api.log_routes import router as log_router
from registry.api.m2m_management_routes import router as m2m_management_router
from registry.api.management_routes import router as management_router
from registry.api.okta_m2m_routes import router as okta_m2m_router
from registry.api.peer_management_routes import router as peer_management_router
from registry.api.registry_management_routes import router as registry_management_router
from registry.api.registry_routes import router as registry_router
from registry.api.search_routes import router as search_router
from registry.api.server_routes import router as servers_router
from registry.api.skill_routes import router as skill_router
from registry.api.system_routes import router as system_router
from registry.api.system_routes import set_server_start_time
from registry.api.virtual_server_routes import router as virtual_server_router
from registry.api.wellknown_routes import router as wellknown_router

# Import audit logging
from registry.audit import AuditLogger, add_audit_middleware
from registry.audit.routes import router as audit_router

# Import auth dependencies
from registry.auth.dependencies import (
    get_ui_permissions_for_user,
    nginx_proxied_auth,
)

# Import domain routers
from registry.auth.routes import router as auth_router

# Import core configuration
from registry.core.config import (
    RegistryMode,
    _print_config_warning_banner,
    _validate_mode_combination,
    log_tab_visibility_warnings,
    settings,
)
from registry.core.metrics import DEPLOYMENT_MODE_INFO
from registry.core.nginx_service import nginx_service
from registry.core.telemetry import (
    initialize_telemetry,
    send_startup_ping,
    start_heartbeat_scheduler,
    stop_heartbeat_scheduler,
)
from registry.health.routes import router as health_router
from registry.health.service import health_service

# Import registry mode middleware
from registry.middleware.mode_filter import RegistryModeMiddleware
from registry.repositories.factory import get_search_repository
from registry.services.agent_service import agent_service
from registry.services.peer_federation_service import get_peer_federation_service
from registry.services.peer_sync_scheduler import get_peer_sync_scheduler

# Import services for initialization
from registry.services.server_service import server_service

# Server start time tracking moved to registry/api/system_routes.py
# Setup logging using shared module (RotatingFileHandler + optional MongoDB)
from registry.utils.logging_setup import setup_logging as _setup_logging

# Import version
from registry.version import __version__

log_file_path = _setup_logging(service_name="registry")
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


async def _sync_agentcore_on_startup(
    federation_config: Any,
    server_service: Any,
) -> None:
    """Sync records from AWS Agent Registry on startup.

    Fetches all records from configured AgentCore registries,
    registers them locally, and reconciles stale records.

    Args:
        federation_config: FederationConfig with agentcore settings
        server_service: ServerService for server registration
    """
    from registry.repositories.factory import (
        get_agent_repository,
        get_server_repository,
        get_skill_repository,
    )
    from registry.schemas.agent_models import AgentCard
    from registry.schemas.skill_models import SkillCard
    from registry.services.agent_service import agent_service
    from registry.services.federation.agentcore_client import (
        AgentCoreFederationClient,
    )
    from registry.services.federation_reconciliation import (
        reconcile_agentcore_records,
    )
    from registry.services.skill_service import get_skill_service

    logger.info("Syncing from AWS Agent Registry...")

    agentcore_client = AgentCoreFederationClient(
        aws_region=federation_config.aws_registry.aws_region
    )
    records = agentcore_client.fetch_all_records(
        registry_configs=federation_config.aws_registry.registries,
        sync_timeout_seconds=federation_config.aws_registry.sync_timeout_seconds,
        max_concurrent_fetches=federation_config.aws_registry.max_concurrent_fetches,
    )

    synced_paths: dict[str, set[str]] = {
        "servers": set(),
        "agents": set(),
        "skills": set(),
    }

    # Register servers (MCP records)
    server_count = 0
    for server_data in records["servers"]:
        try:
            server_path = server_data.get("path")
            if not server_path:
                continue

            if "id" not in server_data or not server_data["id"]:
                server_data["id"] = str(uuid4())

            success = await server_service.register_server(server_data)
            if not success:
                if "id" not in server_data or not server_data["id"]:
                    server_data["id"] = str(uuid4())
                success = await server_service.update_server(server_path, server_data)

            if success:
                await server_service.toggle_service(server_path, True)
                server_count += 1
                synced_paths["servers"].add(server_path)
        except Exception as e:
            logger.error(
                f"Failed to sync AgentCore server {server_data.get('server_name', 'unknown')}: {e}"
            )

    # Register agents (A2A + CUSTOM records)
    agent_count = 0
    for agent_data in records["agents"]:
        try:
            agent_path = agent_data.get("path")
            if not agent_path:
                continue

            try:
                agent_card = AgentCard(**agent_data)
                await agent_service.register_agent(agent_card)
                agent_count += 1
                synced_paths["agents"].add(agent_path)
            except ValueError:
                # Path already exists -- update instead
                await agent_service.update_agent(agent_path, agent_data)
                agent_count += 1
                synced_paths["agents"].add(agent_path)
        except Exception as e:
            logger.error(f"Failed to sync AgentCore agent {agent_data.get('name', 'unknown')}: {e}")

    # Register skills (AGENT_SKILLS records)
    skill_count = 0
    skill_service = get_skill_service()
    skill_repo = get_skill_repository()
    for skill_data in records["skills"]:
        try:
            skill_path = skill_data.get("path")
            if not skill_path:
                continue

            try:
                skill_card = SkillCard(**skill_data)
                await skill_repo.create(skill_card)
                skill_count += 1
                synced_paths["skills"].add(skill_path)
            except Exception as create_err:
                # Skill already exists -- update instead
                logger.debug(f"Skill create failed for {skill_path}, trying update: {create_err}")
                update_fields = {
                    k: v for k, v in skill_data.items() if k not in ("path", "id", "created_at")
                }
                await skill_repo.update(skill_path, update_fields)
                skill_count += 1
                synced_paths["skills"].add(skill_path)
        except Exception as e:
            logger.error(f"Failed to sync AgentCore skill {skill_data.get('name', 'unknown')}: {e}")

    logger.info(
        f"Synced from AWS Agent Registry: "
        f"{server_count} servers, {agent_count} agents, {skill_count} skills"
    )

    # Run reconciliation to remove stale records
    logger.info("Running AgentCore reconciliation after startup sync...")
    try:
        server_repo = get_server_repository()
        agent_repo = get_agent_repository()

        reconciliation_result = await reconcile_agentcore_records(
            config=federation_config,
            server_service=server_service,
            server_repo=server_repo,
            agent_repo=agent_repo,
            skill_repo=skill_repo,
            synced_paths=synced_paths,
        )

        logger.info(
            f"AgentCore reconciliation complete: "
            f"removed {reconciliation_result.get('total_removed', 0)} stale records"
        )
    except Exception as e:
        logger.error(
            f"AgentCore reconciliation failed (continuing with startup): {e}",
            exc_info=True,
        )


async def _apply_aws_registry_env_vars() -> None:
    """Apply AWS_REGISTRY_FEDERATION_ENABLED env var to the federation config.

    When the env var is set (e.g. via ECS task definition or .env),
    it overrides the aws_registry.enabled flag in the MongoDB federation
    config document. Creates the config document if it does not exist.

    Other aws_registry settings (region, sync_on_startup, registries)
    are managed exclusively via the /api/federation/config API.
    """
    from registry.repositories.factory import get_federation_config_repository
    from registry.schemas.federation_schema import FederationConfig

    env_enabled = os.environ.get("AWS_REGISTRY_FEDERATION_ENABLED", "").lower()

    if not env_enabled:
        logger.debug("AWS_REGISTRY_FEDERATION_ENABLED not set, skipping env var override")
        return

    enabled = env_enabled in ("true", "1", "yes")
    logger.info(f"AWS_REGISTRY_FEDERATION_ENABLED={enabled} (from env var)")

    federation_repo = get_federation_config_repository()
    config = await federation_repo.get_config("default")

    if config is None:
        config = FederationConfig()

    config.aws_registry.enabled = enabled

    await federation_repo.save_config(config, "default")
    logger.info(f"Federation config updated: aws_registry.enabled={enabled}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    # Record server start time for uptime tracking
    server_start_time = datetime.now(UTC)
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

    # Log warnings for ineffective SHOW_*_TAB overrides
    log_tab_visibility_warnings(settings)

    # Initialize Prometheus metrics
    _initialize_deployment_metrics()

    # Validate required configuration settings
    logger.info("🔍 Validating configuration...")
    errors = []

    if not settings.registry_url:
        errors.append("REGISTRY_URL is required")

    if not settings.registry_name:
        errors.append("REGISTRY_NAME is required")

    if not settings.registry_organization_name:
        errors.append("REGISTRY_ORGANIZATION_NAME is required")

    if errors:
        logger.error(
            "Configuration validation failed",
            extra={"errors": errors},
        )
        raise RuntimeError(f"Configuration errors: {', '.join(errors)}")

    logger.info(
        "Configuration validated successfully",
        extra={
            "registry_name": settings.registry_name,
            "registry_url": settings.registry_url,
            "organization": settings.registry_organization_name,
        },
    )

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

        logger.info(f"📊 Updating {backend_name} index with all registered skills...")
        from registry.repositories.factory import get_skill_repository

        skill_repo = get_skill_repository()
        all_skills = await skill_repo.list_all(skip=0, limit=10000)
        for skill_card in all_skills:
            try:
                await search_repo.index_skill(skill_card.path, skill_card, skill_card.is_enabled)
                logger.debug(f"Updated {backend_name} index for skill: {skill_card.path}")
            except Exception as e:
                logger.error(
                    f"Failed to update {backend_name} index for skill {skill_card.path}: {e}",
                    exc_info=True,
                )

        logger.info(f"✅ {backend_name} index updated with {len(all_skills)} skills")

        logger.info("🏥 Initializing health monitoring service...")
        await health_service.initialize()

        logger.info("🔗 Checking federation configuration...")
        from registry.repositories.factory import get_federation_config_repository

        # Apply env var overrides (e.g. from ECS/terraform) before loading config
        try:
            await _apply_aws_registry_env_vars()
        except Exception as e:
            logger.error(
                f"Failed to apply AWS Registry env vars (continuing with startup): {e}",
                exc_info=True,
            )

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
                    (
                        federation_config.anthropic.enabled
                        and federation_config.anthropic.sync_on_startup
                    )
                    or (federation_config.asor.enabled and federation_config.asor.sync_on_startup)
                    or (
                        federation_config.aws_registry.enabled
                        and federation_config.aws_registry.sync_on_startup
                    )
                )

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

                                    # Ensure UUID id field exists for federation sync
                                    if "id" not in server_data or not server_data["id"]:
                                        server_data["id"] = str(uuid4())

                                    # Register or update server
                                    success = await server_service.register_server(server_data)
                                    if not success:
                                        # Ensure UUID exists before updating (for servers registered before UUID feature)
                                        if "id" not in server_data or not server_data["id"]:
                                            server_data["id"] = str(uuid4())
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
                                from registry.repositories.factory import (
                                    get_server_repository,
                                )
                                from registry.services.federation_reconciliation import (
                                    reconcile_anthropic_servers,
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

                        # AgentCore sync (AWS Agent Registry)
                        if (
                            federation_config.aws_registry.enabled
                            and federation_config.aws_registry.sync_on_startup
                        ):
                            try:
                                await _sync_agentcore_on_startup(
                                    federation_config=federation_config,
                                    server_service=server_service,
                                )
                            except Exception as e:
                                logger.error(
                                    f"AgentCore federation sync failed (continuing with startup): {e}",
                                    exc_info=True,
                                )

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

        # Start ANS sync scheduler
        if settings.ans_integration_enabled:
            from registry.services.ans_sync_scheduler import get_ans_sync_scheduler

            ans_scheduler = get_ans_sync_scheduler()
            await ans_scheduler.start()
            logger.info("ANS sync scheduler started")

        # Ensure unique index on idp_m2m_clients.client_id for the direct
        # M2M registration API (issue #851). Only when feature is enabled.
        if settings.m2m_direct_registration_enabled:
            try:
                from registry.repositories.documentdb.client import get_documentdb_client
                from registry.services.m2m_management_service import (
                    M2MManagementService,
                )

                m2m_db = await get_documentdb_client()
                await M2MManagementService(m2m_db).ensure_indexes()
            except Exception as e:
                logger.warning(
                    f"Failed to ensure idp_m2m_clients index (continuing with startup): {e}",
                )

        # Initialize built-in demo servers (airegistry-tools)
        # Skipped when DISABLE_AI_REGISTRY_TOOLS_SERVER=true (e.g. GitOps/production deployments)
        if not settings.disable_ai_registry_tools_server:
            logger.info("Initializing demo servers...")
            from registry.services.demo_servers_init import initialize_demo_servers

            await initialize_demo_servers()
        else:
            logger.info(
                "Demo server auto-registration disabled (DISABLE_AI_REGISTRY_TOOLS_SERVER=true)"
            )

        # Verify registration gate connectivity (non-blocking)
        from registry.services.registration_gate_service import verify_gate_connectivity

        try:
            await verify_gate_connectivity()
        except Exception as e:
            logger.warning(
                f"Registration gate connectivity check failed (continuing with startup): {e}"
            )

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

        # Initialize and send anonymous startup telemetry (opt-out: MCP_TELEMETRY_DISABLED=1)
        await initialize_telemetry()
        await send_startup_ping()
        await start_heartbeat_scheduler()

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}", exc_info=True)
        raise

    # Application is ready
    yield

    # Shutdown tasks
    logger.info("🔄 Shutting down MCP Gateway Registry...")
    try:
        # Stop ANS sync scheduler
        if settings.ans_integration_enabled:
            from registry.services.ans_sync_scheduler import get_ans_sync_scheduler

            ans_scheduler = get_ans_sync_scheduler()
            await ans_scheduler.stop()

        # Stop telemetry scheduler
        await stop_heartbeat_scheduler()

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
            "name": "Application Logs",
            "description": "Centralized application log retrieval API. Requires admin permissions.",
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
app.include_router(ans_router, prefix="/api", tags=["ANS Integration"])
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
app.include_router(log_router, prefix="/api", tags=["Application Logs"])
app.include_router(registry_management_router, prefix="/api")

# Register IdP M2M management routers (Okta and Auth0)
app.include_router(okta_m2m_router, prefix="/api", tags=["Okta M2M"])
app.include_router(auth0_m2m_router, prefix="/api", tags=["Auth0 M2M"])

# Direct M2M client registration API (issue #851)
# Does not require IdP Admin API token. Gated by feature flag.
if settings.m2m_direct_registration_enabled:
    app.include_router(m2m_management_router, prefix="/api", tags=["M2M Management"])

# Register Anthropic MCP Registry API (public API for MCP servers only)
app.include_router(registry_router, prefix="/api/registry", tags=["Registry Card"])

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
async def get_current_user(user_context: dict[str, Any] = Depends(nginx_proxied_auth)):
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
_CACHED_INDEX_HTML: str | None = None
_ROOT_PATH: str = os.environ.get("ROOT_PATH", "")


def _build_cached_index_html() -> str | None:
    """Read index.html and inject <base> tag if ROOT_PATH is set.

    Returns:
        Modified HTML string if ROOT_PATH is set, None otherwise.
    """
    if not _ROOT_PATH:
        return None

    index_path = FRONTEND_BUILD_PATH / "index.html"
    if not index_path.exists():
        return None

    with open(index_path) as f:
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
