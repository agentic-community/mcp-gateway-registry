"""
Service for validating tool availability for skills.

Links allowed_tools to MCP servers in the registry.
"""

import logging
import time
from typing import (
    List,
    Optional,
    Set,
)

from ..repositories.factory import get_server_repository
from ..repositories.interfaces import ServerRepositoryBase
from ..schemas.skill_models import (
    SkillCard,
    ToolReference,
    ToolValidationResult,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


class ToolValidationService:
    """Validate tool availability for skill execution."""

    def __init__(self):
        self._server_repo: Optional[ServerRepositoryBase] = None

    def _get_server_repo(self) -> ServerRepositoryBase:
        """Lazy initialization of server repository."""
        if self._server_repo is None:
            self._server_repo = get_server_repository()
        return self._server_repo

    async def validate_tools_available(
        self,
        skill: SkillCard,
        enabled_only: bool = True,
    ) -> ToolValidationResult:
        """Check if required tools are available.

        Args:
            skill: SkillCard with allowed_tools
            enabled_only: Only check enabled servers

        Returns:
            ToolValidationResult with availability status
        """
        start_time = time.perf_counter()
        skill_path = skill.path if skill.path else "unknown"
        required_tool_count = len(skill.allowed_tools) if skill.allowed_tools else 0

        logger.info(
            f"Starting tool validation for skill '{skill_path}' "
            f"with {required_tool_count} required tools"
        )

        if not skill.allowed_tools:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Tool validation completed for skill '{skill_path}': "
                f"all_available=True, found=0, missing=0, duration={elapsed_ms:.2f}ms"
            )
            return ToolValidationResult(
                all_available=True, missing_tools=[], available_tools=[], mcp_servers_required=[]
            )

        # Get all servers
        server_repo = self._get_server_repo()
        servers_dict = await server_repo.list_all()

        logger.debug(f"Retrieved {len(servers_dict)} servers from repository")

        # Build index of available tools
        available_tools: Set[str] = set()
        server_tool_map: dict = {}

        for server_path, server_info in servers_dict.items():
            # Check if server is enabled
            if enabled_only:
                is_enabled = await server_repo.get_state(server_path)
                if not is_enabled:
                    logger.debug(f"Skipping disabled server: {server_path}")
                    continue

            tool_list = server_info.get("tool_list", [])
            tool_names_in_server = []

            for tool in tool_list:
                tool_name = tool.get("name", "")
                if tool_name:
                    available_tools.add(tool_name)
                    tool_names_in_server.append(tool_name)
                    if tool_name not in server_tool_map:
                        server_tool_map[tool_name] = []
                    server_tool_map[tool_name].append(server_path)

            logger.debug(
                f"Server '{server_path}' provides {len(tool_names_in_server)} tools: "
                f"{tool_names_in_server}"
            )

        # Check each required tool
        missing: List[str] = []
        found: List[str] = []
        required_servers: Set[str] = set()

        for tool_ref in skill.allowed_tools:
            tool_name = tool_ref.tool_name

            if tool_name in available_tools:
                found.append(tool_name)
                required_servers.update(server_tool_map.get(tool_name, []))
                logger.debug(f"Tool '{tool_name}' is available")
            else:
                missing.append(tool_name)
                logger.debug(f"Tool '{tool_name}' is NOT available")

        # Log warnings for missing tools
        if missing:
            logger.warning(f"Skill '{skill_path}' has {len(missing)} missing tools: {missing}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        all_available = len(missing) == 0

        logger.info(
            f"Tool validation completed for skill '{skill_path}': "
            f"all_available={all_available}, found={len(found)}, "
            f"missing={len(missing)}, duration={elapsed_ms:.2f}ms"
        )

        return ToolValidationResult(
            all_available=all_available,
            missing_tools=missing,
            available_tools=found,
            mcp_servers_required=list(required_servers),
        )

    async def get_tools_with_servers(
        self,
        tool_refs: List[ToolReference],
    ) -> List[dict]:
        """Get tool references with their providing servers.

        Args:
            tool_refs: List of ToolReference objects

        Returns:
            List of dicts with tool info and server paths
        """
        start_time = time.perf_counter()
        tool_count = len(tool_refs)

        logger.info(f"Looking up servers for {tool_count} tool references")

        server_repo = self._get_server_repo()
        servers_dict = await server_repo.list_all()

        logger.debug(f"Retrieved {len(servers_dict)} servers from repository")

        result = []
        for tool_ref in tool_refs:
            tool_info = {
                "tool_name": tool_ref.tool_name,
                "capabilities": tool_ref.capabilities,
                "servers": [],
            }

            for server_path, server_info in servers_dict.items():
                is_enabled = await server_repo.get_state(server_path)
                tool_list = server_info.get("tool_list", [])

                for tool in tool_list:
                    if tool.get("name") == tool_ref.tool_name:
                        tool_info["servers"].append(
                            {
                                "path": server_path,
                                "name": server_info.get("server_name", ""),
                                "is_enabled": is_enabled,
                            }
                        )
                        logger.debug(
                            f"Tool '{tool_ref.tool_name}' found on server "
                            f"'{server_path}' (enabled={is_enabled})"
                        )
                        break

            if not tool_info["servers"]:
                logger.warning(f"Tool '{tool_ref.tool_name}' not found on any server")

            result.append(tool_info)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        tools_with_servers = sum(1 for t in result if t["servers"])

        logger.info(
            f"Server lookup completed: {tools_with_servers}/{tool_count} tools "
            f"have servers, duration={elapsed_ms:.2f}ms"
        )

        return result


# Singleton
_tool_validation_service: Optional[ToolValidationService] = None


def get_tool_validation_service() -> ToolValidationService:
    """Get or create tool validation service singleton."""
    global _tool_validation_service
    if _tool_validation_service is None:
        _tool_validation_service = ToolValidationService()
    return _tool_validation_service
