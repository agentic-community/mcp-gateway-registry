"""
File-based repository for MCP server storage.

Extracts all file I/O logic from ServerService while maintaining identical behavior.
"""

import json
import logging
from typing import Any

from ...core.config import settings
from ..interfaces import ServerRepositoryBase

logger = logging.getLogger(__name__)


class FileServerRepository(ServerRepositoryBase):
    """File-based implementation of server repository."""

    def __init__(self):
        self._servers: dict[str, dict[str, Any]] = {}
        self._state: dict[str, bool] = {}

    async def load_all(self) -> None:
        """Load server definitions and state from disk."""
        logger.info(f"Loading server definitions from {settings.servers_dir}...")

        settings.servers_dir.mkdir(parents=True, exist_ok=True)

        temp_servers = {}
        server_files = list(settings.servers_dir.glob("**/*.json"))
        logger.info(f"Found {len(server_files)} JSON files")

        for server_file in server_files:
            if server_file.name == settings.state_file_path.name:
                continue

            try:
                with open(server_file) as f:
                    server_info = json.load(f)

                    if (
                        isinstance(server_info, dict)
                        and "path" in server_info
                        and "server_name" in server_info
                    ):
                        server_path = server_info["path"]
                        if server_path in temp_servers:
                            logger.warning(f"Duplicate server path in {server_file}: {server_path}")

                        server_info.setdefault("description", "")
                        server_info.setdefault("tags", [])
                        server_info.setdefault("num_tools", 0)
                        server_info.setdefault("license", "N/A")
                        server_info.setdefault("proxy_pass_url", None)
                        server_info.setdefault("tool_list", [])

                        temp_servers[server_path] = server_info
                    else:
                        logger.warning(f"Invalid server entry in {server_file}")
            except Exception as e:
                logger.error(f"Error loading {server_file}: {e}", exc_info=True)

        self._servers = temp_servers
        logger.info(f"Loaded {len(self._servers)} server definitions")

        await self._load_state()

    async def _load_state(self) -> None:
        """Load persisted service state from disk."""
        logger.info(f"Loading state from {settings.state_file_path}...")
        loaded_state = {}

        try:
            if settings.state_file_path.exists():
                with open(settings.state_file_path) as f:
                    loaded_state = json.load(f)
                if not isinstance(loaded_state, dict):
                    logger.warning("Invalid state format, resetting")
                    loaded_state = {}
                else:
                    logger.info("Successfully loaded persisted state")
            else:
                logger.info("No state file found, initializing")
        except Exception as e:
            logger.error(f"Failed to read state file: {e}", exc_info=True)
            loaded_state = {}

        self._state = {}
        for path in self._servers.keys():
            value = loaded_state.get(path)
            if value is None:
                if path.endswith("/"):
                    value = loaded_state.get(path.rstrip("/"), False)
                else:
                    value = loaded_state.get(path + "/", False)
            self._state[path] = value

        logger.info(f"Initial service state loaded: {self._state}")

    async def _save_state(self) -> None:
        """Persist service state to disk."""
        try:
            with open(settings.state_file_path, "w") as f:
                json.dump(self._state, f, indent=2)
            logger.info(f"Persisted state to {settings.state_file_path}")
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")

    def _path_to_filename(
        self,
        path: str,
    ) -> str:
        """Convert path to safe filename."""
        normalized = path.lstrip("/").replace("/", "_")
        if not normalized.endswith(".json"):
            normalized += ".json"
        return normalized

    async def _save_to_file(
        self,
        server_info: dict[str, Any],
    ) -> bool:
        """Save server data to individual file."""
        try:
            settings.servers_dir.mkdir(parents=True, exist_ok=True)

            path = server_info["path"]
            filename = self._path_to_filename(path)
            file_path = settings.servers_dir / filename

            with open(file_path, "w") as f:
                json.dump(server_info, f, indent=2)

            logger.info(f"Saved server '{server_info['server_name']}' to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save server: {e}", exc_info=True)
            return False

    async def get(
        self,
        path: str,
    ) -> dict[str, Any] | None:
        """Get server by path."""
        server_info = self._servers.get(path)
        if server_info:
            return server_info

        if path.endswith("/"):
            alternate_path = path.rstrip("/")
        else:
            alternate_path = path + "/"

        return self._servers.get(alternate_path)

    async def list_all(self) -> dict[str, dict[str, Any]]:
        """List all servers."""
        return self._servers.copy()

    async def list_by_source(
        self,
        source: str,
    ) -> dict[str, dict[str, Any]]:
        """List all servers from a specific federation source.

        Args:
            source: Federation source identifier (e.g., "anthropic")

        Returns:
            Dictionary mapping server path to server info
        """
        return {
            path: info
            for path, info in self._servers.items()
            if info.get("source") == source
        }

    async def create(
        self,
        server_info: dict[str, Any],
    ) -> bool:
        """Create a new server."""
        path = server_info["path"]

        if path in self._servers:
            logger.error(f"Server path '{path}' already exists")
            return False

        if not await self._save_to_file(server_info):
            return False

        self._servers[path] = server_info
        self._state[path] = False

        await self._save_state()

        logger.info(f"New server registered: '{server_info['server_name']}' at '{path}'")
        return True

    async def update(
        self,
        path: str,
        server_info: dict[str, Any],
    ) -> bool:
        """Update an existing server."""
        if path not in self._servers:
            logger.error(f"Cannot update server at '{path}': not found")
            return False

        server_info["path"] = path

        if not await self._save_to_file(server_info):
            return False

        self._servers[path] = server_info

        logger.info(f"Server '{server_info['server_name']}' ({path}) updated")
        return True

    async def delete(
        self,
        path: str,
    ) -> bool:
        """Delete a server."""
        if path not in self._servers:
            logger.error(f"Cannot delete server at '{path}': not found")
            return False

        try:
            filename = self._path_to_filename(path)
            file_path = settings.servers_dir / filename

            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed server file: {file_path}")
            else:
                logger.warning(f"Server file not found: {file_path}")

            server_name = self._servers[path].get("server_name", "Unknown")
            del self._servers[path]

            if path in self._state:
                del self._state[path]

            await self._save_state()

            logger.info(f"Successfully removed server '{server_name}' from '{path}'")
            return True

        except Exception as e:
            logger.error(f"Failed to remove server at '{path}': {e}", exc_info=True)
            return False

    async def delete_with_versions(
        self,
        path: str,
    ) -> int:
        """Delete a server and all its version documents.

        Deletes the active document at `path` and any version documents
        with keys matching `{path}:{version}`.

        Args:
            path: Server base path (e.g., "/context7")

        Returns:
            Number of documents deleted (0 if none found)
        """
        deleted_count = 0

        # Find all keys that match: exact path or path:version pattern
        version_prefix = f"{path}:"
        keys_to_delete = []
        for key in list(self._servers.keys()):
            if key == path or key.startswith(version_prefix):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            # Remove the server file from disk
            filename = self._path_to_filename(key)
            file_path = settings.servers_dir / filename
            if file_path.exists():
                file_path.unlink()
                logger.info("Removed server file: %s", file_path)

            # Remove from in-memory dicts
            del self._servers[key]
            if key in self._state:
                del self._state[key]
            deleted_count += 1

        if deleted_count > 0:
            await self._save_state()
            logger.info(
                "delete_with_versions: removed %d document(s) for path '%s'",
                deleted_count,
                path,
            )

        return deleted_count

    async def get_state(
        self,
        path: str,
    ) -> bool:
        """Get server enabled/disabled state."""
        result = self._state.get(path)

        if result is None:
            if path.endswith("/"):
                result = self._state.get(path.rstrip("/"), False)
            else:
                result = self._state.get(path + "/", False)

        if result is None:
            result = False

        return result

    async def set_state(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Set server enabled/disabled state."""
        if path not in self._servers:
            logger.error(f"Cannot toggle service at '{path}': not found")
            return False

        self._state[path] = enabled
        await self._save_state()

        server_name = self._servers[path]["server_name"]
        logger.info(f"Toggled '{server_name}' ({path}) to {enabled}")

        return True

    async def count(self) -> int:
        """Get total count of servers.

        Returns:
            Total number of servers in the repository.
        """
        return len(self._servers)

    async def update_field(
        self,
        path: str,
        field: str,
        value: Any,
    ) -> bool:
        """Update a single field on a document (file-based)."""
        data = await self.get(path)
        if not data:
            return False

        if value is None:
            parts = field.split(".")
            obj = data
            for part in parts[:-1]:
                obj = obj.get(part, {})
            obj.pop(parts[-1], None)
        else:
            parts = field.split(".")
            obj = data
            for part in parts[:-1]:
                if part not in obj:
                    obj[part] = {}
                obj = obj[part]
            obj[parts[-1]] = value

        await self._save_to_file(data)
        self._servers[path] = data
        return True

    async def find_with_filter(
        self,
        filter_dict: dict[str, Any],
    ) -> dict[str, dict]:
        """Find documents matching a filter (file-based, basic support)."""
        all_data = await self.list_all()
        results = {}
        for path, data in all_data.items():
            match = True
            for field_name, condition in filter_dict.items():
                if isinstance(condition, dict) and "$exists" in condition:
                    has_field = data.get(field_name) is not None
                    if condition.get("$ne") is not None:
                        has_field = has_field and data.get(field_name) != condition["$ne"]
                    if condition["$exists"] != has_field:
                        match = False
                        break
            if match:
                results[path] = data
        return results
