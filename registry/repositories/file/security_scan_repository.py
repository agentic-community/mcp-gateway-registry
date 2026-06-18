"""
File-based repository for security scan results storage.

Reads security scan results from ~/mcp-gateway/security_scans/*.json files.
"""

import json
import logging
from pathlib import Path
from typing import Any

from ..interfaces import SecurityScanRepositoryBase

logger = logging.getLogger(__name__)


class FileSecurityScanRepository(SecurityScanRepositoryBase):
    """File-based implementation of security scan repository."""

    def __init__(self):
        self._scans: dict[str, dict[str, Any]] = {}
        self._scans_dir = Path.home() / "mcp-gateway" / "security_scans"

    async def load_all(self) -> None:
        """Load all security scan results from disk."""
        logger.info(f"Loading security scans from {self._scans_dir}...")

        if not self._scans_dir.exists():
            logger.info(f"Security scans directory does not exist: {self._scans_dir}")
            self._scans = {}
            return

        try:
            self._scans = {}
            scan_files = list(self._scans_dir.glob("*.json"))
            logger.info(f"Found {len(scan_files)} scan files")

            for scan_file in scan_files:
                try:
                    with open(scan_file) as f:
                        scan_data = json.load(f)

                    if isinstance(scan_data, dict) and "server_path" in scan_data:
                        server_path = scan_data["server_path"]
                        self._scans[server_path] = scan_data
                    else:
                        logger.warning(f"Invalid scan file format: {scan_file}")

                except Exception as e:
                    logger.error(f"Error loading scan file {scan_file}: {e}", exc_info=True)

            logger.info(f"Loaded {len(self._scans)} security scan results")

        except Exception as e:
            logger.error(f"Failed to load security scans: {e}", exc_info=True)
            self._scans = {}

    async def get(
        self,
        server_path: str,
    ) -> dict[str, Any] | None:
        """Get latest security scan result for a server."""
        return self._scans.get(server_path)

    async def list_all(self) -> list[dict[str, Any]]:
        """List all security scan results."""
        return list(self._scans.values())

    async def list_latest(self) -> list[dict[str, Any]]:
        """List the latest security scan per server path.

        The file backend keys ``self._scans`` by path and overwrites on each
        save, so it already holds exactly one (the latest) scan per path —
        list_all() is already collapsed.
        """
        return list(self._scans.values())

    async def create(
        self,
        scan_result: dict[str, Any],
    ) -> bool:
        """Create/update a security scan result."""
        try:
            if "server_path" not in scan_result:
                logger.error("Scan result must contain 'server_path' field")
                return False

            server_path = scan_result["server_path"]
            self._scans[server_path] = scan_result

            self._scans_dir.mkdir(parents=True, exist_ok=True)

            sanitized_path = server_path.lstrip("/").replace("/", "_")
            scan_file = self._scans_dir / f"{sanitized_path}_scan.json"

            with open(scan_file, "w") as f:
                json.dump(scan_result, f, indent=2)

            logger.info(f"Saved security scan for {server_path} to {scan_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save security scan: {e}", exc_info=True)
            return False

    async def get_latest(
        self,
        server_path: str,
    ) -> dict[str, Any] | None:
        """Get latest scan result for a server."""
        return await self.get(server_path)

    async def query_by_status(
        self,
        status: str,
    ) -> list[dict[str, Any]]:
        """Query scan results by status."""
        return [scan for scan in self._scans.values() if scan.get("scan_status") == status]
