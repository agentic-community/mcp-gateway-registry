"""
File-based repository for skill security scan results storage.

Reads skill security scan results from ~/mcp-gateway/skill_security_scans/*.json files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..interfaces import SkillSecurityScanRepositoryBase

logger = logging.getLogger(__name__)


class FileSkillSecurityScanRepository(SkillSecurityScanRepositoryBase):
    """File-based implementation of skill security scan repository."""

    def __init__(self):
        self._scans: Dict[str, Dict[str, Any]] = {}
        self._scans_dir = Path.home() / "mcp-gateway" / "skill_security_scans"

    async def load_all(self) -> None:
        """Load all skill security scan results from disk."""
        logger.info(f"Loading skill security scans from {self._scans_dir}...")

        if not self._scans_dir.exists():
            logger.info(f"Skill security scans directory does not exist: {self._scans_dir}")
            self._scans = {}
            return

        try:
            self._scans = {}
            scan_files = list(self._scans_dir.glob("*.json"))
            logger.info(f"Found {len(scan_files)} skill scan files")

            for scan_file in scan_files:
                try:
                    with open(scan_file, "r") as f:
                        scan_data = json.load(f)

                    if isinstance(scan_data, dict) and "skill_path" in scan_data:
                        skill_path = scan_data["skill_path"]
                        self._scans[skill_path] = scan_data
                    else:
                        logger.warning(f"Invalid skill scan file format: {scan_file}")

                except Exception as e:
                    logger.error(f"Error loading skill scan file {scan_file}: {e}", exc_info=True)

            logger.info(f"Loaded {len(self._scans)} skill security scan results")

        except Exception as e:
            logger.error(f"Failed to load skill security scans: {e}", exc_info=True)
            self._scans = {}

    async def get(
        self,
        skill_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Get latest security scan result for a skill."""
        return self._scans.get(skill_path)

    async def list_all(self) -> List[Dict[str, Any]]:
        """List all skill security scan results."""
        return list(self._scans.values())

    async def create(
        self,
        scan_result: Dict[str, Any],
    ) -> bool:
        """Create/update a skill security scan result."""
        try:
            if "skill_path" not in scan_result:
                logger.error("Scan result must contain 'skill_path' field")
                return False

            skill_path = scan_result["skill_path"]
            self._scans[skill_path] = scan_result

            self._scans_dir.mkdir(parents=True, exist_ok=True)

            sanitized_path = skill_path.lstrip("/").replace("/", "_")
            scan_file = self._scans_dir / f"{sanitized_path}_scan.json"

            with open(scan_file, "w") as f:
                json.dump(scan_result, f, indent=2)

            logger.info(f"Saved skill security scan for {skill_path} to {scan_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save skill security scan: {e}", exc_info=True)
            return False

    async def get_latest(
        self,
        skill_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Get latest scan result for a skill."""
        return await self.get(skill_path)

    async def query_by_status(
        self,
        status: str,
    ) -> List[Dict[str, Any]]:
        """Query scan results by status."""
        return [scan for scan in self._scans.values() if scan.get("scan_status") == status]
