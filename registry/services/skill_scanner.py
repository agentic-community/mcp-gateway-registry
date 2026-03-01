"""
Skill Scanner Service

Wraps the Cisco AI Defense skill-scanner CLI tool for security scanning
of AI agent skills during registration.
"""

import asyncio
import json
import logging
import os
import re
import subprocess  # nosec B404
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ..core.config import settings
from ..repositories.factory import get_skill_security_scan_repository
from ..schemas.skill_security import SkillSecurityScanConfig, SkillSecurityScanResult

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent.parent / "skill_security_scans"


class SkillScannerService:
    """Service for scanning skills for security vulnerabilities."""

    def __init__(self) -> None:
        """Initialize the skill scanner service."""
        self._ensure_output_directory()
        self._scan_repo = None

    @property
    def scan_repo(self):
        """Lazy-load the scan repository."""
        if self._scan_repo is None:
            self._scan_repo = get_skill_security_scan_repository()
        return self._scan_repo

    def _ensure_output_directory(self) -> Path:
        """Ensure output directory exists."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR

    def get_scan_config(self) -> SkillSecurityScanConfig:
        """Get skill security scan configuration from settings."""
        return SkillSecurityScanConfig(
            enabled=settings.skill_security_scan_enabled,
            scan_on_registration=settings.skill_security_scan_on_registration,
            block_unsafe_skills=settings.skill_security_block_unsafe_skills,
            analyzers=settings.skill_security_analyzers,
            scan_timeout_seconds=settings.skill_security_scan_timeout,
            llm_api_key=settings.skill_scanner_llm_api_key
            or os.getenv("SKILL_SCANNER_LLM_API_KEY"),
            virustotal_api_key=settings.skill_scanner_virustotal_api_key
            or os.getenv("VIRUSTOTAL_API_KEY"),
            ai_defense_api_key=settings.skill_scanner_ai_defense_api_key
            or os.getenv("AI_DEFENSE_API_KEY"),
            add_security_pending_tag=settings.skill_security_add_pending_tag,
        )

    async def scan_skill(
        self,
        skill_path: str,
        skill_md_url: str | None = None,
        skill_content_path: str | None = None,
        analyzers: str | None = None,
        timeout: int | None = None,
    ) -> SkillSecurityScanResult:
        """
        Scan a skill for security vulnerabilities.

        Args:
            skill_path: Registry path of the skill (e.g., /skills/pdf-processing)
            skill_md_url: URL to SKILL.md file (for remote scanning)
            skill_content_path: Local path to skill content (for local scanning)
            analyzers: Comma-separated list of analyzers
            timeout: Scan timeout in seconds

        Returns:
            SkillSecurityScanResult containing scan results
        """
        config = self.get_scan_config()

        if analyzers is None:
            analyzers = config.analyzers
        if timeout is None:
            timeout = config.scan_timeout_seconds

        logger.info(f"Starting skill security scan for {skill_path} with analyzers: {analyzers}")

        try:
            raw_output = await asyncio.to_thread(
                self._run_skill_scanner,
                skill_path=skill_path,
                skill_md_url=skill_md_url,
                skill_content_path=skill_content_path,
                analyzers=analyzers,
                timeout=timeout,
            )

            is_safe, critical, high, medium, low = self._analyze_scan_results(raw_output)

            result = SkillSecurityScanResult(
                skill_path=skill_path,
                skill_md_url=skill_md_url,
                scan_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                is_safe=is_safe,
                critical_issues=critical,
                high_severity=high,
                medium_severity=medium,
                low_severity=low,
                analyzers_used=analyzers.split(","),
                raw_output=raw_output,
                scan_failed=False,
            )

            await self.scan_repo.create(result.model_dump())

            logger.info(
                f"Skill security scan completed for {skill_path}. "
                f"Safe: {is_safe}, Critical: {critical}, High: {high}"
            )

            return result

        except Exception as e:
            logger.error(f"Skill security scan failed for {skill_path}: {e}")

            result = SkillSecurityScanResult(
                skill_path=skill_path,
                skill_md_url=skill_md_url,
                scan_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                is_safe=False,
                analyzers_used=analyzers.split(",") if analyzers else [],
                raw_output={"error": str(e), "scan_failed": True},
                scan_failed=True,
                error_message=str(e),
            )

            await self.scan_repo.create(result.model_dump())
            return result

    def _run_skill_scanner(
        self,
        skill_path: str,
        skill_md_url: str | None = None,
        skill_content_path: str | None = None,
        analyzers: str = "static",
        timeout: int = 120,
    ) -> dict:
        """
        Run skill-scanner command and return raw output.

        This is a synchronous method that runs in a thread pool.

        Args:
            skill_path: Registry path of the skill
            skill_md_url: URL to SKILL.md file
            skill_content_path: Local path to skill content
            analyzers: Comma-separated list of analyzers
            timeout: Scan timeout in seconds

        Returns:
            Dict containing parsed scan results

        Raises:
            RuntimeError: If scan times out or CLI returns non-zero exit code
            ValueError: If neither skill_content_path nor skill_md_url is provided
        """
        logger.info(f"Running skill security scan on: {skill_path}")

        # Determine scan target
        if skill_content_path:
            target = skill_content_path
        elif skill_md_url:
            target = self._download_skill_content(skill_md_url)
        else:
            raise ValueError("Either skill_content_path or skill_md_url must be provided")

        try:
            cmd = [
                "skill-scanner",
                "scan",
                target,
                "--format",
                "json",
            ]

            # Add optional analyzer flags based on config
            config = self.get_scan_config()
            analyzer_list = [a.strip() for a in analyzers.split(",")]
            if "behavioral" in analyzer_list:
                cmd.append("--use-behavioral")
            if "llm" in analyzer_list:
                cmd.append("--use-llm")
            if "virustotal" in analyzer_list:
                cmd.append("--use-virustotal")
            if "ai-defense" in analyzer_list:
                cmd.append("--use-aidefense")

            # Set environment variables for API keys
            env = os.environ.copy()
            if config.llm_api_key:
                env["LLM_API_KEY"] = config.llm_api_key
            if config.virustotal_api_key:
                env["VIRUSTOTAL_API_KEY"] = config.virustotal_api_key
            if config.ai_defense_api_key:
                env["AI_DEFENSE_API_KEY"] = config.ai_defense_api_key

            result = subprocess.run(  # nosec B603 - args are hardcoded flags and validated config values
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                timeout=timeout,
            )

            return self._parse_scanner_output(result.stdout)

        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Skill scan timed out after {timeout} seconds") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Skill scanner failed: {e.stderr}") from e

    def _download_skill_content(self, skill_md_url: str) -> str:
        """
        Download skill content for scanning.

        Args:
            skill_md_url: URL to SKILL.md file

        Returns:
            Path to temporary directory containing downloaded skill content

        Raises:
            httpx.HTTPError: If download fails
        """
        import httpx

        temp_dir = tempfile.mkdtemp(prefix="skill_scan_")
        skill_md_path = Path(temp_dir) / "SKILL.md"

        response = httpx.get(skill_md_url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        skill_md_path.write_text(response.text)

        return temp_dir

    def _parse_scanner_output(self, stdout: str) -> dict:
        """
        Parse JSON output from skill-scanner CLI.

        Strips ANSI escape codes, parses JSON, and organizes findings by analyzer.

        Args:
            stdout: Raw stdout from skill-scanner CLI

        Returns:
            Dict with analysis_results organized by analyzer and raw scan_results
        """
        # Remove ANSI codes
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        clean_stdout = ansi_escape.sub("", stdout.strip())

        # Parse JSON
        scan_results = json.loads(clean_stdout)

        # Organize into standard format
        raw_output = {
            "analysis_results": {},
            "scan_results": scan_results,
        }

        # Extract findings by analyzer
        findings = scan_results.get("findings", [])
        for finding in findings:
            analyzer = finding.get("analyzer", "unknown")
            if analyzer not in raw_output["analysis_results"]:
                raw_output["analysis_results"][analyzer] = {"findings": []}
            raw_output["analysis_results"][analyzer]["findings"].append(finding)

        return raw_output

    def _analyze_scan_results(self, raw_output: dict) -> tuple[bool, int, int, int, int]:
        """
        Analyze scan results and extract severity counts.

        Args:
            raw_output: Parsed scanner output dict

        Returns:
            Tuple of (is_safe, critical_count, high_count, medium_count, low_count)
        """
        critical = high = medium = low = 0

        analysis_results = raw_output.get("analysis_results", {})
        for analyzer_data in analysis_results.values():
            if isinstance(analyzer_data, dict):
                for finding in analyzer_data.get("findings", []):
                    severity = finding.get("severity", "").lower()
                    if severity == "critical":
                        critical += 1
                    elif severity == "high":
                        high += 1
                    elif severity == "medium":
                        medium += 1
                    elif severity == "low":
                        low += 1

        is_safe = critical == 0 and high == 0

        logger.info(
            f"Skill security analysis: Critical={critical}, High={high}, Medium={medium}, Low={low}"
        )
        return is_safe, critical, high, medium, low

    async def get_scan_result(self, skill_path: str) -> dict | None:
        """
        Get the latest scan result for a skill.

        Args:
            skill_path: Skill path (e.g., /skills/pdf-processing)

        Returns:
            Dictionary containing scan results, or None if no scan found
        """
        try:
            scan_result = await self.scan_repo.get_latest(skill_path)
            if scan_result:
                if hasattr(scan_result, "model_dump"):
                    return scan_result.model_dump()
                return scan_result
            return None
        except Exception:
            logger.exception(f"Error loading skill scan results for {skill_path}")
            return None


# Global singleton instance
skill_scanner_service = SkillScannerService()
