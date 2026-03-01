#!/usr/bin/env python3
"""
End-to-End Test Script for Agent Skills API.

This script exercises all Agent Skills related API endpoints using
the RegistryClient and produces a report at the end.

Usage:
    # Run with defaults (localhost, .token)
    uv run python tests/e2e_agent_skills_test.py

    # Run with custom registry URL
    uv run python tests/e2e_agent_skills_test.py --registry-url https://myregistry.com

    # Run with custom token file
    uv run python tests/e2e_agent_skills_test.py --token-file /path/to/token

    # Run with debug output
    uv run python tests/e2e_agent_skills_test.py --debug
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

# Add api directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from registry_client import (
    RegistryClient,
    SkillRegistrationRequest,
    SkillCard,
    SkillListResponse,
    SkillHealthResponse,
    SkillContentResponse,
    SkillSearchResponse,
    SkillToggleResponse,
    SkillRatingResponse,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Test Constants
TEST_SKILL_MD_URL = "https://github.com/anthropics/skills/blob/main/skills/mcp-builder/SKILL.md"
TEST_SKILL_NAME = "e2e-test-mcp-builder"
TEST_SKILL_DESCRIPTION = "E2E Test: Build and configure MCP servers"
TEST_SKILL_TAGS = ["e2e-test", "mcp", "builder", "automation"]


class TestStatus(Enum):
    """Test result status."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class TestResult:
    """Individual test result."""

    name: str
    status: TestStatus
    duration_ms: float
    message: str = ""
    details: Optional[Dict[str, Any]] = None


class AgentSkillsE2ETest:
    """End-to-end test runner for Agent Skills API using RegistryClient."""

    def __init__(
        self,
        registry_url: str,
        token: str,
    ):
        """Initialize the test runner.

        Args:
            registry_url: Base URL of the registry
            token: JWT authentication token
        """
        self.client = RegistryClient(registry_url, token)
        self.results: List[TestResult] = []
        self.skill_path: Optional[str] = None

    def _record_result(
        self,
        name: str,
        status: TestStatus,
        duration_ms: float,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a test result."""
        result = TestResult(
            name=name,
            status=status,
            duration_ms=duration_ms,
            message=message,
            details=details,
        )
        self.results.append(result)

        status_str = f"[{status.value}]"
        logger.info(f"{status_str} {name}: {message} ({duration_ms:.2f}ms)")

    def test_register_skill(self) -> bool:
        """Test registering a new skill."""
        test_name = "Register Skill"
        start_time = time.time()

        try:
            request = SkillRegistrationRequest(
                name=TEST_SKILL_NAME,
                skill_md_url=TEST_SKILL_MD_URL,
                description=TEST_SKILL_DESCRIPTION,
                tags=TEST_SKILL_TAGS,
                visibility="public",
            )

            skill = self.client.register_skill(request)
            duration_ms = (time.time() - start_time) * 1000

            self.skill_path = skill.path
            self._record_result(
                test_name,
                TestStatus.PASSED,
                duration_ms,
                f"Skill registered at {skill.path}",
                {"skill": skill.model_dump()},
            )
            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def test_list_skills(self) -> bool:
        """Test listing skills."""
        test_name = "List Skills"
        start_time = time.time()

        try:
            response = self.client.list_skills()
            duration_ms = (time.time() - start_time) * 1000

            # Check if our test skill is in the list
            skill_names = [s.name for s in response.skills]
            has_test_skill = TEST_SKILL_NAME in skill_names

            if has_test_skill:
                self._record_result(
                    test_name,
                    TestStatus.PASSED,
                    duration_ms,
                    f"Found {len(response.skills)} skills, test skill present",
                    {"total_count": response.total_count},
                )
                return True
            else:
                self._record_result(
                    test_name,
                    TestStatus.FAILED,
                    duration_ms,
                    f"Test skill not found in {len(response.skills)} skills",
                )
                return False

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def test_get_skill(self) -> bool:
        """Test getting skill details."""
        test_name = "Get Skill Details"
        start_time = time.time()

        if not self.skill_path:
            self._record_result(
                test_name,
                TestStatus.SKIPPED,
                0,
                "No skill path available",
            )
            return False

        try:
            skill = self.client.get_skill(self.skill_path)
            duration_ms = (time.time() - start_time) * 1000

            self._record_result(
                test_name,
                TestStatus.PASSED,
                duration_ms,
                f"Retrieved skill: {skill.name}",
                {"skill": skill.model_dump()},
            )
            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def test_update_skill(self) -> bool:
        """Test updating skill."""
        test_name = "Update Skill"
        start_time = time.time()

        if not self.skill_path:
            self._record_result(
                test_name,
                TestStatus.SKIPPED,
                0,
                "No skill path available",
            )
            return False

        try:
            # Note: PUT requires full request body with name and skill_md_url
            request = SkillRegistrationRequest(
                name=TEST_SKILL_NAME,
                skill_md_url=TEST_SKILL_MD_URL,
                description=f"{TEST_SKILL_DESCRIPTION} (updated)",
                tags=TEST_SKILL_TAGS + ["updated"],
            )

            updated = self.client.update_skill(self.skill_path, request)
            duration_ms = (time.time() - start_time) * 1000

            self._record_result(
                test_name,
                TestStatus.PASSED,
                duration_ms,
                "Skill updated successfully",
                {"skill": updated.model_dump()},
            )
            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def test_disable_skill(self) -> bool:
        """Test disabling skill using toggle endpoint."""
        test_name = "Disable Skill"
        start_time = time.time()

        if not self.skill_path:
            self._record_result(
                test_name,
                TestStatus.SKIPPED,
                0,
                "No skill path available",
            )
            return False

        try:
            response = self.client.toggle_skill(self.skill_path, enabled=False)
            duration_ms = (time.time() - start_time) * 1000

            if not response.is_enabled:
                self._record_result(
                    test_name,
                    TestStatus.PASSED,
                    duration_ms,
                    "Skill disabled successfully",
                )
                return True
            else:
                self._record_result(
                    test_name,
                    TestStatus.FAILED,
                    duration_ms,
                    f"Skill still enabled: {response.is_enabled}",
                )
                return False

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def test_enable_skill(self) -> bool:
        """Test enabling skill using toggle endpoint."""
        test_name = "Enable Skill"
        start_time = time.time()

        if not self.skill_path:
            self._record_result(
                test_name,
                TestStatus.SKIPPED,
                0,
                "No skill path available",
            )
            return False

        try:
            response = self.client.toggle_skill(self.skill_path, enabled=True)
            duration_ms = (time.time() - start_time) * 1000

            if response.is_enabled:
                self._record_result(
                    test_name,
                    TestStatus.PASSED,
                    duration_ms,
                    "Skill enabled successfully",
                )
                return True
            else:
                self._record_result(
                    test_name,
                    TestStatus.FAILED,
                    duration_ms,
                    f"Skill still disabled: {response.is_enabled}",
                )
                return False

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def test_health_check(self) -> bool:
        """Test skill health check."""
        test_name = "Health Check"
        start_time = time.time()

        if not self.skill_path:
            self._record_result(
                test_name,
                TestStatus.SKIPPED,
                0,
                "No skill path available",
            )
            return False

        try:
            response = self.client.check_skill_health(self.skill_path)
            duration_ms = (time.time() - start_time) * 1000

            if response.healthy:
                self._record_result(
                    test_name,
                    TestStatus.PASSED,
                    duration_ms,
                    "SKILL.md is accessible",
                    {"status_code": response.status_code},
                )
                return True
            else:
                self._record_result(
                    test_name,
                    TestStatus.FAILED,
                    duration_ms,
                    f"SKILL.md not accessible: {response.error}",
                )
                return False

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def test_get_content(self) -> bool:
        """Test getting SKILL.md content."""
        test_name = "Get SKILL.md Content"
        start_time = time.time()

        if not self.skill_path:
            self._record_result(
                test_name,
                TestStatus.SKIPPED,
                0,
                "No skill path available",
            )
            return False

        try:
            response = self.client.get_skill_content(self.skill_path)
            duration_ms = (time.time() - start_time) * 1000

            content_len = len(response.content)
            if content_len > 0:
                self._record_result(
                    test_name,
                    TestStatus.PASSED,
                    duration_ms,
                    f"Retrieved {content_len} characters of content",
                )
                return True
            else:
                self._record_result(
                    test_name,
                    TestStatus.FAILED,
                    duration_ms,
                    "Empty content returned",
                )
                return False

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def test_rate_skill(self) -> bool:
        """Test rating a skill."""
        test_name = "Rate Skill"
        start_time = time.time()

        if not self.skill_path:
            self._record_result(
                test_name,
                TestStatus.SKIPPED,
                0,
                "No skill path available",
            )
            return False

        try:
            response = self.client.rate_skill(self.skill_path, rating=5)
            duration_ms = (time.time() - start_time) * 1000

            avg_rating = response.get("average_rating", 0)
            self._record_result(
                test_name,
                TestStatus.PASSED,
                duration_ms,
                f"Rated 5 stars, average: {avg_rating}",
            )
            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def test_get_rating(self) -> bool:
        """Test getting skill rating."""
        test_name = "Get Rating"
        start_time = time.time()

        if not self.skill_path:
            self._record_result(
                test_name,
                TestStatus.SKIPPED,
                0,
                "No skill path available",
            )
            return False

        try:
            response = self.client.get_skill_rating(self.skill_path)
            duration_ms = (time.time() - start_time) * 1000

            self._record_result(
                test_name,
                TestStatus.PASSED,
                duration_ms,
                f"Rating: {response.num_stars} stars",
            )
            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def test_search_skills(self) -> bool:
        """Test searching for skills."""
        test_name = "Search Skills"
        start_time = time.time()

        try:
            response = self.client.search_skills(query="mcp builder")
            duration_ms = (time.time() - start_time) * 1000

            if response.total_count > 0:
                self._record_result(
                    test_name,
                    TestStatus.PASSED,
                    duration_ms,
                    f"Found {response.total_count} matching skills",
                )
                return True
            else:
                self._record_result(
                    test_name,
                    TestStatus.FAILED,
                    duration_ms,
                    "No matching skills found",
                )
                return False

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def test_delete_skill(self) -> bool:
        """Test deleting skill (cleanup)."""
        test_name = "Delete Skill (Cleanup)"
        start_time = time.time()

        if not self.skill_path:
            self._record_result(
                test_name,
                TestStatus.SKIPPED,
                0,
                "No skill path available",
            )
            return False

        try:
            self.client.delete_skill(self.skill_path)
            duration_ms = (time.time() - start_time) * 1000

            self._record_result(
                test_name,
                TestStatus.PASSED,
                duration_ms,
                "Skill deleted successfully",
            )
            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_result(
                test_name,
                TestStatus.FAILED,
                duration_ms,
                f"Exception: {str(e)}",
            )
            return False

    def run_all_tests(self) -> bool:
        """Run all tests in sequence."""
        logger.info("=" * 60)
        logger.info("Starting Agent Skills E2E Tests")
        logger.info(f"Registry URL: {self.client.registry_url}")
        logger.info(f"Test Skill URL: {TEST_SKILL_MD_URL}")
        logger.info("=" * 60)

        # Run tests in order
        self.test_register_skill()
        self.test_list_skills()
        self.test_get_skill()
        self.test_update_skill()
        self.test_disable_skill()
        self.test_enable_skill()
        self.test_health_check()
        self.test_get_content()
        self.test_rate_skill()
        self.test_get_rating()
        self.test_search_skills()
        self.test_delete_skill()

        return self._print_report()

    def _print_report(self) -> bool:
        """Print test report and return success status."""
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        total_time = sum(r.duration_ms for r in self.results)

        print("\n")
        print("=" * 70)
        print("                    AGENT SKILLS E2E TEST REPORT")
        print("=" * 70)
        print(f"  Registry URL: {self.client.registry_url}")
        print(f"  Test Run:     {datetime.now().isoformat()}")
        print("=" * 70)
        print("\n  TEST RESULTS:")
        print("  " + "-" * 66)

        for result in self.results:
            if result.status == TestStatus.PASSED:
                status_color = "\033[92m"  # Green
            elif result.status == TestStatus.FAILED:
                status_color = "\033[91m"  # Red
            else:
                status_color = "\033[93m"  # Yellow

            reset_color = "\033[0m"
            status_str = f"{status_color}[{result.status.value}]{reset_color}"

            print(f"  {status_str} {result.name:35} {result.duration_ms:>10.2f}ms")
            if result.message:
                print(f"       {result.message}")

        print("  " + "-" * 66)
        print("\n  SUMMARY:")
        print(f"    Total Tests:  {len(self.results)}")
        print(f"    \033[92mPassed:\033[0m       {passed}")
        print(f"    \033[91mFailed:\033[0m       {failed}")
        print(f"    \033[93mSkipped:\033[0m      {skipped}")
        print(f"    Total Time:   {total_time:.2f}ms ({total_time / 1000:.2f}s)")

        if failed > 0:
            print(f"\n  \033[91m*** {failed} TEST(S) FAILED ***\033[0m")
        else:
            print(f"\n  \033[92m*** ALL TESTS PASSED ***\033[0m")

        print("=" * 70)
        print()

        return failed == 0


def _load_token(
    token_file: str,
) -> str:
    """Load JWT token from file.

    Args:
        token_file: Path to token file

    Returns:
        JWT token string

    Raises:
        FileNotFoundError: If token file not found
        ValueError: If token file is empty or invalid
    """
    token_path = Path(token_file)

    if not token_path.exists():
        raise FileNotFoundError(f"Token file not found: {token_file}")

    content = token_path.read_text().strip()

    if not content:
        raise ValueError(f"Token file is empty: {token_file}")

    # Handle JSON token files (like ingress.json or .token)
    if content.startswith("{"):
        try:
            data = json.loads(content)
            # Try different possible token field names at top level
            for key in ["access_token", "token", "jwt"]:
                if key in data:
                    return data[key]
            # Check for nested tokens object (common format from auth endpoints)
            if "tokens" in data and isinstance(data["tokens"], dict):
                tokens = data["tokens"]
                for key in ["access_token", "token", "jwt"]:
                    if key in tokens:
                        return tokens[key]
            raise ValueError(f"No token field found in JSON file: {token_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in token file: {e}") from e

    # Plain text token
    return content


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="End-to-End Test Script for Agent Skills API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with defaults
    uv run python tests/e2e_agent_skills_test.py

    # Run with custom registry
    uv run python tests/e2e_agent_skills_test.py --registry-url https://myregistry.com

    # Run with debug output
    uv run python tests/e2e_agent_skills_test.py --debug
""",
    )

    parser.add_argument(
        "--registry-url",
        default="http://localhost",
        help="Registry base URL (default: http://localhost)",
    )
    parser.add_argument(
        "--token-file",
        default=".token",
        help="Path to token file (default: .token)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        token = _load_token(args.token_file)
        logger.info(f"Loaded token from {args.token_file}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"ERROR: {e}")
        return 1

    test_runner = AgentSkillsE2ETest(
        registry_url=args.registry_url,
        token=token,
    )

    success = test_runner.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
