#!/usr/bin/env python3
"""
Test runner script for MCP Registry.

This script provides a unified interface for running tests with various configurations,
checking dependencies, and generating reports.
"""

import argparse
import logging
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import List, Optional


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


# Required test dependencies
# Note: These are the actual Python import names, not package names
REQUIRED_DEPENDENCIES = [
    "pytest",
    "pytest_asyncio",
    "pytest_cov",
    "pytest_mock",
    "xdist",  # pytest-xdist package
    "pytest_html",
    "pytest_jsonreport",  # pytest-json-report package
    "factory",  # factory-boy package
    "faker",
    "freezegun",
    "itsdangerous",
]


def _print_colored(message: str, color: str = Colors.RESET) -> None:
    """Print a colored message to stdout.

    Args:
        message: The message to print
        color: ANSI color code
    """
    print(f"{color}{message}{Colors.RESET}")


def _print_header(message: str) -> None:
    """Print a section header.

    Args:
        message: The header message
    """
    _print_colored(f"\n{'=' * 70}", Colors.CYAN)
    _print_colored(f"{message}", Colors.CYAN + Colors.BOLD)
    _print_colored(f"{'=' * 70}\n", Colors.CYAN)


def _check_dependency(module_name: str) -> bool:
    """Check if a Python module is installed.

    Args:
        module_name: Name of the module to check

    Returns:
        True if module is installed, False otherwise
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _check_dependencies() -> bool:
    """Check if all required test dependencies are installed.

    Returns:
        True if all dependencies are installed, False otherwise
    """
    _print_header("Checking Test Dependencies")

    missing_deps = []
    for dep in REQUIRED_DEPENDENCIES:
        if _check_dependency(dep):
            _print_colored(f"  ✓ {dep}", Colors.GREEN)
        else:
            _print_colored(f"  ✗ {dep} (MISSING)", Colors.RED)
            missing_deps.append(dep)

    if missing_deps:
        _print_colored("\n❌ Missing Dependencies!", Colors.RED + Colors.BOLD)
        _print_colored("\nTo install missing dependencies, run:", Colors.YELLOW)
        _print_colored("  uv sync --extra dev\n", Colors.CYAN)
        return False

    _print_colored("\n✅ All dependencies installed!", Colors.GREEN + Colors.BOLD)
    return True


def _run_pytest(args: List[str], description: str, workers: Optional[str] = None) -> int:
    """Run pytest with the specified arguments.

    Args:
        args: List of pytest arguments
        description: Description of what is being tested
        workers: Number of parallel workers or 'auto' (None = serial)

    Returns:
        Exit code from pytest
    """
    _print_header(description)

    # Ensure reports directory exists
    reports_dir = Path("tests/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Add worker configuration if specified
    if workers is not None:
        if "-n" not in args:
            args = args + ["-n", str(workers)]
            if workers != "auto" and int(workers) > 2:
                _print_colored(
                    f"WARNING: Running with {workers} workers may cause OOM on EC2", Colors.YELLOW
                )

    # Build the command
    cmd = ["pytest"] + args

    logger.info(f"Running: {' '.join(cmd)}")

    # Run pytest
    start_time = time.time()
    result = subprocess.run(cmd, cwd=Path.cwd())
    elapsed_time = time.time() - start_time

    # Display elapsed time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    if minutes > 0:
        logger.info(f"Completed in {minutes} minutes and {seconds:.1f} seconds")
    else:
        logger.info(f"Completed in {seconds:.1f} seconds")

    if result.returncode == 0:
        _print_colored(f"\n✅ {description} - PASSED", Colors.GREEN + Colors.BOLD)
    else:
        _print_colored(f"\n❌ {description} - FAILED", Colors.RED + Colors.BOLD)

    return result.returncode


def _run_check() -> int:
    """Check if test dependencies are installed.

    Returns:
        Exit code (0 if all dependencies present, 1 otherwise)
    """
    if _check_dependencies():
        return 0
    return 1


def _run_unit(workers: Optional[str] = None) -> int:
    """Run unit tests only.

    Args:
        workers: Number of parallel workers or 'auto'

    Returns:
        Exit code from pytest
    """
    args = ["-m", "unit", "-v"]
    return _run_pytest(args, "Running Unit Tests", workers)


def _run_integration(workers: Optional[str] = None) -> int:
    """Run integration tests only.

    Args:
        workers: Number of parallel workers or 'auto'

    Returns:
        Exit code from pytest
    """
    # Override coverage threshold for integration tests (they don't hit all code paths)
    args = ["-m", "integration", "-v", "--cov-fail-under=0"]
    return _run_pytest(args, "Running Integration Tests", workers)


def _run_e2e(workers: Optional[str] = None) -> int:
    """Run end-to-end tests only.

    Args:
        workers: Number of parallel workers or 'auto'

    Returns:
        Exit code from pytest
    """
    args = ["-m", "e2e", "-v"]
    return _run_pytest(args, "Running End-to-End Tests", workers)


def _run_fast(workers: Optional[str] = None) -> int:
    """Run fast tests (exclude slow tests).

    Args:
        workers: Number of parallel workers or 'auto'

    Returns:
        Exit code from pytest
    """
    # Use 2 workers by default for fast tests if not specified
    if workers is None:
        workers = "2"
    args = ["-m", "not slow", "-v"]
    return _run_pytest(args, "Running Fast Tests (Excluding Slow)", workers)


def _run_full(workers: Optional[str] = None) -> int:
    """Run full test suite serially (memory-safe for EC2).

    Args:
        workers: Number of parallel workers or 'auto'

    Returns:
        Exit code from pytest
    """
    # Run serially by default to avoid OOM crashes on EC2
    args = ["-v"]
    return _run_pytest(args, "Running Full Test Suite", workers)


def _run_coverage(workers: Optional[str] = None) -> int:
    """Generate coverage reports.

    Args:
        workers: Number of parallel workers or 'auto'

    Returns:
        Exit code from pytest
    """
    args = [
        "-v",
        "--cov=registry",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
    ]
    return _run_pytest(args, "Running Tests with Coverage", workers)


def _run_domain(domain: str, workers: Optional[str] = None) -> int:
    """Run domain-specific tests.

    Args:
        domain: Domain name (auth, servers, search, health, core)
        workers: Number of parallel workers or 'auto'

    Returns:
        Exit code from pytest
    """
    args = ["-m", domain, "-v"]
    description = f"Running {domain.capitalize()} Domain Tests"
    return _run_pytest(args, description, workers)


def main() -> int:
    """Main entry point for the test runner.

    Returns:
        Exit code from the selected test command
    """
    parser = argparse.ArgumentParser(
        description="Test runner for MCP Registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check dependencies
    python scripts/test.py check

    # Run unit tests
    python scripts/test.py unit

    # Run integration tests
    python scripts/test.py integration

    # Run full test suite
    python scripts/test.py full

    # Run fast tests (exclude slow)
    python scripts/test.py fast

    # Generate coverage reports
    python scripts/test.py coverage

    # Run domain-specific tests
    python scripts/test.py auth
    python scripts/test.py servers
    python scripts/test.py search
    python scripts/test.py health
    python scripts/test.py core
""",
    )

    parser.add_argument(
        "command",
        choices=[
            "check",
            "unit",
            "integration",
            "e2e",
            "fast",
            "full",
            "coverage",
            "auth",
            "servers",
            "search",
            "health",
            "core",
        ],
        help="Test command to run",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "-n",
        "--workers",
        type=str,
        default=None,
        help="Number of parallel workers or 'auto' (default: serial). Use with caution on EC2.",
    )

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Route to appropriate function
    workers = args.workers
    command_map = {
        "check": _run_check,
        "unit": lambda: _run_unit(workers),
        "integration": lambda: _run_integration(workers),
        "e2e": lambda: _run_e2e(workers),
        "fast": lambda: _run_fast(workers),
        "full": lambda: _run_full(workers),
        "coverage": lambda: _run_coverage(workers),
        "auth": lambda: _run_domain("auth", workers),
        "servers": lambda: _run_domain("servers", workers),
        "search": lambda: _run_domain("search", workers),
        "health": lambda: _run_domain("health", workers),
        "core": lambda: _run_domain("core", workers),
    }

    return command_map[args.command]()


if __name__ == "__main__":
    sys.exit(main())
