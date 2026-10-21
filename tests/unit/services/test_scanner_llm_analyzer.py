"""Tests for the 'llm' analyzer guard and the mcp-scanner LiteLLM stand-in (issue #1826)."""

import logging
import subprocess
import sys
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from registry.services import scanner_llm_analyzer
from registry.services.scanner_llm_analyzer import (
    MCP_SCANNER_LITELLM_STANDIN,
    LLMAnalyzerUnavailableError,
    ensure_llm_analyzer_available,
    llm_analyzer_settings_in_use,
    print_llm_analyzer_unavailable_banner,
)

SANDBOX_TIMEOUT_SECONDS: int = 60


@pytest.fixture
def litellm_absent():
    """Make the guard believe LiteLLM is not installed."""
    with patch.object(scanner_llm_analyzer, "_litellm_installed", return_value=False):
        yield


@pytest.fixture
def litellm_present():
    """Make the guard believe LiteLLM is installed."""
    with patch.object(scanner_llm_analyzer, "_litellm_installed", return_value=True):
        yield


@pytest.mark.unit
class TestEnsureLLMAnalyzerAvailable:
    """The guard raises only when 'llm' is requested and LiteLLM is absent."""

    @pytest.mark.parametrize("analyzers", ["llm", "yara,llm", " YARA , LLM "])
    def test_raises_when_llm_requested(self, analyzers, litellm_absent):
        with pytest.raises(LLMAnalyzerUnavailableError) as exc_info:
            ensure_llm_analyzer_available(analyzers, "SECURITY_ANALYZERS")
        assert "Remove 'llm' from SECURITY_ANALYZERS" in str(exc_info.value)

    @pytest.mark.parametrize("analyzers", ["yara", "yara,spec", "static,behavioral", "", None])
    def test_silent_without_llm(self, analyzers, litellm_absent):
        ensure_llm_analyzer_available(analyzers, "SECURITY_ANALYZERS")

    def test_silent_when_litellm_installed(self, litellm_present):
        ensure_llm_analyzer_available("yara,llm", "SECURITY_ANALYZERS")

    def test_llm_substring_is_not_a_match(self, litellm_absent):
        # Only the exact analyzer name counts, not names that contain "llm".
        ensure_llm_analyzer_available("yara,llm-judge", "SECURITY_ANALYZERS")


@pytest.mark.unit
class TestStartupBanner:
    """The banner names the settings that will make scans fail."""

    def test_lists_settings_using_llm(self, litellm_absent, capsys, caplog):
        with caplog.at_level(logging.WARNING):
            print_llm_analyzer_unavailable_banner(
                {"SECURITY_ANALYZERS": "yara,llm", "SKILL_SECURITY_ANALYZERS": "static"}
            )
        out = capsys.readouterr().out
        assert "SECURITY_ANALYZERS" in out
        assert "SKILL_SECURITY_ANALYZERS" not in out
        assert "unavailable" in caplog.text

    def test_silent_when_no_setting_uses_llm(self, litellm_absent, capsys):
        print_llm_analyzer_unavailable_banner({"SECURITY_ANALYZERS": "yara"})
        assert capsys.readouterr().out == ""

    def test_nothing_in_use_when_litellm_installed(self, litellm_present):
        assert llm_analyzer_settings_in_use({"SECURITY_ANALYZERS": "llm"}) == []

    def test_banner_called_at_startup(self):
        from pathlib import Path

        main_source = (Path(__file__).resolve().parents[3] / "registry" / "main.py").read_text()
        assert "print_llm_analyzer_unavailable_banner(" in main_source


@pytest.mark.unit
class TestMCPScannerStandIn:
    """The stand-in lets mcp-scanner import without LiteLLM, and only then."""

    def _run(
        self,
        code: str,
    ) -> subprocess.CompletedProcess:
        """Run code in a fresh interpreter, the way the scanner shim runs."""
        return subprocess.run(  # nosec B603 - fixed interpreter and test-authored code
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT_SECONDS,
        )

    def test_provides_the_names_mcp_scanner_imports(self):
        # The stand-in decides with importlib.util.find_spec, so making that return
        # None for litellm reproduces an image without the package.
        code = (
            "import importlib.util\n"
            "_orig = importlib.util.find_spec\n"
            "importlib.util.find_spec = lambda n, *a: None if n == 'litellm' else _orig(n, *a)\n"
            + MCP_SCANNER_LITELLM_STANDIN
            + "from litellm import acompletion\n"
            "import asyncio, litellm\n"
            "assert getattr(litellm.exceptions, 'RateLimitError', None) is None\n"
            "try:\n"
            "    asyncio.run(acompletion(model='x', messages=[]))\n"
            "except RuntimeError as e:\n"
            "    print('raised:', e)\n"
        )
        result = self._run(code)
        assert result.returncode == 0, result.stderr
        assert "raised: LLM analyzer unavailable" in result.stdout

    def test_leaves_a_real_litellm_alone(self):
        code = (
            "import sys, types, importlib.util\n"
            "real = types.ModuleType('litellm'); real.marker = 'real'\n"
            "sys.modules['litellm'] = real\n"
            "_orig = importlib.util.find_spec\n"
            "importlib.util.find_spec = lambda n, *a: object() if n == 'litellm' else _orig(n, *a)\n"
            + MCP_SCANNER_LITELLM_STANDIN
            + "import litellm\n"
            "print(getattr(litellm, 'marker', 'standin'))\n"
        )
        result = self._run(code)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "real"


@pytest.mark.unit
class TestScanServicesFailClosed:
    """Each scan service records a failed, unsafe scan when 'llm' is unavailable."""

    @pytest.mark.asyncio
    async def test_agent_scan_fails_before_scanner_runs(self, litellm_absent):
        from registry.services.agent_scanner import AgentScannerService

        service = AgentScannerService.__new__(AgentScannerService)
        service._scan_repo = MagicMock(create=AsyncMock())
        with patch.object(AgentScannerService, "_run_a2a_scanner") as run_scanner:
            result = await service.scan_agent(
                agent_card={"name": "a", "url": "https://agent.example.com"},
                agent_path="/a",
                analyzers="yara,llm",
                api_key=None,
                timeout=5,
            )
        run_scanner.assert_not_called()
        assert result.scan_failed is True
        assert result.is_safe is False
        assert "AGENT_SECURITY_ANALYZERS" in result.error_message

    @pytest.mark.asyncio
    async def test_skill_scan_fails_before_scanner_runs(self, litellm_absent):
        from registry.services.skill_scanner import SkillScannerService

        service = SkillScannerService.__new__(SkillScannerService)
        service._scan_repo = MagicMock(create=AsyncMock())
        with patch.object(SkillScannerService, "_run_skill_scanner") as run_scanner:
            result = await service.scan_skill(
                skill_path="/s",
                skill_md_url="https://example.com/SKILL.md",
                analyzers="static,llm",
                timeout=5,
            )
        run_scanner.assert_not_called()
        assert result.scan_failed is True
        assert result.is_safe is False
        assert "SKILL_SECURITY_ANALYZERS" in result.error_message


@pytest.mark.unit
class TestMCPScanFailureMessage:
    """The MCP scan failure message passes the guard's text through unchanged."""

    def test_no_credential_hint_for_llm_error(self):
        from registry.services.security_scanner import _scan_failure_message

        error = LLMAnalyzerUnavailableError(
            "security scan failed: the 'llm' analyzer is unavailable"
        )
        message = _scan_failure_message(error, headers=None)
        assert message == str(error)
        assert "credential" not in message

    def test_shim_includes_the_standin(self):
        from registry.services.security_scanner import _SCANNER_SHIM

        assert _SCANNER_SHIM.startswith(MCP_SCANNER_LITELLM_STANDIN)
        assert "from mcpscanner.cli import cli_entry_point" in _SCANNER_SHIM
