"""Fail scans cleanly when the 'llm' analyzer is configured but LiteLLM is absent.

LiteLLM was removed from the registry image in issue #1826. The three Cisco AI
Defense scanners (MCP, A2A, skill) only need it for their 'llm' analyzer; YARA,
spec, heuristic, endpoint, static and behavioral analysis work without it.

If an operator still lists 'llm' in SECURITY_ANALYZERS, AGENT_SECURITY_ANALYZERS
or SKILL_SECURITY_ANALYZERS, each scanner reacts differently (the A2A scanner
exits 0 and silently skips the analyzer). So the registry fails the scan itself,
before starting the scanner, with one clear message. A failed scan is recorded as
unsafe, so this fails closed.

The MCP scanner also imports LiteLLM at module load, even for a YARA-only scan.
MCP_SCANNER_LITELLM_STANDIN gives it a stand-in so it can start. Remove the
stand-in and this guard once cisco-ai-mcp-scanner makes LiteLLM optional:
https://github.com/cisco-ai-defense/mcp-scanner/issues/285
"""

import importlib.util
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

LLM_ANALYZER: str = "llm"

# Python source run inside the MCP scanner subprocess before mcpscanner.cli is
# imported. Only when the real litellm is missing, it registers a module that
# provides the two names mcp-scanner uses at import time: acompletion (raises if
# called) and an empty exceptions namespace (read with getattr defaults). It
# exists only in that subprocess; nothing else in the registry sees it.
MCP_SCANNER_LITELLM_STANDIN: str = (
    "import importlib.util,sys,types\n"
    "if importlib.util.find_spec('litellm') is None:\n"
    "    _litellm=types.ModuleType('litellm')\n"
    "    async def _unavailable(*args,**kwargs):\n"
    "        raise RuntimeError('LLM analyzer unavailable: litellm is not installed')\n"
    "    _litellm.acompletion=_unavailable\n"
    "    _litellm.exceptions=types.SimpleNamespace()\n"
    "    sys.modules['litellm']=_litellm\n"
)


class LLMAnalyzerUnavailableError(RuntimeError):
    """Raised when a scan requests the 'llm' analyzer and LiteLLM is not installed."""


@lru_cache(maxsize=1)
def _litellm_installed() -> bool:
    """Return True if the real litellm package can be imported."""
    return importlib.util.find_spec("litellm") is not None


def _requests_llm_analyzer(
    analyzers: str | None,
) -> bool:
    """Return True if a comma-separated analyzer list includes 'llm'."""
    if not analyzers:
        return False
    names = [name.strip().lower() for name in analyzers.split(",")]
    return LLM_ANALYZER in names


def _unavailable_message(
    setting_name: str,
) -> str:
    """Build the message an operator sees for a scan that asked for the 'llm' analyzer."""
    return (
        f"security scan failed: the '{LLM_ANALYZER}' analyzer is unavailable in this "
        f"release because LiteLLM is no longer installed. Remove '{LLM_ANALYZER}' from "
        f"{setting_name} (for example use 'yara'), then rescan."
    )


def ensure_llm_analyzer_available(
    analyzers: str | None,
    setting_name: str,
) -> None:
    """Raise if the analyzers include 'llm' and LiteLLM is not installed.

    Call inside the scan's try block, before the scanner subprocess starts, so the
    existing failure path records the scan as failed and unsafe.

    Args:
        analyzers: Comma-separated analyzer list the scan will use.
        setting_name: Environment variable that holds the list, named in the message.

    Raises:
        LLMAnalyzerUnavailableError: If 'llm' is requested and LiteLLM is absent.
    """
    if _requests_llm_analyzer(analyzers) and not _litellm_installed():
        raise LLMAnalyzerUnavailableError(_unavailable_message(setting_name))


def llm_analyzer_settings_in_use(
    analyzer_settings: dict[str, str],
) -> list[str]:
    """Return the setting names whose analyzer list includes 'llm' while LiteLLM is absent.

    Args:
        analyzer_settings: Map of environment variable name to its analyzer list.

    Returns:
        Names of the settings that would make every such scan fail. Empty when
        LiteLLM is installed or no setting requests the 'llm' analyzer.
    """
    if _litellm_installed():
        return []
    return [name for name, value in analyzer_settings.items() if _requests_llm_analyzer(value)]


def print_llm_analyzer_unavailable_banner(
    analyzer_settings: dict[str, str],
) -> None:
    """Warn loudly at startup when a configured analyzer list will make scans fail."""
    in_use = llm_analyzer_settings_in_use(analyzer_settings)
    if not in_use:
        return
    banner = f"""
================================================================================
WARNING: the '{LLM_ANALYZER}' security analyzer is configured but unavailable.

LiteLLM is no longer installed in the registry, so scans that use the
'{LLM_ANALYZER}' analyzer FAIL and the entity is recorded as unsafe.

Settings that include '{LLM_ANALYZER}': {", ".join(in_use)}

Remove '{LLM_ANALYZER}' from those settings (for example use 'yara').
================================================================================
"""
    logger.warning(banner)
    print(banner)
