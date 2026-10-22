"""The A2A scan card must not carry null fields the registry added (issue #1921).

AgentCard.model_dump() writes every unset optional field as null. The scanner's
YARA rule UnauthorizedNetworkAccess_UnsignedMessages matched the added
'"signature": null' (HIGH), and the spec analyzer flagged a null skill
'examples', so every unsigned agent registered through the registry was marked
unsafe and disabled by default.
"""

import json
import shutil
import subprocess  # nosec B404 - runs the installed a2a-scanner in one test
import sys
import tempfile
from pathlib import Path
from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from registry.schemas.agent_models import AgentCard
from registry.services.agent_scanner import _drop_none_values

SCANNER_TIMEOUT_SECONDS: int = 120
UNSIGNED_RULE: str = "UnauthorizedNetworkAccess_UnsignedMessages"

# A card with no signature and no skill examples, as an operator would send it.
MINIMAL_CARD: dict = {
    "name": "Weather Agent",
    "description": "Answers questions about the weather forecast for a city.",
    "url": "https://agent.example.com/a2a",
    "path": "/weather-agent",
    "supportedProtocol": "a2a",
    "protocol_version": "0.3.0",
    "version": "1.0.0",
    "capabilities": {"streaming": False},
    "default_input_modes": ["text"],
    "default_output_modes": ["text"],
    "skills": [
        {
            "id": "forecast",
            "name": "Forecast",
            "description": "Returns the forecast for a city.",
            "tags": ["weather"],
        }
    ],
}


def _registry_card() -> dict:
    """Build the dict the registry hands to the scanner, nulls included."""
    return AgentCard(**MINIMAL_CARD).model_dump()


def _capture_scanned_card(
    agent_card: dict,
) -> dict:
    """Run _run_a2a_scanner with subprocess mocked and return the card it wrote."""
    with patch("registry.services.agent_scanner.get_security_scan_repository"):
        from registry.services.agent_scanner import AgentScannerService

        service = AgentScannerService()
    captured: dict = {}

    def _fake_run(cmd, *args, **kwargs):
        with open(cmd[2]) as handle:
            captured.update(json.load(handle))
        result = MagicMock()
        result.stdout = "{}"
        result.stderr = ""
        result.returncode = 0
        return result

    with patch("registry.services.agent_scanner.subprocess.run", side_effect=_fake_run):
        service._run_a2a_scanner(
            agent_card=agent_card, agent_path="/weather-agent", analyzers="yara"
        )
    return captured


def _a2a_scanner_path() -> str | None:
    """Return the installed a2a-scanner executable, preferring the active venv."""
    candidate = Path(sys.executable).parent / "a2a-scanner"
    if candidate.exists():
        return str(candidate)
    return shutil.which("a2a-scanner")


@pytest.mark.unit
class TestDropNoneValues:
    """None is removed at every depth; every other value is kept as-is."""

    def test_removes_none_at_top_level(self):
        assert _drop_none_values({"a": 1, "signature": None}) == {"a": 1}

    def test_removes_none_inside_nested_lists_and_dicts(self):
        value = {"skills": [{"id": "x", "examples": None, "tags": ["t", None]}]}
        assert _drop_none_values(value) == {"skills": [{"id": "x", "tags": ["t"]}]}

    def test_keeps_falsy_values_that_are_not_none(self):
        value = {"verify_signature": False, "signature": "", "count": 0, "tags": []}
        assert _drop_none_values(value) == value

    def test_does_not_mutate_the_input(self):
        value = {"a": None, "b": {"c": None}}
        _drop_none_values(value)
        assert value == {"a": None, "b": {"c": None}}


@pytest.mark.unit
class TestScannedCardHasNoNulls:
    """The card written for the scanner carries only what the agent provided."""

    def test_registry_card_has_nulls_to_begin_with(self):
        # Guards the premise: if model_dump stops emitting these, the fix is moot.
        card = _registry_card()
        assert card["signature"] is None
        assert card["skills"][0]["examples"] is None

    def test_scanned_card_omits_null_signature_and_examples(self):
        scanned = _capture_scanned_card(_registry_card())
        assert "signature" not in scanned
        assert "signatures" not in scanned
        assert "examples" not in scanned["skills"][0]
        assert "null" not in json.dumps(scanned)

    def test_real_signature_values_are_still_scanned(self):
        card = _registry_card() | {"signature": "", "verify_signature": False}
        scanned = _capture_scanned_card(card)
        assert scanned["signature"] == ""
        assert scanned["verify_signature"] is False


@pytest.mark.unit
class TestRealScannerOnRegistryCard:
    """Run the installed a2a-scanner on the registry's card, as registration does."""

    def _findings(
        self,
        card: dict,
    ) -> list[str]:
        scanner = _a2a_scanner_path()
        if scanner is None:
            pytest.skip("a2a-scanner is not installed")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(card, handle, default=str)
            path = handle.name
        result = subprocess.run(  # nosec B603 - fixed scanner path and test-authored args
            [scanner, "scan-card", path, "--analyzers", "yara,spec", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=SCANNER_TIMEOUT_SECONDS,
        )
        output = json.loads(result.stdout[result.stdout.find("{") :])
        return [json.dumps(finding, default=str) for finding in output.get("findings", [])]

    def test_null_signature_triggers_the_rule(self):
        findings = self._findings(_registry_card())
        assert any(UNSIGNED_RULE in finding for finding in findings)

    def test_cleaned_card_has_no_false_findings(self):
        assert self._findings(_drop_none_values(_registry_card())) == []
