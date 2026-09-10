"""Unit tests for the CLI duplicate-check command and register pre-flight.

The registry never blocks a registration on the duplicate check, and neither
does the CLI by default: a match is reported and the registration proceeds.
`--fail-on-duplicate` turns it into a CI gate, `--skip-duplicate-check` opts
out. Issue #1696 — before this, nothing outside the registration UI ever
consulted the check, so a CLI or CI registration learned about a duplicate URL
only from the 409 that followed.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "api"))

import registry_management as rm  # noqa: E402


def _args(**overrides) -> argparse.Namespace:
    base = {
        "type": "server",
        "config": None,
        "name": None,
        "description": None,
        "url": None,
        "self_path": None,
        "json": False,
        "fail_on_duplicate": False,
        "skip_duplicate_check": False,
        "registry_url": "http://localhost",
        "token_file": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _envelope(collisions=(), advisory=(), threshold=0.45, available=True) -> dict:
    return {
        "collision_with": list(collisions),
        "advisory_matches": list(advisory),
        "threshold": threshold,
        "similarity_search_available": available,
        "has_collision": bool(collisions),
    }


class TestConfigExtraction:
    """A config the register commands accept must be readable by the check."""

    def test_server_config_uses_server_name_and_proxy_pass_url(self) -> None:
        got = rm._duplicate_check_args_from_config(
            "server",
            {
                "server_name": "Cloudflare Documentation MCP Server",
                "description": "Search Cloudflare documentation",
                "proxy_pass_url": "https://docs.mcp.cloudflare.com/mcp",
            },
        )
        assert got["name"] == "Cloudflare Documentation MCP Server"
        assert got["identity_url"] == "https://docs.mcp.cloudflare.com/mcp"

    def test_server_config_falls_back_to_name(self) -> None:
        got = rm._duplicate_check_args_from_config("server", {"name": "fallback"})
        assert got["name"] == "fallback"
        assert got["identity_url"] is None

    def test_agent_config_uses_url(self) -> None:
        got = rm._duplicate_check_args_from_config(
            "agent", {"name": "Topic Agent", "url": "https://example.com/agents/topic"}
        )
        assert got["identity_url"] == "https://example.com/agents/topic"

    def test_skill_config_prefers_skill_md_url(self) -> None:
        got = rm._duplicate_check_args_from_config(
            "skill",
            {
                "skill_name": "PDF",
                "skill_md_url": "https://example.com/SKILL.md",
                "url": "https://ignored",
            },
        )
        assert got["name"] == "PDF"
        assert got["identity_url"] == "https://example.com/SKILL.md"


class TestReportDuplicateCheck:
    def test_reports_collision_and_advisory(self, caplog) -> None:
        envelope = _envelope(
            collisions=[
                {
                    "path": "/cloudflare-docs",
                    "entity_type": "mcp_server",
                    "name": "Cloudflare Docs",
                    "match_reason": "exact URL match",
                }
            ],
            advisory=[
                {
                    "path": "/similar",
                    "entity_type": "mcp_server",
                    "name": "Similar",
                    "relevance_score": 0.9123,
                }
            ],
        )
        with caplog.at_level("INFO"):
            assert rm._report_duplicate_check(envelope, "server 'x'") is True

        assert "/cloudflare-docs" in caplog.text
        assert "exact URL match" in caplog.text
        assert "0.9123" in caplog.text

    def test_reports_nothing_found(self, caplog) -> None:
        with caplog.at_level("INFO"):
            assert rm._report_duplicate_check(_envelope(), "server 'x'") is False
        assert "No duplicates found" in caplog.text

    def test_degraded_similarity_is_called_out(self, caplog) -> None:
        """An empty advisory during an embedding outage must not read as 'all clear'."""
        with caplog.at_level("INFO"):
            rm._report_duplicate_check(_envelope(available=False), "server 'x'")
        assert "similarity check did not run" in caplog.text


class TestCheckDuplicatesCommand:
    def test_reads_a_config_file(self, tmp_path) -> None:
        cfg = tmp_path / "server.json"
        cfg.write_text(
            json.dumps(
                {
                    "server_name": "topic-mcp-server",
                    "description": "Extracts topics",
                    "proxy_pass_url": "http://topic:9001/",
                }
            )
        )
        client = MagicMock()
        client.check_duplicates.return_value = _envelope()

        with patch.object(rm, "_create_client", return_value=client):
            assert rm.cmd_check_duplicates(_args(config=str(cfg))) == 0

        client.check_duplicates.assert_called_once_with(
            "server",
            name="topic-mcp-server",
            description="Extracts topics",
            identity_url="http://topic:9001/",
            self_path=None,
        )

    def test_explicit_flags_without_a_config(self) -> None:
        client = MagicMock()
        client.check_duplicates.return_value = _envelope()

        with patch.object(rm, "_create_client", return_value=client):
            assert rm.cmd_check_duplicates(_args(name="topic", description="d", url="u")) == 0

        _, kwargs = client.check_duplicates.call_args
        assert kwargs["name"] == "topic"
        assert kwargs["identity_url"] == "u"

    def test_requires_name_or_config(self) -> None:
        assert rm.cmd_check_duplicates(_args()) == 1

    def test_fail_on_duplicate_exits_nonzero(self) -> None:
        client = MagicMock()
        client.check_duplicates.return_value = _envelope(
            collisions=[{"path": "/dup", "entity_type": "mcp_server", "name": "D"}]
        )
        with patch.object(rm, "_create_client", return_value=client):
            assert rm.cmd_check_duplicates(_args(name="x", fail_on_duplicate=True)) == 1

    def test_match_alone_still_exits_zero(self) -> None:
        """The check is advisory: finding something is not an error."""
        client = MagicMock()
        client.check_duplicates.return_value = _envelope(
            collisions=[{"path": "/dup", "entity_type": "mcp_server", "name": "D"}]
        )
        with patch.object(rm, "_create_client", return_value=client):
            assert rm.cmd_check_duplicates(_args(name="x")) == 0


class TestRegisterPreflight:
    def test_preflight_reports_and_allows_registration(self, caplog) -> None:
        client = MagicMock()
        client.check_duplicates.return_value = _envelope(
            collisions=[
                {
                    "path": "/cloudflare-docs",
                    "entity_type": "mcp_server",
                    "name": "Cloudflare Docs",
                    "match_reason": "exact URL match",
                }
            ]
        )
        with patch.object(rm, "_create_client", return_value=client), caplog.at_level("INFO"):
            code = rm._preflight_duplicate_check(
                _args(), "server", {"server_name": "x", "proxy_pass_url": "u"}
            )

        assert code == 0
        assert "/cloudflare-docs" in caplog.text

    def test_fail_on_duplicate_blocks_before_the_api_call(self) -> None:
        client = MagicMock()
        client.check_duplicates.return_value = _envelope(
            collisions=[{"path": "/dup", "entity_type": "mcp_server", "name": "D"}]
        )
        with patch.object(rm, "_create_client", return_value=client):
            code = rm._preflight_duplicate_check(
                _args(fail_on_duplicate=True), "server", {"server_name": "x"}
            )
        assert code == 1

    def test_skip_flag_makes_no_request(self) -> None:
        client = MagicMock()
        with patch.object(rm, "_create_client", return_value=client):
            code = rm._preflight_duplicate_check(
                _args(skip_duplicate_check=True), "server", {"server_name": "x"}
            )
        assert code == 0
        client.check_duplicates.assert_not_called()

    def test_a_failing_check_never_blocks_registration(self, caplog) -> None:
        """A hint that cannot run must not become the reason a registration fails."""
        client = MagicMock()
        client.check_duplicates.side_effect = RuntimeError("registry unreachable")
        with patch.object(rm, "_create_client", return_value=client), caplog.at_level("INFO"):
            code = rm._preflight_duplicate_check(_args(), "server", {"server_name": "x"})

        assert code == 0
        assert "Duplicate check skipped" in caplog.text

    def test_config_without_a_name_skips_the_check(self) -> None:
        client = MagicMock()
        with patch.object(rm, "_create_client", return_value=client):
            assert rm._preflight_duplicate_check(_args(), "server", {}) == 0
        client.check_duplicates.assert_not_called()


class TestClientCheckDuplicates:
    """The endpoint takes JSON; /api/servers/* otherwise defaults to form-encoded."""

    @pytest.mark.parametrize(
        ("entity_type", "endpoint", "url_field"),
        [
            ("server", "/api/servers/check-duplicates", "proxy_pass_url"),
            ("agent", "/api/agents/check-duplicates", "url"),
            ("skill", "/api/skills/check-duplicates", "skill_md_url"),
        ],
    )
    def test_payload_and_endpoint_per_entity_type(self, entity_type, endpoint, url_field) -> None:
        from registry_client import RegistryClient

        client = RegistryClient.__new__(RegistryClient)
        response = MagicMock()
        response.json.return_value = _envelope()
        with patch.object(RegistryClient, "_make_request", return_value=response) as request:
            client.check_duplicates(entity_type, "n", description="d", identity_url="u")

        kwargs = request.call_args.kwargs
        assert kwargs["endpoint"] == endpoint
        assert kwargs["data"] == {"name": "n", "description": "d", url_field: "u"}

    def test_rejects_an_unknown_entity_type(self) -> None:
        from registry_client import RegistryClient

        client = RegistryClient.__new__(RegistryClient)
        with pytest.raises(ValueError, match="server, agent, or skill"):
            client.check_duplicates("widget", "n")

    def test_check_duplicates_endpoints_are_sent_as_json(self) -> None:
        """Regression: /api/servers/check-duplicates must not be form-encoded."""
        import inspect

        from registry_client import RegistryClient

        source = inspect.getsource(RegistryClient._make_request)
        assert '"/check-duplicates" in endpoint' in source
