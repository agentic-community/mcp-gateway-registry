"""The OpenAPI MCP state gauge reports only while the feature is on (issue #2022).

Reachability starts at 0, so with the feature off the gauge would make the
documented "container_reachable stays at 0" alert fire on every deployment.
"""

from unittest.mock import patch

from registry.core.config import settings
from registry.observability import meters
from registry.services import openapi_mcp_reconciler as reconciler


def _kinds() -> list[str]:
    return [o.attributes["kind"] for o in meters._openapi_mcp_state_callback(None)]


def test_feature_off_reports_nothing():
    with patch.object(settings, "openapi_mcp_enabled", False):
        assert _kinds() == []


def test_feature_on_reports_servers_targets_and_reachability():
    with (
        patch.object(settings, "openapi_mcp_enabled", True),
        patch.object(reconciler, "_container_reachable", True),
    ):
        observations = list(meters._openapi_mcp_state_callback(None))

    assert [o.attributes["kind"] for o in observations] == [
        "servers",
        "targets",
        "container_reachable",
    ]
    assert observations[2].value == 1
