"""Unit tests for get_custom_entity in servers/mcpgw/server.py.

The tool builds GET /api/custom/{type}/{uuid} from a caller-supplied record
path. This server authenticates to the registry with a privileged credential,
so a traversal value in either segment (e.g. "../../api/management/iam/users")
would otherwise reach arbitrary registry endpoints with that privilege after
URL normalization. Both segments must be validated at the interpolation site
and fail closed (no HTTP request made) on any non-matching value. The
rationale matches test_get_skill_content.py.
"""

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# The mcpgw server depends on `fastmcp` which is not installed in the main
# project venv. Stub it out before importing the server module. FastMCP.tool()
# is a decorator, so make it a passthrough to keep the functions callable.
if "fastmcp" not in sys.modules:
    _fastmcp_stub = types.ModuleType("fastmcp")
    _fastmcp_stub.Context = type("Context", (), {})
    _mock_mcp = MagicMock()
    _mock_mcp.tool.return_value = lambda fn: fn
    _fastmcp_stub.FastMCP = MagicMock(return_value=_mock_mcp)
    sys.modules["fastmcp"] = _fastmcp_stub

# Add servers/mcpgw to sys.path so that `from models import ...` works
# when importing servers.mcpgw.server
_mcpgw_path = str(Path(__file__).resolve().parents[4] / "servers" / "mcpgw")
if _mcpgw_path not in sys.path:
    sys.path.insert(0, _mcpgw_path)

import servers.mcpgw.server as mcpgw_server

VALID_UUID = "3f2b8c1e-9d4a-4b7e-8f21-6a5c0d9e7b13"
VALID_PATH = f"/data_product/{VALID_UUID}"

RECORD = {
    "path": VALID_PATH,
    "entity_type": "data_product",
    "name": "Customer churn dataset",
    "description": "Monthly churn snapshot",
    "visibility": "public",
    "owner": "alice",
    "tags": ["analytics"],
    "is_enabled": True,
    "num_stars": 4.5,
    "created_at": "2026-10-01T12:00:00Z",
    "updated_at": "2026-10-02T12:00:00Z",
    "attributes": {"retention_days": 90, "steward": "data-platform"},
    # Fields the registry returns that CustomEntityInfo drops.
    "allowed_groups": ["data-stewards"],
    "rating_details": [{"user": "bob", "rating": 5}],
    "is_proxied": False,
}


def _make_mock_response(
    payload: dict | None = None,
    status_code: int = 200,
) -> MagicMock:
    """Create a mock httpx response; a 4xx/5xx raises from raise_for_status."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = payload or {}
    if status_code >= 400:
        request = httpx.Request("GET", "http://registry/api/custom/x")
        response = httpx.Response(status_code, request=request)
        mock_resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("error", request=request, response=response)
        )
    else:
        mock_resp.raise_for_status = MagicMock()
    return mock_resp


async def _call_get_custom_entity(
    path: str,
    response: MagicMock | None = None,
    capture: dict | None = None,
) -> dict:
    """Call get_custom_entity with a mocked pooled client and registry headers.

    The tool requests through the process-lifetime pooled client
    (``servers/mcpgw/http_pool.py``), so the seam is ``shared_async_client``
    plus ``client.request(method, url, ...)``.

    Args:
        path: Value passed to the tool.
        response: Mock response the registry returns (defaults to RECORD).
        capture: If provided, populated with the requested "method" and "url".

    Returns:
        The tool's result dict plus "_http_call_count".
    """
    mock_response = response or _make_mock_response(RECORD)
    called = {"count": 0}

    async def mock_request(method, url, **request_kwargs):
        called["count"] += 1
        if capture is not None:
            capture["method"] = method
            capture["url"] = url
        return mock_response

    mock_client = AsyncMock()
    mock_client.request = mock_request

    with (
        patch.object(mcpgw_server, "shared_async_client", return_value=mock_client),
        patch.object(
            mcpgw_server,
            "_get_registry_headers",
            AsyncMock(return_value={"Authorization": "Bearer test-token"}),
        ),
    ):
        result = await mcpgw_server.get_custom_entity(path)

    # A pooled client must never be closed by a request path: that would tear the
    # pool down for every other concurrent caller.
    mock_client.aclose.assert_not_awaited()
    result["_http_call_count"] = called["count"]
    return result


class TestGetCustomEntityValidPath:
    """A well-formed record path reaches the registry and returns the record."""

    async def test_returns_full_record_with_attributes(self):
        capture: dict = {}
        result = await _call_get_custom_entity(VALID_PATH, capture=capture)

        assert result["status"] == "success"
        assert result["record"]["attributes"] == RECORD["attributes"]
        assert capture["method"] == "GET"
        assert capture["url"].endswith(f"/api/custom/data_product/{VALID_UUID}")
        assert result["_http_call_count"] == 1

    async def test_record_is_shaped_by_custom_entity_info(self):
        result = await _call_get_custom_entity(VALID_PATH)

        assert result["record"] == {
            "path": VALID_PATH,
            "entity_type": "data_product",
            "name": "Customer churn dataset",
            "description": "Monthly churn snapshot",
            "visibility": "public",
            "owner": "alice",
            "tags": ["analytics"],
            "enabled": True,
            "num_stars": 4.5,
            "created_at": "2026-10-01T12:00:00Z",
            "updated_at": "2026-10-02T12:00:00Z",
            "attributes": {"retention_days": 90, "steward": "data-platform"},
        }

    async def test_logs_only_the_validated_path(self, caplog):
        with caplog.at_level("INFO", logger=mcpgw_server.logger.name):
            await _call_get_custom_entity(f"  data_product/{VALID_UUID}/\n")

        assert f"get_custom_entity called: path=/data_product/{VALID_UUID}" in caplog.text

    @pytest.mark.parametrize(
        "path",
        [
            f"data_product/{VALID_UUID}",
            f"  {VALID_PATH}\n",
            f"{VALID_PATH}/",
        ],
    )
    async def test_accepted_path_variants_build_canonical_url(self, path):
        # A missing leading slash, surrounding whitespace and a trailing slash
        # are tolerated; the URL is rebuilt from the validated segments.
        capture: dict = {}
        result = await _call_get_custom_entity(path, capture=capture)

        assert result["status"] == "success"
        assert capture["url"].endswith(f"/api/custom/data_product/{VALID_UUID}")


class TestGetCustomEntityTraversalRejected:
    """Non-matching values in either segment fail closed with no HTTP call."""

    @pytest.mark.parametrize(
        "path",
        [
            # Type segment
            f"/../{VALID_UUID}",
            f"/..%2f..%2fapi/{VALID_UUID}",
            f"/data.product/{VALID_UUID}",
            f"/Data_Product/{VALID_UUID}",
            f"/data product/{VALID_UUID}",
            f"/{'a' * 65}/{VALID_UUID}",
            f"//{VALID_UUID}",
            # UUID segment
            "/data_product/../../api/management/iam/users",
            "/data_product/..%2f..%2fapi",
            "/data_product/not-a-uuid",
            f"/data_product/{VALID_UUID}x",
            f"/data_product/{VALID_UUID}/rate",
            f"/data_product/{VALID_UUID}\n../x",
            f"/data_product/{VALID_UUID}?x=1",
            # Whole value
            "../../api/management/iam/users",
            "",
            "   ",
            VALID_UUID,
        ],
    )
    async def test_invalid_path_rejected_without_request(self, path):
        capture: dict = {}
        result = await _call_get_custom_entity(path, capture=capture)

        assert result["status"] == "failed"
        assert result["_http_call_count"] == 0
        assert "url" not in capture

    async def test_rejected_path_is_logged_truncated_and_escaped(self, caplog):
        # A newline inside the logged prefix must be escaped, not start a
        # forged log line, and the tail past the cap must not be logged.
        payload = "../\nFORGED-LINE " + "a" * 500
        with caplog.at_level("INFO", logger=mcpgw_server.logger.name):
            await _call_get_custom_entity(payload)

        rejected = [r.getMessage() for r in caplog.records if "rejected" in r.getMessage()]
        assert len(rejected) == 1
        assert "\n" not in rejected[0]
        assert repr(payload[: mcpgw_server.MAX_LOGGED_INPUT_LEN]) in rejected[0]
        assert "a" * mcpgw_server.MAX_LOGGED_INPUT_LEN not in rejected[0]


class TestGetCustomEntityRegistryErrors:
    """Registry refusals surface as clean tool errors."""

    async def test_denied_or_missing_record_returns_clean_404(self):
        # The registry answers 404 both when the record is absent and when the
        # caller may not view it, so it does not disclose existence.
        result = await _call_get_custom_entity(
            VALID_PATH, response=_make_mock_response(status_code=404)
        )

        assert result["status"] == "failed"
        assert result["error"] == "HTTP 404"
        assert "record" not in result

    async def test_server_error_returns_clean_error(self):
        result = await _call_get_custom_entity(
            VALID_PATH, response=_make_mock_response(status_code=500)
        )

        assert result["status"] == "failed"
        assert result["error"] == "HTTP 500"
