"""Unit tests for get_skill_content in servers/mcpgw/server.py.

These tests verify that skill_name is validated against a strict identifier
allowlist before being interpolated into the outbound registry URL path
segment. Because this server authenticates to the registry with a privileged
credential, a traversal skill_name (e.g. "../../api/management/iam/users")
would otherwise reach arbitrary registry endpoints with that privilege after
URL normalization. Validation must happen at the interpolation site and must
fail closed (no HTTP request made) on any non-matching value.
"""

import base64
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The mcpgw server depends on `fastmcp` which is not installed in the main
# project venv. Stub it out before importing the server module.
# FastMCP.tool() is a decorator — make it a passthrough so the original
# async functions remain callable.
if "fastmcp" not in sys.modules:
    _fastmcp_stub = types.ModuleType("fastmcp")
    _fastmcp_stub.Context = type("Context", (), {})
    _mock_mcp = MagicMock()
    _mock_mcp.tool.return_value = lambda fn: fn  # decorator is a no-op
    _fastmcp_stub.FastMCP = MagicMock(return_value=_mock_mcp)
    sys.modules["fastmcp"] = _fastmcp_stub

# Add servers/mcpgw to sys.path so that `from models import ...` works
# when importing servers.mcpgw.server
_mcpgw_path = str(Path(__file__).resolve().parents[4] / "servers" / "mcpgw")
if _mcpgw_path not in sys.path:
    sys.path.insert(0, _mcpgw_path)

# Import (not re-import) the server module. Popping it from sys.modules here
# would break sibling test modules that already hold a reference to the
# previously imported module object; the stub above is only installed when
# fastmcp is genuinely absent.
import servers.mcpgw.server as mcpgw_server


def _make_mock_response(payload=None, status_code=200):
    """Create a mock httpx response with the given JSON payload."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = payload or {}
    return mock_resp


async def _call_get_skill_content(
    skill_name,
    resource_path=None,
    capture=None,
    payload=None,
    include_binary=False,
):
    """Call get_skill_content with a mocked HTTP client and registry headers.

    Args:
        skill_name: Value passed to the tool.
        resource_path: Optional resource_path value.
        capture: If provided, a dict populated with the captured .get() args
                 (keys "url" and "params") whenever the client is invoked.
        payload: Optional JSON body returned by the registry.
        include_binary: Value passed to the tool's include_binary parameter.

    Returns:
        The result dict from get_skill_content.
    """
    mock_response = _make_mock_response(
        payload
        if payload is not None
        else {"url": "https://registry.example/skills/x", "content": "hello"}
    )

    called = {"count": 0}

    async def mock_get(url, **get_kwargs):
        called["count"] += 1
        if capture is not None:
            capture["url"] = url
            capture["params"] = get_kwargs.get("params")
        return mock_response

    mock_client = AsyncMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(mcpgw_server.httpx, "AsyncClient", return_value=mock_client),
        patch.object(
            mcpgw_server,
            "_get_registry_headers",
            AsyncMock(return_value={"Authorization": "Bearer test-token"}),
        ),
    ):
        result = await mcpgw_server.get_skill_content(
            skill_name,
            resource_path=resource_path,
            include_binary=include_binary,
        )

    result["_http_call_count"] = called["count"]
    return result


class TestGetSkillContentValidName:
    """Valid skill names proceed to the registry request."""

    async def test_valid_name_builds_expected_url(self):
        capture: dict = {}
        result = await _call_get_skill_content("my-skill-1", capture=capture)

        assert result["status"] == "success"
        assert result["_http_call_count"] == 1
        assert capture["url"].endswith("/api/skills/my-skill-1/content")

    async def test_valid_name_with_surrounding_whitespace_is_stripped(self):
        # Leading/trailing whitespace (incl. a trailing newline) is stripped
        # before validation, so a valid name wrapped in whitespace still works.
        capture: dict = {}
        result = await _call_get_skill_content("  my-skill-1\n", capture=capture)

        assert result["status"] == "success"
        assert capture["url"].endswith("/api/skills/my-skill-1/content")

    async def test_valid_name_with_benign_resource_path(self):
        capture: dict = {}
        result = await _call_get_skill_content(
            "pr-review",
            resource_path="references/architecture.md",
            capture=capture,
        )

        assert result["status"] == "success"
        assert result["_http_call_count"] == 1
        assert capture["params"] == {"resource": "references/architecture.md"}


class TestGetSkillContentTraversalRejected:
    """Traversal / non-identifier skill names are rejected with no HTTP call."""

    @pytest.mark.parametrize(
        "payload",
        [
            "../../api/management/iam/users",
            "..%2f..%2fx",
            "foo/bar",
            "foo bar",
            "foo..bar",
            "/abs",
            "",
            "   ",
            "%2e%2e",
            "foo%2Fbar",
            "..",
            # An embedded newline (not merely trailing, which .strip removes)
            # must be rejected by the pattern, not normalized away downstream.
            "valid-name\n../x",
            # Uppercase and underscore are not permitted by the registry's
            # skill-name rule, so they are rejected here too (exact, not just safe).
            "MySkill",
            "my_skill",
            # A leading/trailing hyphen or empty group is not a valid identifier.
            "-leading",
            "trailing-",
            "a--b",
        ],
    )
    async def test_traversal_payload_rejected(self, payload):
        capture: dict = {}
        result = await _call_get_skill_content(payload, capture=capture)

        assert result["status"] == "failed"
        # No registry request was made.
        assert result["_http_call_count"] == 0
        assert "url" not in capture


class TestGetSkillContentResourcePathDefense:
    """resource_path defense-in-depth: reject absolute / traversal values."""

    @pytest.mark.parametrize(
        "resource_path",
        [
            "../x",
            "/abs",
            "a/../../etc/passwd",
        ],
    )
    async def test_bad_resource_path_rejected(self, resource_path):
        capture: dict = {}
        result = await _call_get_skill_content(
            "valid-skill",
            resource_path=resource_path,
            capture=capture,
        )

        assert result["status"] == "failed"
        assert result["_http_call_count"] == 0
        assert "url" not in capture

    async def test_benign_resource_path_passes(self):
        capture: dict = {}
        result = await _call_get_skill_content(
            "valid-skill",
            resource_path="references/notes.md",
            capture=capture,
        )

        assert result["status"] == "success"
        assert result["_http_call_count"] == 1


def _binary_asset_payload(path: str, data: bytes) -> dict:
    """Build the registry JSON body for an asset resource."""
    return {
        "content_base64": base64.b64encode(data).decode("ascii"),
        "encoding": "base64",
        "size_bytes": len(data),
        "path": path,
        "type": "asset",
        "url": f"https://raw.example.com/{path}",
    }


class TestGetSkillContentBinaryResource:
    """Binary assets return metadata by default and base64 only on request."""

    async def test_binary_resource_returns_metadata_by_default(self):
        png_bytes = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
        payload = _binary_asset_payload("assets/logo.png", png_bytes)

        result = await _call_get_skill_content(
            "valid-skill",
            resource_path="assets/logo.png",
            payload=payload,
        )

        assert result["status"] == "success"
        assert result["source_url"] == payload["url"]
        assert result["resource_path"] == "assets/logo.png"
        assert result["resource_type"] == "asset"
        assert result["content_type"] == "image/png"
        assert result["size_bytes"] == len(png_bytes)
        assert "content" not in result
        assert "content_base64" not in result
        assert "content_encoding" not in result

    async def test_include_binary_adds_base64(self):
        png_bytes = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
        payload = _binary_asset_payload("assets/logo.png", png_bytes)

        result = await _call_get_skill_content(
            "valid-skill",
            resource_path="assets/logo.png",
            payload=payload,
            include_binary=True,
        )

        assert result["content_encoding"] == "base64"
        assert base64.b64decode(result["content_base64"]) == png_bytes
        assert result["resource_type"] == "asset"
        assert result["source_url"] == payload["url"]
        assert "content" not in result

    async def test_unknown_extension_falls_back_to_octet_stream(self):
        payload = _binary_asset_payload("assets/blob.dat", b"\x00\x01")

        result = await _call_get_skill_content(
            "valid-skill",
            resource_path="assets/blob.dat",
            payload=payload,
        )

        assert result["content_type"] == "application/octet-stream"

    async def test_text_resource_keeps_content(self):
        result = await _call_get_skill_content(
            "valid-skill",
            resource_path="references/notes.md",
        )

        assert result["status"] == "success"
        assert result["content"] == "hello"
        assert result["resource_type"] == ""
        assert result["content_type"] == "text/markdown"
        assert "content_base64" not in result

    async def test_text_asset_returns_content_without_include_binary(self):
        payload = {
            "content": "# Report\n\nSummary: {{summary}}\n",
            "size_bytes": 31,
            "path": "assets/report-template.md",
            "type": "asset",
            "url": "https://raw.example.com/assets/report-template.md",
        }

        result = await _call_get_skill_content(
            "valid-skill",
            resource_path="assets/report-template.md",
            payload=payload,
        )

        assert result["status"] == "success"
        assert result["content"] == payload["content"]
        assert result["resource_type"] == "asset"
        assert result["source_url"] == payload["url"]
        assert result["size_bytes"] == 31
        assert result["content_type"] == "text/markdown"
        assert "content_base64" not in result

    async def test_text_resource_without_known_extension_defaults_to_text_plain(self):
        payload = {
            "content": "MIT License\n",
            "size_bytes": 12,
            "path": "assets/LICENSE",
            "type": "asset",
            "url": "https://raw.example.com/assets/LICENSE",
        }

        result = await _call_get_skill_content(
            "valid-skill",
            resource_path="assets/LICENSE",
            payload=payload,
        )

        assert result["status"] == "success"
        assert result["content_type"] == "text/plain"

    async def test_skill_md_fetch_has_no_content_type(self):
        result = await _call_get_skill_content("valid-skill")

        assert result["status"] == "success"
        assert "content_type" not in result
