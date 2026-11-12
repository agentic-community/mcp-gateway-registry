"""Tests for the shared MCP 2026-07-28 helpers in registry/common/mcp_protocol.py."""

import pytest

from registry.common.mcp_protocol import (
    DENIED_BY_SCOPES,
    DENIED_NO_SCOPES,
    DENIED_ORIGIN,
    GATEWAY_ACCESS_DENIED,
    MAX_DENIED_NAME_LENGTH,
    META_PROTOCOL_VERSION_KEY,
    NO_REUSE_TTL_MS,
    access_denied_error,
    is_modern_request,
    request_meta,
    shape_result,
)

MODERN: dict = {"params": {"_meta": {META_PROTOCOL_VERSION_KEY: "2026-07-28"}}}
LEGACY: dict = {"params": {"name": "search"}}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (MODERN, True),
        ({"params": {"_meta": {META_PROTOCOL_VERSION_KEY: "2027-01-01"}}}, True),
        ({"params": {"_meta": {META_PROTOCOL_VERSION_KEY: "2025-11-25"}}}, False),
        (LEGACY, False),
        ({"params": {"_meta": "not a dict"}}, False),
        (None, False),
        ("not a dict", False),
    ],
)
def test_is_modern_request(payload, expected):
    assert is_modern_request(payload) is expected


def test_request_meta_is_empty_for_malformed_payloads():
    assert request_meta({"params": None}) == {}
    assert request_meta([]) == {}


@pytest.mark.parametrize(
    "method",
    ["tools/list", "prompts/list", "resources/list", "resources/templates/list", "resources/read"],
)
def test_cacheable_results_get_all_three_fields(method):
    shaped = shape_result({"items": []}, method, MODERN)

    assert shaped["resultType"] == "complete"
    assert shaped["cacheScope"] == "private"
    assert shaped["ttlMs"] == NO_REUSE_TTL_MS


def test_other_results_get_result_type_only():
    shaped = shape_result({"content": []}, "tools/call", MODERN)

    assert shaped == {"content": [], "resultType": "complete"}


def test_existing_fields_are_kept():
    result = {"tools": [], "resultType": "complete", "cacheScope": "public", "ttlMs": 5000}

    assert shape_result(result, "tools/list", MODERN) == result


def test_explicit_ttl_is_used():
    assert shape_result({"tools": []}, "tools/list", MODERN, ttl_ms=60000)["ttlMs"] == 60000


def test_legacy_requests_are_unchanged():
    result = {"tools": []}

    assert shape_result(result, "tools/list", LEGACY) is result


def test_the_input_result_is_not_modified():
    result = {"tools": []}

    shape_result(result, "tools/list", MODERN)

    assert result == {"tools": []}


CALL: dict = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "rm"}}


def test_denied_error_names_method_tool_and_server():
    error = access_denied_error(CALL, DENIED_BY_SCOPES, "tools/call", "rm", "/github/")

    assert error["id"] == 7
    assert error["error"]["code"] == GATEWAY_ACCESS_DENIED
    assert error["error"]["message"] == (
        'Access denied: tools/call for tool "rm" on /github/ is not permitted for your account.'
    )
    assert error["error"]["data"] == {
        "reason": DENIED_BY_SCOPES,
        "method": "tools/call",
        "server": "/github/",
        "tool": "rm",
    }


def test_denied_error_without_a_tool_names_the_method():
    error = access_denied_error(CALL, DENIED_BY_SCOPES, "resources/read", None, "/github/")

    assert error["error"]["message"] == (
        "Access denied: resources/read on /github/ is not permitted for your account."
    )
    assert "tool" not in error["error"]["data"]


def test_no_scopes_and_origin_reasons_have_their_own_messages():
    no_scopes = access_denied_error(CALL, DENIED_NO_SCOPES, "tools/list", None, "/a/")
    origin = access_denied_error(CALL, DENIED_ORIGIN, "tools/list", None, "/a/")

    assert "no scopes" in no_scopes["error"]["message"]
    assert "bearer token" in origin["error"]["message"]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "not a dict",
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"id": 1, "method": "tools/list"},
    ],
)
def test_denied_error_is_none_without_a_json_rpc_request(payload):
    assert access_denied_error(payload, DENIED_BY_SCOPES, "tools/list", None, "/a/") is None


@pytest.mark.parametrize("request_id", [{"x": 1}, [1], True, 1.5])
def test_denied_error_drops_an_invalid_id(request_id):
    payload = {"jsonrpc": "2.0", "id": request_id, "method": "tools/list"}

    assert access_denied_error(payload, DENIED_BY_SCOPES, "tools/list", None, "/a/")["id"] is None


def test_denied_error_truncates_a_long_tool_name():
    error = access_denied_error(CALL, DENIED_BY_SCOPES, "tools/call", "x" * 500, "/a/")

    assert len(error["error"]["data"]["tool"]) == MAX_DENIED_NAME_LENGTH
