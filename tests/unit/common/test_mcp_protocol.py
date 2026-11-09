"""Tests for the shared MCP 2026-07-28 helpers in registry/common/mcp_protocol.py."""

import pytest

from registry.common.mcp_protocol import (
    META_PROTOCOL_VERSION_KEY,
    NO_REUSE_TTL_MS,
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
