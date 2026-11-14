"""Tests for OpenAPI spec fetching and checks (issue #2022)."""

import httpx
import pytest

from registry.services import openapi_spec_checks
from registry.services.openapi_errors import OpenApiServerError

VALID_YAML: bytes = b"""
openapi: 3.0.4
info: {title: Items, version: "2.1"}
paths:
  /items:
    get: {operationId: listItems, responses: {"200": {description: ok}}}
x-released: 2024-01-01
"""


def _code(
    raw: bytes,
) -> str:
    with pytest.raises(OpenApiServerError) as info:
        openapi_spec_checks.parse_spec(raw)
    return info.value.code


class TestParseSpec:
    """What parse_spec accepts and refuses."""

    def test_valid_yaml_parses_and_dates_become_strings(self):
        fetched = openapi_spec_checks.parse_spec(VALID_YAML)

        assert (fetched.title, fetched.version, fetched.openapi_version) == (
            "Items",
            "2.1",
            "3.0.4",
        )
        assert fetched.spec["x-released"] == "2024-01-01"
        assert len(fetched.sha256) == 64

    def test_json_parses(self):
        raw = b'{"openapi": "3.1.0", "info": {"title": "J"}, "paths": {"/a": {}}}'

        assert openapi_spec_checks.parse_spec(raw).title == "J"

    def test_yaml_alias_rejected(self):
        raw = b"openapi: 3.0.0\npaths: {/a: {}}\nx: &a [1, 2]\ny: *a\n"

        assert _code(raw) == "spec_invalid"

    def test_alias_bomb_rejected_quickly(self):
        lines = [
            "openapi: 3.0.0",
            "paths: {/a: {}}",
            'a0: &a0 ["x","x","x","x","x","x","x","x","x"]',
        ]
        for i in range(1, 9):
            lines.append(
                f"a{i}: &a{i} [*a{i - 1},*a{i - 1},*a{i - 1},*a{i - 1},*a{i - 1},*a{i - 1}]"
            )

        assert _code("\n".join(lines).encode()) == "spec_invalid"

    def test_swagger2_rejected_with_conversion_hint(self):
        with pytest.raises(OpenApiServerError, match="convert it to OpenAPI 3"):
            openapi_spec_checks.parse_spec(b'{"swagger": "2.0", "paths": {"/a": {}}}')

    @pytest.mark.parametrize(
        "raw",
        [
            b"\xff\xfe not utf-8",
            b"[1, 2, 3]",
            b"openapi: 2.0\npaths: {/a: {}}",
            b"openapi: 3.0.0\npaths: {}",
            b"{bad",
        ],
    )
    def test_invalid_documents_rejected(self, raw):
        assert _code(raw) == "spec_invalid"

    def test_deep_nesting_rejected(self):
        raw = b"openapi: 3.0.0\npaths: {/a: {}}\nx: " + b"[" * 5000 + b"]" * 5000

        assert _code(raw) == "spec_invalid"


class TestExternalRefs:
    """Only local '#/...' references are accepted, whatever the conversion library does."""

    BASE: str = "openapi: 3.0.3\ninfo: {title: t, version: '1'}\npaths:\n  /a:\n    get:\n      responses: {'200': {description: ok}}\n"

    @pytest.mark.parametrize(
        "ref",
        [
            "http://169.254.169.254/latest/meta-data/",
            "https://example.com/defs.json#/A",
            "file:///etc/passwd",
            "other.json#/A",
            "//example.com/defs.json",
        ],
    )
    def test_non_local_ref_rejected(self, ref):
        raw = (
            self.BASE
            + f"components:\n  schemas:\n    A:\n      properties:\n        p: {{$ref: '{ref}'}}\n"
        ).encode()

        with pytest.raises(OpenApiServerError) as info:
            openapi_spec_checks.parse_spec(raw)

        assert info.value.code == "spec_invalid"
        assert info.value.extra["refs"] == [ref]

    def test_ref_in_a_list_is_found(self):
        raw = (
            self.BASE
            + "components:\n  schemas:\n    A: {allOf: [{$ref: 'https://evil.example/x.json'}]}\n"
        ).encode()

        with pytest.raises(OpenApiServerError, match="not local"):
            openapi_spec_checks.parse_spec(raw)

    def test_non_string_ref_rejected(self):
        raw = (self.BASE + "components:\n  schemas:\n    A: {$ref: 5}\n").encode()

        with pytest.raises(OpenApiServerError):
            openapi_spec_checks.parse_spec(raw)

    def test_local_refs_accepted(self):
        raw = (
            self.BASE
            + "components:\n  schemas:\n    A: {type: object}\n    B: {$ref: '#/components/schemas/A'}\n"
        ).encode()

        assert (
            openapi_spec_checks.parse_spec(raw).spec["components"]["schemas"]["B"]["$ref"]
            == "#/components/schemas/A"
        )

    def test_property_named_ref_inside_properties_is_checked_too(self):
        """A schema property literally called "$ref" is rare; refusing it fails closed."""
        raw = (
            self.BASE + "components:\n  schemas:\n    A: {properties: {$ref: {type: string}}}\n"
        ).encode()

        with pytest.raises(OpenApiServerError):
            openapi_spec_checks.parse_spec(raw)


class TestToolChecks:
    """Tool count and tool name checks."""

    def test_too_many_tools_lists_tags(self):
        with pytest.raises(OpenApiServerError) as info:
            openapi_spec_checks.check_tool_count(["a", "b", "c"], {"pet": 2, "store": 1}, 2)

        body = info.value.to_body()
        assert (body["error"], body["count"], body["limit"], body["tags"]) == (
            "too_many_tools",
            3,
            2,
            {"pet": 2, "store": 1},
        )

    def test_tool_count_at_limit_passes(self):
        openapi_spec_checks.check_tool_count(["a", "b"], {}, 2)

    @pytest.mark.parametrize("names", [["x" * 65], ["has space"], ["dup", "dup"]])
    def test_bad_tool_names_rejected(self, names):
        with pytest.raises(OpenApiServerError) as info:
            openapi_spec_checks.check_tool_names(names)

        assert info.value.code == "tool_name_invalid"
        assert info.value.extra["names"]


class TestFetchSpec:
    """Size cap, status handling and redirects, with a mock transport in place of the network."""

    @pytest.fixture
    def serve(self, monkeypatch):
        def _install(
            response: httpx.Response,
        ) -> None:
            def _client(**kwargs):
                return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response))

            monkeypatch.setattr(openapi_spec_checks, "guarded_async_client", _client)

        return _install

    async def test_returns_bytes_and_content_type(self, serve):
        serve(httpx.Response(200, content=VALID_YAML, headers={"content-type": "application/yaml"}))

        raw, content_type = await openapi_spec_checks.fetch_spec_bytes(
            "https://api.example.com/spec", 10_000
        )

        assert (raw, content_type) == (VALID_YAML, "application/yaml")

    async def test_over_cap_is_spec_too_large(self, serve):
        serve(httpx.Response(200, content=b"x" * 5000))

        with pytest.raises(OpenApiServerError) as info:
            await openapi_spec_checks.fetch_spec_bytes("https://api.example.com/spec", 4096)

        assert info.value.code == "spec_too_large"

    @pytest.mark.parametrize("status", [301, 404, 500])
    async def test_non_200_is_fetch_failed(self, serve, status):
        serve(httpx.Response(status, headers={"location": "http://169.254.169.254/"}))

        with pytest.raises(OpenApiServerError) as info:
            await openapi_spec_checks.fetch_spec_bytes("https://api.example.com/spec", 4096)

        assert (info.value.status_code, info.value.code) == (502, "spec_fetch_failed")

    async def test_metadata_address_blocked_by_real_guard(self):
        with pytest.raises(OpenApiServerError) as info:
            await openapi_spec_checks.fetch_spec_bytes(
                "http://169.254.169.254/latest/meta-data/", 4096
            )

        assert info.value.code == "url_blocked"

    def test_validate_target_url_blocks_loopback(self):
        with pytest.raises(OpenApiServerError) as info:
            openapi_spec_checks.validate_target_url("http://127.0.0.1:8080/api", "base_url")

        assert (info.value.code, info.value.extra["field"]) == ("url_blocked", "base_url")
