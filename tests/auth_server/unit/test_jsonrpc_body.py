"""Tests for strict JSON-RPC body parsing (auth_server.jsonrpc_body).

The gateway reads ``method`` and ``params.name`` from a request body to decide
whether a call is allowed, then forwards the same bytes to the backend. Some
backend JSON decoders match object keys without regard to case, so a body with
two spellings of one member can read one way to the gateway and another to the
backend. ``parse_jsonrpc_body`` refuses any body with more than one reading so
the two sides cannot diverge.

Bodies are written as raw strings because ``json.dumps`` cannot emit a repeated
key.
"""

import json

import pytest

from auth_server.jsonrpc_body import (
    AmbiguousJsonRpcBodyError,
    parse_jsonrpc_body,
)


class TestUnambiguousBodiesAccepted:
    """Bodies with a single reading decode normally."""

    def test_plain_tools_call(self):
        body = '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo"}}'
        result = parse_jsonrpc_body(body)
        assert result["method"] == "tools/call"
        assert result["params"]["name"] == "echo"

    def test_bytes_input(self):
        body = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
        assert parse_jsonrpc_body(body)["method"] == "ping"

    def test_arguments_keys_differing_only_by_case_allowed(self):
        """Tool arguments are opaque to the gateway, so a case-only pair there
        is not an authorization ambiguity and must still be accepted."""
        body = (
            '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
            '{"name":"echo","arguments":{"Content-Type":"a","content-type":"b"}}}'
        )
        result = parse_jsonrpc_body(body)
        assert result["params"]["arguments"]["Content-Type"] == "a"
        assert result["params"]["arguments"]["content-type"] == "b"

    def test_batch_of_valid_messages(self):
        body = (
            '[{"jsonrpc":"2.0","id":1,"method":"ping"},'
            '{"jsonrpc":"2.0","id":2,"method":"tools/list"}]'
        )
        result = parse_jsonrpc_body(body)
        assert [m["method"] for m in result] == ["ping", "tools/list"]


class TestDuplicateKeysRejected:
    """An exactly repeated key, at any depth, is refused."""

    def test_repeated_envelope_key(self):
        body = '{"jsonrpc":"2.0","id":1,"method":"tools/list","method":"tools/call"}'
        with pytest.raises(AmbiguousJsonRpcBodyError):
            parse_jsonrpc_body(body)

    def test_repeated_params_name(self):
        body = (
            '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
            '{"name":"echo","name":"restricted"}}'
        )
        with pytest.raises(AmbiguousJsonRpcBodyError):
            parse_jsonrpc_body(body)

    def test_repeated_key_inside_arguments(self):
        body = (
            '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
            '{"name":"echo","arguments":{"x":1,"x":2}}}'
        )
        with pytest.raises(AmbiguousJsonRpcBodyError):
            parse_jsonrpc_body(body)


class TestCaseCollidingKeysRejected:
    """A key a case-insensitive decoder would read as a gateway-read member is
    refused from the envelope and from params."""

    def test_envelope_method_case_variant(self):
        body = (
            '{"jsonrpc":"2.0","id":1,"method":"tools/list",'
            '"Method":"tools/call","params":{"name":"echo"}}'
        )
        with pytest.raises(AmbiguousJsonRpcBodyError):
            parse_jsonrpc_body(body)

    def test_params_name_case_variant(self):
        body = (
            '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
            '{"name":"echo","Name":"restricted"}}'
        )
        with pytest.raises(AmbiguousJsonRpcBodyError):
            parse_jsonrpc_body(body)

    def test_non_ascii_long_s_fold_rejected(self):
        """str.casefold folds U+017F to 's', matching Go's fold, so 'paramſ'
        reads as 'params' to such a decoder and must be refused."""
        body = '{"jsonrpc":"2.0","id":1,"method":"ping","paramſ":{"name":"x"}}'
        with pytest.raises(AmbiguousJsonRpcBodyError):
            parse_jsonrpc_body(body)


class TestInvalidJsonStillRaisesStandardErrors:
    """Non-JSON and non-UTF-8 bodies raise the usual errors, not the custom one."""

    def test_not_json(self):
        with pytest.raises(json.JSONDecodeError):
            parse_jsonrpc_body("{not json")

    def test_not_utf8(self):
        with pytest.raises(UnicodeDecodeError):
            parse_jsonrpc_body(b"\xff\xfe")


# A body that reads as the allowed tool to the gateway but as the withheld tool
# to a case-insensitive backend. Written as raw bytes (json.dumps cannot emit
# the colliding key).
AMBIGUOUS_TOOLS_CALL = (
    b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
    b'{"name":"echo","Name":"restricted","arguments":{}}}'
)


class TestForwardedBodyCheckRefusesAmbiguous:
    """_authorize_forwarded_mcp_body refuses an ambiguous body with 403, before
    any scope decision, so it cannot be mistaken for an ordinary denial."""

    @pytest.mark.asyncio
    async def test_ambiguous_body_rejected_even_with_permissive_scope(
        self, mock_scope_repository_with_data
    ):
        from unittest.mock import patch

        from fastapi import HTTPException

        from auth_server.server import _authorize_forwarded_mcp_body

        # write:servers would allow tools/call on the allowed reading; the body
        # is still refused because it has two readings.
        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _authorize_forwarded_mcp_body(
                    "test-server",
                    AMBIGUOUS_TOOLS_CALL,
                    ["write:servers"],
                )
        assert exc_info.value.status_code == 403


class TestValidateRefusesAmbiguous:
    """/validate fails closed with 403 on an ambiguous X-Body. 403 (not the 400
    the unparseable path uses) so nginx auth_request forwards it cleanly instead
    of collapsing it to a 500."""

    def test_ambiguous_x_body_fails_closed(
        self,
        auth_env_vars,
        mock_cognito_provider,
        mock_scope_repository_with_data,
    ):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        import auth_server.server as server_module

        with patch(
            "auth_server.server.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            with patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ):
                client = TestClient(server_module.app)
                response = client.get(
                    "/validate",
                    headers={
                        "Authorization": "Bearer test-token",
                        "X-Original-URL": "https://example.com/test-server/mcp",
                        "X-Body": AMBIGUOUS_TOOLS_CALL.decode("utf-8"),
                    },
                )
        assert response.status_code == 403
