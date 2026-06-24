"""Unit tests for auth_server._emit_token_mint_audit (#1308).

Verifies the metric labels, the audit record fields (including username
hashing), and the best-effort contract: a failure in the metric or audit sink
must never propagate out of the emit helper, so token minting is never broken
by observability.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auth_server.server import _emit_token_mint_audit, hash_username

pytestmark = [pytest.mark.unit, pytest.mark.auth]


_COMMON = {
    "request_id": "req-1",
    "correlation_id": "corr-1",
    "username": "alice@example.com",
    "auth_method": "oauth2",
    "provider": "keycloak",
    "internal_caller": "registry",
    "requested_scopes": ["mcp-servers-unrestricted/read"],
    "expires_in_seconds": 3600,
}


@patch("auth_server.server.get_audit_logger", return_value=None)
@patch("auth_server.server.emit_audit_event")
@patch("auth_server.server.token_mint_total")
async def test_success_increments_metric_with_labels(mock_metric, mock_emit, _mock_logger):
    await _emit_token_mint_audit(
        token_kind="user",
        resource_type=None,
        resource_id=None,
        token_path="self_signed",
        outcome="success",
        **_COMMON,
    )
    # resource_type None must collapse to the "none" label (not an empty string).
    mock_metric.add.assert_called_once_with(
        1,
        {
            "token_kind": "user",
            "resource_type": "none",
            "token_path": "self_signed",
            "outcome": "success",
        },
    )
    mock_emit.assert_called_once()


@patch("auth_server.server.get_audit_logger", return_value=None)
@patch("auth_server.server.emit_audit_event")
@patch("auth_server.server.token_mint_total")
async def test_resource_label_uses_value_when_set(mock_metric, _mock_emit, _mock_logger):
    await _emit_token_mint_audit(
        token_kind="resource",
        resource_type="server",
        resource_id="fininfo",
        token_path="self_signed",
        outcome="success",
        **_COMMON,
    )
    labels = mock_metric.add.call_args.args[1]
    assert labels["resource_type"] == "server"
    assert labels["token_kind"] == "resource"


@patch("auth_server.server.get_audit_logger", return_value=None)
@patch("auth_server.server.emit_audit_event")
@patch("auth_server.server.token_mint_total")
async def test_record_username_is_hashed_never_raw(_mock_metric, mock_emit, _mock_logger):
    await _emit_token_mint_audit(
        token_kind="user",
        resource_type=None,
        resource_id=None,
        token_path="m2m",
        outcome="success",
        **_COMMON,
    )
    record = mock_emit.call_args.args[0]
    assert record.username_hash == hash_username("alice@example.com")
    assert record.username_hash.startswith("user_")
    assert "alice@example.com" not in record.username_hash
    assert record.correlation_id == "corr-1"
    assert record.outcome == "success"


@patch("auth_server.server.emit_audit_event")
@patch("auth_server.server.token_mint_total")
async def test_record_written_to_audit_logger_when_present(mock_metric, mock_emit):
    audit_logger = MagicMock()
    audit_logger.log_event = AsyncMock()
    with patch("auth_server.server.get_audit_logger", return_value=audit_logger):
        await _emit_token_mint_audit(
            token_kind="user",
            resource_type=None,
            resource_id=None,
            token_path="self_signed",
            outcome="failure",
            failure_reason="rate_limited",
            **_COMMON,
        )
    audit_logger.log_event.assert_awaited_once()
    written = audit_logger.log_event.call_args.args[0]
    assert written.outcome == "failure"
    assert written.failure_reason == "rate_limited"


@patch("auth_server.server.get_audit_logger", return_value=None)
@patch("auth_server.server.emit_audit_event")
@patch("auth_server.server.token_mint_total")
async def test_metric_failure_is_swallowed_and_audit_still_emitted(
    mock_metric, mock_emit, _mock_logger
):
    # A broken metric backend must not stop the audit record nor raise.
    mock_metric.add.side_effect = RuntimeError("otel down")
    await _emit_token_mint_audit(
        token_kind="user",
        resource_type=None,
        resource_id=None,
        token_path="self_signed",
        outcome="success",
        **_COMMON,
    )
    mock_emit.assert_called_once()


@patch("auth_server.server.get_audit_logger", return_value=None)
@patch("auth_server.server.emit_audit_event")
@patch("auth_server.server.token_mint_total")
async def test_audit_sink_failure_is_swallowed(_mock_metric, mock_emit, _mock_logger):
    # A broken audit sink must be swallowed; the helper returns None, never raises.
    mock_emit.side_effect = RuntimeError("sink down")
    result = await _emit_token_mint_audit(
        token_kind="user",
        resource_type=None,
        resource_id=None,
        token_path="self_signed",
        outcome="success",
        **_COMMON,
    )
    assert result is None
