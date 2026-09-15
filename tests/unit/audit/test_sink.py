"""Unit tests for the audit event sink's log-stream isolation.

`emit_audit_event` renders an audit record as JSON. Those records carry durable
IdP identifiers -- `principal_name` (a UPN/email), `canonical_id`, `subject`,
`object_id`, `tenant_id`, `app_id` -- and they must NOT reach the APPLICATION log
stream. The root logger's handlers (stdout, a `RotatingFileHandler`, and
`MongoDBLogHandler`, which writes the `application_logs` collection) have a
shorter retention (~1 day by default) and a wider access boundary (the admin
log-viewer API) than the audit collection the compliance documentation reasons
about.

Validates: Issue #1642 (audit identity claims), data-minimisation follow-up.
"""

import json
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.audit.events import ToolFilterAuditEvent
from registry.audit.models import (
    Identity,
    MCPRequest,
    MCPResponse,
    MCPServer,
    MCPServerAccessRecord,
    Request,
    TokenMintAuditRecord,
)
from registry.audit.service import AuditLogger
from registry.audit.sink import (
    _RECORD_HANDLER_NAME,
    _configure_record_logger,
    emit_audit_event,
)
from registry.common.log_redaction import REDACTED

RECORD_LOGGER_NAME = "registry.audit.records"

# Synthetic values that preserve the SHAPE of the real thing (Entra oid/tid are
# GUIDs, canonical_id is `oid@tid`, principal_name is a UPN).
CLAIM_VALUES = {
    "subject": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "canonical_id": "11111111-2222-3333-4444-555555555555@66666666-7777-8888-9999-000000000000",
    "principal_name": "alice@example.com",
    "object_id": "11111111-2222-3333-4444-555555555555",
    "tenant_id": "66666666-7777-8888-9999-000000000000",
    "app_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
}


def make_mint_record(**overrides) -> TokenMintAuditRecord:
    """A token_mint record: the claims are TOP LEVEL on this stream."""
    fields = {
        "request_id": "req-mint-1",
        "username": "alice@example.com",
        "username_hash": "user_1234abcd",
        "auth_method": "oauth2",
        "provider": "entra",
        "internal_caller": "registry",
        "token_kind": "resource",
        "resource_type": "server",
        "resource_id": "fininfo",
        "token_path": "m2m",
        "outcome": "success",
        **CLAIM_VALUES,
    }
    fields.update(overrides)
    return TokenMintAuditRecord(**fields)


def make_mcp_record() -> MCPServerAccessRecord:
    """An mcp_access record: the claims are NESTED under `identity`."""
    return MCPServerAccessRecord(
        timestamp=datetime.now(UTC),
        request_id="req-mcp-1",
        identity=Identity(
            username="alice@example.com",
            auth_method="oauth2",
            credential_type="bearer_token",
            **CLAIM_VALUES,
        ),
        request=Request(method="POST", path="/outlook/mcp", client_ip="10.0.0.1"),
        mcp_server=MCPServer(name="outlook", path="/outlook", proxy_target="http://outlook:8000"),
        mcp_request=MCPRequest(method="tools/call"),
        mcp_response=MCPResponse(status="success", duration_ms=12.5),
    )


class _Capture(logging.Handler):
    """Collects the formatted message of every record it is handed."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


def _is_audit_origin(record: logging.LogRecord) -> bool:
    """Could this record carry an audit body?

    Only the sink's own logger tree (`registry.*`) and writes made straight to
    the root logger (`logging.info(...)`, whose records are named "root") can.
    Third-party loggers are excluded on purpose: raising the root level to DEBUG
    below also un-gates their DEBUG output, and pymongo's topology monitor logs
    server heartbeats from a background thread for as long as any client in the
    process is alive, which would make this capture non-deterministic.
    """
    return record.name == "root" or record.name.split(".", 1)[0] == "registry"


@pytest.fixture
def app_log_stream():
    """Everything that reaches a ROOT handler, i.e. the application log stream.

    stdout, the rotating file and `MongoDBLogHandler` are all root handlers, so
    a record that reaches this fixture would reach all three in production.
    """
    root = logging.getLogger()
    handler = _Capture()
    handler.addFilter(_is_audit_origin)
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


@pytest.fixture
def audit_stream():
    """The dedicated audit-record stream (the sink's own handler chain)."""
    record_logger = logging.getLogger(RECORD_LOGGER_NAME)
    handler = _Capture()
    record_logger.addHandler(handler)
    try:
        yield handler
    finally:
        record_logger.removeHandler(handler)


class TestClaimValuesNeverReachTheApplicationLogStream:
    """The BLOCKER: six durable identifiers were being copied into sinks with a
    different retention and a different audience than the audit collection."""

    def test_no_claim_value_reaches_a_root_handler(self, app_log_stream, audit_stream):
        emit_audit_event(make_mint_record())

        assert app_log_stream.messages == [], (
            f"audit record body reached the application log stream: {app_log_stream.messages}"
        )
        for value in CLAIM_VALUES.values():
            assert value not in app_log_stream.text

    def test_nested_claim_values_do_not_reach_a_root_handler(self, app_log_stream, audit_stream):
        """The claims sit under `identity` on every stream except token_mint."""
        emit_audit_event(make_mcp_record())

        assert app_log_stream.messages == []
        for value in CLAIM_VALUES.values():
            assert value not in app_log_stream.text

    def test_record_logger_does_not_propagate(self, app_log_stream):
        """Fails the moment someone re-enables propagation on this logger."""
        record_logger = logging.getLogger(RECORD_LOGGER_NAME)

        assert record_logger.propagate is False

        record_logger.info("canary-line")
        assert app_log_stream.messages == []

    def test_claim_values_are_masked_in_the_record_stream_too(self, audit_stream):
        """stdout is scraped into the same store, so not even the dedicated
        handler may carry the values -- only the fact that they exist."""
        emit_audit_event(make_mint_record())

        payload = json.loads(audit_stream.messages[0])
        for field in CLAIM_VALUES:
            assert payload[field] == REDACTED, field

    def test_absent_claims_stay_null_rather_than_looking_present(self, audit_stream):
        """An IdP that omits the claims must not be reported as carrying them."""
        emit_audit_event(make_mint_record(**dict.fromkeys(CLAIM_VALUES, None)))

        payload = json.loads(audit_stream.messages[0])
        for field in CLAIM_VALUES:
            assert payload[field] is None, field

    def test_nested_claims_are_masked_at_depth(self, audit_stream):
        emit_audit_event(make_mcp_record())

        payload = json.loads(audit_stream.messages[0])
        for field in CLAIM_VALUES:
            assert payload["identity"][field] == REDACTED, field


class TestAuditStreamStillWorks:
    """The fix must not weaken the sink: it is the ONLY sink for the tool-filter
    event and the operator-facing copy of every other record."""

    def test_record_reaches_its_own_sink(self, audit_stream):
        emit_audit_event(make_mint_record())

        assert len(audit_stream.messages) == 1
        payload = json.loads(audit_stream.messages[0])
        assert payload["request_id"] == "req-mint-1"
        assert payload["log_type"] == "token_mint"
        assert payload["outcome"] == "success"
        # The readable display identity is NOT a claim column and stays verbatim,
        # so the stream remains usable for attribution.
        assert payload["username"] == "alice@example.com"

    def test_non_claim_fields_are_not_over_redacted(self, audit_stream):
        """REGRESSION: a generic substring redactor (`redact_mapping`) masks any
        key containing "token", which on this record means `token_kind` and
        `token_path` -- the answer to "what was minted, by which signing path".
        Masking them would gut the record while protecting nothing."""
        emit_audit_event(make_mint_record())

        payload = json.loads(audit_stream.messages[0])
        assert payload["token_kind"] == "resource"
        assert payload["token_path"] == "m2m"
        assert payload["resource_id"] == "fininfo"
        assert payload["username_hash"] == "user_1234abcd"

    def test_tool_filter_event_is_emitted_verbatim(self, audit_stream):
        """This event has no durable sink at all; nothing about it may be lost."""
        emit_audit_event(
            ToolFilterAuditEvent(
                username="alice@example.com",
                endpoint="mcp_tools_list",
                server_name="outlook",
                pruned_count=2,
                kept_count=1,
                pruned_tool_names=["send_mail", "delete_mail"],
                user_scopes=["outlook-read"],
            )
        )

        payload = json.loads(audit_stream.messages[0])
        assert payload["username"] == "alice@example.com"
        assert payload["pruned_tool_names"] == ["send_mail", "delete_mail"]
        assert payload["user_scopes"] == ["outlook-read"]
        assert payload["pruned_count"] == 2

    def test_audit_module_loggers_still_reach_the_application_log_stream(self, app_log_stream):
        """REGRESSION: `registry.audit.service` emits the CRITICAL 'AUDIT RECORD
        DROPPED' line. Making `registry.audit` itself terminal would have
        stranded that line (and every other audit module log) inside the sink's
        handler, which is why the record stream is a CHILD logger."""
        logging.getLogger("registry.audit.service").critical("AUDIT RECORD DROPPED: canary")

        assert any("AUDIT RECORD DROPPED: canary" in m for m in app_log_stream.messages)

    def test_dedicated_handler_is_installed_exactly_once(self):
        """A re-import or a re-run of logging setup must not double every line."""
        for _ in range(3):
            _configure_record_logger()

        installed = [
            handler
            for handler in logging.getLogger(RECORD_LOGGER_NAME).handlers
            if handler.get_name() == _RECORD_HANDLER_NAME
        ]
        assert len(installed) == 1

    def test_emission_failure_is_swallowed_and_surfaced_to_operators(self, app_log_stream):
        """A broken sink must not break the request path, but the fault itself is
        an application error and belongs in the application log stream."""
        broken = MagicMock()
        broken.model_dump.side_effect = RuntimeError("dump exploded")

        emit_audit_event(broken)

        assert any("emit_audit_event failed" in m for m in app_log_stream.messages)


class TestDurablePersistenceIsUnaffected:
    """The durable store keeps FULL fidelity -- the masking is a log-stream
    concern only."""

    async def test_durable_sink_receives_the_claim_values(self):
        repository = AsyncMock()
        audit_logger = AuditLogger(
            stream_name="token-mint",
            mongodb_enabled=True,
            audit_repository=repository,
        )
        record = make_mint_record()

        await audit_logger.log_event(record)

        repository.insert.assert_awaited_once_with(record)
        persisted = repository.insert.await_args.args[0]
        for field, value in CLAIM_VALUES.items():
            assert getattr(persisted, field) == value, field
