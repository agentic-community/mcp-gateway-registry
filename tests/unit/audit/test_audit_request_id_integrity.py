"""Security regression tests: audit records cannot be suppressed via a
client-chosen X-Request-ID.

Threat: the audit collection has a UNIQUE composite index on
``(request_id, log_type)``. If ``request_id`` were taken verbatim from the
client ``X-Request-ID`` header, an authenticated attacker could log a benign
request under a chosen id, then reuse the SAME id on a malicious request of the
same ``log_type`` — the second insert would collide with the unique index and be
silently dropped, leaving the malicious action with no audit trail.

The invariant these tests protect: the durable audit key is ALWAYS
server-generated, so two different requests carrying the same client
``X-Request-ID`` produce two distinct, durable audit records. The client value
is retained only as a sanitized, non-key correlation field.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymongo.errors import DuplicateKeyError
from starlette.datastructures import Headers

from registry.audit.middleware import AuditMiddleware
from registry.audit.models import Identity, RegistryApiAccessRecord, Request, Response
from registry.audit.request_id import (
    new_audit_request_id,
    sanitize_correlation_id,
)
from registry.repositories.audit_repository import DocumentDBAuditRepository


class TestServerControlledKey:
    """The unique audit key must be server-generated, never client-chosen."""

    def test_new_audit_request_id_is_unique_per_call(self):
        """Each mint returns a distinct id so distinct events never collide."""
        ids = {new_audit_request_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_new_audit_request_id_ignores_all_input(self):
        """The mint takes no client input — it cannot be steered to a value."""
        # A caller cannot pass a value; the signature accepts none. This is the
        # structural guarantee that no client header can become the key.
        import inspect

        assert list(inspect.signature(new_audit_request_id).parameters) == []


class TestCorrelationIdSanitization:
    """The retained client value is validated/bounded and never trusted."""

    def test_none_and_empty_fail_closed(self):
        assert sanitize_correlation_id(None) is None
        assert sanitize_correlation_id("") is None
        assert sanitize_correlation_id("   ") is None

    def test_valid_ids_are_accepted(self):
        # nginx $request_id (32 hex), a UUID, and a W3C-style traceparent token.
        assert sanitize_correlation_id("a1b2c3d4e5f60718293a4b5c6d7e8f90") == (
            "a1b2c3d4e5f60718293a4b5c6d7e8f90"
        )
        assert sanitize_correlation_id("550e8400-e29b-41d4-a716-446655440000") == (
            "550e8400-e29b-41d4-a716-446655440000"
        )
        assert sanitize_correlation_id("  trace-42:00  ") == "trace-42:00"

    def test_injection_and_control_chars_rejected(self):
        """Values that could carry an injection payload fail closed."""
        assert sanitize_correlation_id("id with spaces") is None
        assert sanitize_correlation_id("$(rm -rf /)") is None
        assert sanitize_correlation_id("a\nb") is None
        assert sanitize_correlation_id("<script>") is None
        assert sanitize_correlation_id("id;drop") is None

    def test_overlong_value_rejected(self):
        """Unbounded attacker input must not be stored."""
        assert sanitize_correlation_id("a" * 5000) is None

    def test_embedded_newline_rejected_trailing_stripped_never_stored_raw(self):
        """A CR/LF must never end up in the STORED value (log-injection guard).

        Two behaviours make this strip-independent:
          - An EMBEDDED newline (payload in the middle) fails closed to None,
            because ``re.fullmatch`` requires the WHOLE string to match — the
            classic ``$``-matches-before-trailing-``\\n`` weakness cannot slip a
            ``value\\nSet-Cookie: ...`` past the anchor.
          - A purely TRAILING newline/whitespace is stripped first, then the
            remaining token validates cleanly. That is safe: the stored value
            never contains the newline.
        The invariant asserted regardless of which branch fires: the returned
        value is either None or a newline-free token.
        """
        # Embedded control chars / injection payloads fail closed.
        assert sanitize_correlation_id("ab\ncd") is None
        assert sanitize_correlation_id("valid\nSet-Cookie: x=y") is None
        assert sanitize_correlation_id("a\rb") is None
        # Trailing whitespace/newline is stripped, then validated.
        assert sanitize_correlation_id("abc\n") == "abc"
        assert sanitize_correlation_id("  trace-1  \n") == "trace-1"
        # The core guarantee: nothing that survives ever carries a newline.
        for candidate in ("abc\n", "abc\r\n", "  trace-1  \n", "ab\ncd"):
            result = sanitize_correlation_id(candidate)
            assert result is None or ("\n" not in result and "\r" not in result)


class TestTokenMintCorrelationNeverStoredRaw:
    """The token-mint carrier (X-Correlation-ID -> /internal/tokens ->
    TokenMintAuditRecord) must never persist a raw, hostile client value.

    Invariant: hostile client correlation input is never persisted raw on the
    durable token-mint audit record. Proven at the two layers that enforce it:
    the sanitize call site (what the /api/tokens/generate and /internal/tokens
    paths run on the header/request value) and the model Field cap (the backstop
    for any future producer).
    """

    def test_hostile_header_sanitizes_to_none_at_call_site(self):
        """CR/LF, control chars, and over-cap length all fail closed to None.

        This is exactly what ``server_routes.py`` (the /api/tokens/generate
        forwarder) and ``auth_server.server`` (the /internal/tokens consumer)
        now apply to the client-supplied correlation value before it can reach
        ``TokenMintAuditRecord``.
        """
        assert sanitize_correlation_id("corr\nSet-Cookie: evil=1") is None
        assert sanitize_correlation_id("corr\r\ninjected") is None
        assert sanitize_correlation_id("has spaces and ;drop") is None
        assert sanitize_correlation_id("A" * 5000) is None

    def test_valid_header_survives_as_correlation_only(self):
        """A well-formed trace id is retained verbatim (stitching still works)."""
        assert sanitize_correlation_id("550e8400-e29b-41d4-a716-446655440000") == (
            "550e8400-e29b-41d4-a716-446655440000"
        )

    def test_model_cap_rejects_over_length_correlation(self):
        """Backstop: the record itself refuses an unbounded correlation value.

        Even if a future producer forgot to sanitize, the durable record cannot
        store the raw over-cap value — construction fails at the Field cap.
        """
        from pydantic import ValidationError

        from registry.audit.models import TokenMintAuditRecord

        with pytest.raises(ValidationError):
            TokenMintAuditRecord(
                request_id="server-minted-id",
                correlation_id="a" * 201,
                username_hash="user_abcd1234",
                auth_method="oauth2",
                internal_caller="registry",
                token_kind="user",
                token_path="self_signed",
                outcome="success",
            )

    def test_sanitized_none_is_a_valid_correlation_on_the_record(self):
        """After a hostile value fails closed to None, the record still builds."""
        from registry.audit.models import TokenMintAuditRecord

        record = TokenMintAuditRecord(
            request_id="server-minted-id",
            correlation_id=sanitize_correlation_id("corr\nSet-Cookie: evil=1"),
            username_hash="user_abcd1234",
            auth_method="oauth2",
            internal_caller="registry",
            token_kind="user",
            token_path="self_signed",
            outcome="success",
        )
        assert record.correlation_id is None
        assert record.request_id == "server-minted-id"


class TestMiddlewareDoesNotTrustClientHeader:
    """The middleware keys audit records on a server id, not X-Request-ID."""

    def _build_middleware(self) -> AuditMiddleware:
        return AuditMiddleware(app=MagicMock(), audit_logger=MagicMock())

    async def _capture_record(
        self,
        x_request_id: str,
    ) -> RegistryApiAccessRecord:
        """Drive dispatch() with a chosen X-Request-ID; return the built record."""
        middleware = self._build_middleware()
        captured: list = []

        async def _fake_log_event(record):
            captured.append(record)

        middleware.audit_logger.log_event = AsyncMock(side_effect=_fake_log_event)

        request = MagicMock()
        request.url.path = "/api/registry/servers"
        request.method = "POST"
        request.headers = Headers({"X-Request-ID": x_request_id})
        request.query_params = {}
        request.cookies = {}
        request.state = MagicMock(spec=[])  # no user_context/audit_action

        async def _call_next(_req):
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {}
            return resp

        with (
            patch.object(
                AuditMiddleware,
                "_extract_identity",
                new_callable=AsyncMock,
                return_value=Identity(
                    username="attacker",
                    auth_method="oauth2",
                    credential_type="bearer_token",
                ),
            ),
            patch("registry.audit.middleware.get_client_ip", return_value="127.0.0.1"),
            patch("registry.audit.middleware.resolve_instance_id", return_value="i-1"),
        ):
            await middleware.dispatch(request, _call_next)

        assert captured, "middleware did not emit an audit record"
        return captured[0]

    async def test_two_requests_same_client_id_get_distinct_keys(self):
        """ATTACK STOPPED: reusing an X-Request-ID does not merge two requests.

        Two separate requests carrying the same client header must produce two
        records with different (server-generated) request_id keys, so neither is
        dropped by the unique index.
        """
        chosen = "attacker-chosen-id-0000000000000000"
        record_a = await self._capture_record(chosen)
        record_b = await self._capture_record(chosen)

        # Distinct server keys -> both survive the unique (request_id, log_type).
        assert record_a.request_id != record_b.request_id
        # And neither key is the client-chosen value.
        assert record_a.request_id != chosen
        assert record_b.request_id != chosen

    async def test_client_id_kept_only_as_correlation(self):
        """The client value is retained (sanitized) as correlation, not the key."""
        record = await self._capture_record("a1b2c3d4e5f60718293a4b5c6d7e8f90")
        assert record.correlation_id == "a1b2c3d4e5f60718293a4b5c6d7e8f90"
        assert record.request_id != "a1b2c3d4e5f60718293a4b5c6d7e8f90"

    async def test_malicious_client_id_not_stored_even_as_correlation(self):
        """A hostile client value fails closed to no correlation id."""
        record = await self._capture_record("$(evil);drop")
        assert record.correlation_id is None
        assert record.request_id  # a server id was still minted


class TestUnexpectedDuplicateIsLoud:
    """A collision on a server key is an audit-integrity event, not routine."""

    async def test_duplicate_key_is_logged_critical_and_not_fatal(self, caplog):
        """DuplicateKeyError still returns True (never fail the request) but is
        surfaced at CRITICAL so the dropped record is alertable."""
        mock_collection = AsyncMock()
        mock_collection.insert_one.side_effect = DuplicateKeyError("dup")

        record = RegistryApiAccessRecord(
            timestamp=datetime.now(UTC),
            request_id="server-generated-id",
            identity=Identity(username="u", auth_method="oauth2", credential_type="bearer_token"),
            request=Request(method="GET", path="/api/test", client_ip="127.0.0.1"),
            response=Response(status_code=200, duration_ms=1.0),
        )

        with patch.object(
            DocumentDBAuditRepository, "_get_collection", return_value=mock_collection
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection
            with caplog.at_level("CRITICAL"):
                result = await repo.insert(record)

        assert result is True  # request path is never failed on an audit blip
        assert any(
            rec.levelname == "CRITICAL" and "AUDIT RECORD DROPPED" in rec.message
            for rec in caplog.records
        )
