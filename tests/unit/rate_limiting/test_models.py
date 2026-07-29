"""Unit tests for rate-limit Pydantic models and their validation."""

import pytest
from pydantic import ValidationError

from registry.rate_limiting.models import RateLimitDecision, RateLimitDefinition


class TestRateLimitDefinition:
    """Tests for RateLimitDefinition validation and id construction."""

    def test_valid_caller_group_definition(self):
        """A caller/group definition with per-caller-type limits is accepted."""
        d = RateLimitDefinition(
            axis="caller",
            entity_type="group",
            name="developers",
            user_max_requests=100,
            agent_max_requests=50,
            window_seconds=60,
        )
        assert d.build_id() == "caller:group:developers:60"
        assert d.limit_for_caller_type("user") == 100
        assert d.limit_for_caller_type("agent") == 50

    def test_caller_group_requires_at_least_one_limit(self):
        """A group definition with neither user nor agent limit is rejected."""
        with pytest.raises(ValidationError, match="must set user_max_requests"):
            RateLimitDefinition(
                axis="caller", entity_type="group", name="developers", window_seconds=60
            )

    def test_caller_group_allows_user_only(self):
        """A group may set just the user limit (agent unset => agents ungated by it)."""
        d = RateLimitDefinition(
            axis="caller", entity_type="group", name="devs", user_max_requests=30
        )
        assert d.limit_for_caller_type("user") == 30
        assert d.limit_for_caller_type("agent") is None

    def test_valid_target_mcp_server_definition(self):
        """A target/mcp_server definition is accepted."""
        d = RateLimitDefinition(
            axis="target",
            entity_type="mcp_server",
            name="mcpgw",
            max_requests=500,
        )
        assert d.build_id() == "target:mcp_server:mcpgw:60"

    def test_valid_target_a2a_agent_definition(self):
        """An A2A agent target is a first-class entity type."""
        d = RateLimitDefinition(
            axis="target",
            entity_type="a2a_agent",
            name="booking-agent",
            max_requests=200,
            window_seconds=60,
        )
        assert d.entity_type == "a2a_agent"

    def test_valid_server_group_definition(self):
        """A server_group target with max_requests + non-empty members is accepted."""
        d = RateLimitDefinition(
            axis="target",
            entity_type="server_group",
            name="fragile-backends",
            max_requests=100,
            window_seconds=60,
            members=["airegistry-tools", "aws-kb"],
        )
        assert d.build_id() == "target:server_group:fragile-backends:60"
        assert d.members == ["airegistry-tools", "aws-kb"]

    def test_server_group_requires_members(self):
        """A server_group with no members (or empty list) is rejected."""
        with pytest.raises(ValidationError, match="non-empty members"):
            RateLimitDefinition(
                axis="target",
                entity_type="server_group",
                name="empty",
                max_requests=100,
            )
        with pytest.raises(ValidationError, match="non-empty members"):
            RateLimitDefinition(
                axis="target",
                entity_type="server_group",
                name="empty",
                max_requests=100,
                members=[],
            )

    def test_server_group_requires_max_requests(self):
        """A server_group still needs the single target max_requests."""
        with pytest.raises(ValidationError, match="must set max_requests"):
            RateLimitDefinition(
                axis="target",
                entity_type="server_group",
                name="grp",
                members=["mcpgw"],
            )

    def test_server_group_rejects_duplicate_and_blank_members(self):
        """Members must be clean: no duplicates, no blank strings."""
        with pytest.raises(ValidationError, match="duplicates"):
            RateLimitDefinition(
                axis="target",
                entity_type="server_group",
                name="dup",
                max_requests=100,
                members=["mcpgw", "mcpgw"],
            )
        with pytest.raises(ValidationError, match="non-empty server-path"):
            RateLimitDefinition(
                axis="target",
                entity_type="server_group",
                name="blank",
                max_requests=100,
                members=["mcpgw", "  "],
            )

    def test_single_target_rejects_members(self):
        """members is only valid for server_group; a plain mcp_server def rejects it."""
        with pytest.raises(ValidationError, match="only valid for a server_group"):
            RateLimitDefinition(
                axis="target",
                entity_type="mcp_server",
                name="mcpgw",
                max_requests=500,
                members=["mcpgw"],
            )

    def test_legacy_target_has_no_members(self):
        """A pre-existing mcp_server def (no members key) parses with members=None."""
        d = RateLimitDefinition(
            axis="target",
            entity_type="mcp_server",
            name="mcpgw",
            max_requests=500,
        )
        assert d.members is None

    def test_server_group_members_normalized(self):
        """Members with leading/trailing slashes normalize to the bare classifier name."""
        d = RateLimitDefinition(
            axis="target",
            entity_type="server_group",
            name="grp",
            max_requests=10,
            members=["/aws-kb", "airegistry-tools/", "/mcpgw/"],
        )
        assert d.members == ["aws-kb", "airegistry-tools", "mcpgw"]

    def test_server_group_members_dedup_after_normalization(self):
        """'aws-kb' and '/aws-kb' are the same member after normalization -> rejected as dup."""
        with pytest.raises(ValidationError, match="duplicates"):
            RateLimitDefinition(
                axis="target",
                entity_type="server_group",
                name="grp",
                max_requests=10,
                members=["aws-kb", "/aws-kb"],
            )

    def test_daily_volume_window_is_allowed(self):
        """A full-day window (volume cap) is within bounds."""
        d = RateLimitDefinition(
            axis="caller",
            entity_type="group",
            name="developers",
            user_max_requests=5000,
            window_seconds=86400,
        )
        assert d.window_seconds == 86400

    def test_caller_axis_rejects_target_entity_type(self):
        """entity_type must be in the allowlist for its axis (fail closed)."""
        with pytest.raises(ValidationError, match="invalid for axis"):
            RateLimitDefinition(
                axis="caller",
                entity_type="mcp_server",
                name="x",
                max_requests=1,
            )

    def test_invalid_axis_rejected(self):
        """An unknown axis is rejected."""
        with pytest.raises(ValidationError, match="axis must be one of"):
            RateLimitDefinition(
                axis="bogus",
                entity_type="group",
                name="x",
                max_requests=1,
            )

    def test_window_over_one_day_rejected(self):
        """window_seconds above 86400 is out of bounds."""
        with pytest.raises(ValidationError):
            RateLimitDefinition(
                axis="caller",
                entity_type="group",
                name="x",
                user_max_requests=1,
                window_seconds=86401,
            )

    def test_limit_must_be_positive(self):
        """A limit below 1 is rejected."""
        with pytest.raises(ValidationError):
            RateLimitDefinition(
                axis="caller",
                entity_type="group",
                name="x",
                user_max_requests=0,
            )

    def test_empty_name_rejected(self):
        """An empty name is rejected."""
        with pytest.raises(ValidationError):
            RateLimitDefinition(
                axis="caller",
                entity_type="group",
                name="",
                user_max_requests=1,
            )


class TestRateLimitDecision:
    """Tests for decision construction and header rendering."""

    def test_allow_has_no_deny_headers(self):
        """An allow decision is allowed and carries best-effort remaining."""
        d = RateLimitDecision.allow(remaining=7)
        assert d.allowed is True
        assert d.remaining == 7

    def test_deny_builds_rate_limit_headers(self):
        """A deny decision renders the standard 429 headers."""
        d = RateLimitDecision(
            allowed=False,
            axis="clr",
            entity_type="group",
            limit=5,
            remaining=0,
            reset_epoch=1000,
            retry_after=30,
        )
        headers = d.headers()
        # X-RateLimit-Throttled is the marker nginx auth_request uses to tell a
        # throttle-403 apart from a genuine authorization 403 (issue #295): the
        # throttle leaves /validate as a 403 and nginx rewrites it into a 429.
        assert headers["X-RateLimit-Throttled"] == "1"
        assert headers["X-RateLimit-Limit"] == "5"
        assert headers["X-RateLimit-Remaining"] == "0"
        assert headers["X-RateLimit-Reset"] == "1000"
        assert headers["Retry-After"] == "30"
        assert headers["Connection"] == "close"


class TestCallerTargetAndQuarantineModels:
    """Tests for the caller_target axis, the quarantine sentinel, and target subjects."""

    def test_caller_target_definition_id(self):
        """A caller_target group definition validates and builds its composite id."""
        from registry.rate_limiting.models import RateLimitDefinition

        d = RateLimitDefinition(
            axis="caller_target",
            entity_type="group",
            name="per-server-cap",
            user_max_requests=60,
            window_seconds=60,
        )
        assert d.build_id() == "caller_target:group:per-server-cap:60"

    def test_caller_target_requires_a_per_type_limit(self):
        from registry.rate_limiting.models import RateLimitDefinition

        with pytest.raises(ValidationError):
            RateLimitDefinition(axis="caller_target", entity_type="group", name="x")

    def test_quarantine_sentinel_valid(self):
        from registry.rate_limiting.models import RateLimitDefinition

        d = RateLimitDefinition(
            axis="quarantine",
            entity_type="group",
            name="quarantine-callers",
            scope="caller",
            window_seconds=1,
        )
        assert d.build_id() == "quarantine:group:quarantine-callers:1"

    def test_quarantine_rejects_rate(self):
        from registry.rate_limiting.models import RateLimitDefinition

        with pytest.raises(ValidationError):
            RateLimitDefinition(
                axis="quarantine",
                entity_type="group",
                name="quarantine-callers",
                scope="caller",
                user_max_requests=5,
                window_seconds=1,
            )

    def test_quarantine_requires_reserved_name(self):
        from registry.rate_limiting.models import RateLimitDefinition

        with pytest.raises(ValidationError):
            RateLimitDefinition(
                axis="quarantine", entity_type="group", name="not-reserved", scope="caller"
            )

    def test_reserved_name_rejected_on_rate_axis(self):
        """An operator cannot shadow a kill-switch group with a rate definition."""
        from registry.rate_limiting.models import RateLimitDefinition

        with pytest.raises(ValidationError):
            RateLimitDefinition(
                axis="caller",
                entity_type="group",
                name="quarantine-callers",
                user_max_requests=5,
            )

    def test_target_subject_membership(self):
        from registry.rate_limiting.models import QUARANTINE_TARGET_GROUP, RateLimitMembership

        m = RateLimitMembership(
            subject_type="server", subject="mcpgw", groups=[QUARANTINE_TARGET_GROUP]
        )
        assert m.build_id() == "server:mcpgw"

    def test_target_subject_only_quarantine_group(self):
        from registry.rate_limiting.models import RateLimitMembership

        with pytest.raises(ValidationError):
            RateLimitMembership(subject_type="server", subject="mcpgw", groups=["some-rate-group"])

    def test_caller_cannot_join_target_quarantine(self):
        from registry.rate_limiting.models import QUARANTINE_TARGET_GROUP, RateLimitMembership

        with pytest.raises(ValidationError):
            RateLimitMembership(
                subject_type="user", subject="alice", groups=[QUARANTINE_TARGET_GROUP]
            )

    def test_quarantine_deny_decision(self):
        from registry.rate_limiting.models import RateLimitDecision

        d = RateLimitDecision.quarantine_deny("clr", "group")
        assert d.allowed is False
        assert d.quarantined is True


class TestTargetSubjectNormalization:
    """Tests for the target-name normalizers (match the request classifier's forms)."""

    def test_normalize_server_name_strips_slashes(self):
        from registry.rate_limiting.models import normalize_server_name

        assert normalize_server_name("aws-kb") == "aws-kb"
        assert normalize_server_name("/aws-kb") == "aws-kb"
        assert normalize_server_name("/aws-kb/") == "aws-kb"
        # Only the first path segment (the classifier keys on path_parts[0]).
        assert normalize_server_name("/aws-kb/mcp") == "aws-kb"
        assert normalize_server_name("") == ""

    def test_normalize_agent_path_ensures_leading_slash(self):
        from registry.rate_limiting.models import normalize_agent_path

        assert normalize_agent_path("/flight-booking-agent") == "/flight-booking-agent"
        assert normalize_agent_path("flight-booking-agent") == "/flight-booking-agent"
        assert normalize_agent_path("/flight-booking-agent/") == "/flight-booking-agent"
        # Multi-segment agent paths are preserved (agents can be multi-segment).
        assert normalize_agent_path("agents/booking") == "/agents/booking"
        assert normalize_agent_path("") == ""

    def test_normalize_target_subject_by_type(self):
        from registry.rate_limiting.models import normalize_target_subject

        assert normalize_target_subject("server", "/aws-kb") == "aws-kb"
        assert normalize_target_subject("agent", "booking") == "/booking"
        # Caller subjects are identities, never slash-mangled.
        assert normalize_target_subject("user", "/alice") == "/alice"
        assert normalize_target_subject("client", "my-client") == "my-client"
