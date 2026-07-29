"""Unit tests for registry/api/rate_limit_routes.py (admin CRUD for rate limits)."""

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


@pytest.fixture
def admin_user_context() -> dict[str, Any]:
    """Admin user context."""
    return {"username": "admin", "is_admin": True, "groups": ["mcp-registry-admin"]}


@pytest.fixture
def non_admin_user_context() -> dict[str, Any]:
    """Non-admin user context."""
    return {"username": "user", "is_admin": False, "groups": ["mcp-users"]}


@pytest.fixture
def mock_auth_admin(admin_user_context):
    """Override nginx_proxied_auth with an admin user."""
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[nginx_proxied_auth] = lambda: admin_user_context
    yield admin_user_context
    app.dependency_overrides.clear()


@pytest.fixture
def mock_auth_regular(non_admin_user_context):
    """Override nginx_proxied_auth with a non-admin user."""
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[nginx_proxied_auth] = lambda: non_admin_user_context
    yield non_admin_user_context
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    """TestClient over the real app."""
    from registry.main import app

    return TestClient(app)


@pytest.fixture
def mock_repository():
    """Patch the routes' repository singleton with an AsyncMock."""
    repo = AsyncMock()
    with patch("registry.api.rate_limit_routes._get_repository", return_value=repo):
        yield repo


@pytest.fixture
def mock_memberships_repository():
    """Patch the routes' memberships repository singleton with an AsyncMock."""
    repo = AsyncMock()
    with patch("registry.api.rate_limit_routes._get_memberships_repository", return_value=repo):
        yield repo


@pytest.mark.unit
class TestPutRateLimit:
    """Tests for PUT /api/rate-limits/{id}."""

    def test_put_valid_definition(self, client, mock_auth_admin, mock_repository):
        """A valid definition whose body matches the URL id is stored."""
        from registry.rate_limiting.models import RateLimitDefinition

        definition = RateLimitDefinition(
            axis="caller",
            entity_type="group",
            name="developers",
            user_max_requests=25,
            window_seconds=60,
        )
        mock_repository.upsert.return_value = definition

        body = {
            "axis": "caller",
            "entity_type": "group",
            "name": "developers",
            "user_max_requests": 25,
            "window_seconds": 60,
        }
        resp = client.put("/api/rate-limits/caller:group:developers:60", json=body)
        assert resp.status_code == 200
        assert resp.json()["user_max_requests"] == 25

    def test_put_valid_server_group(self, client, mock_auth_admin, mock_repository):
        """A server_group target with members is stored and echoes members back."""
        from registry.rate_limiting.models import RateLimitDefinition

        definition = RateLimitDefinition(
            axis="target",
            entity_type="server_group",
            name="fragile",
            max_requests=100,
            window_seconds=60,
            members=["airegistry-tools", "aws-kb"],
        )
        mock_repository.upsert.return_value = definition
        body = {
            "axis": "target",
            "entity_type": "server_group",
            "name": "fragile",
            "max_requests": 100,
            "window_seconds": 60,
            "members": ["airegistry-tools", "aws-kb"],
        }
        resp = client.put("/api/rate-limits/target:server_group:fragile:60", json=body)
        assert resp.status_code == 200
        assert resp.json()["members"] == ["airegistry-tools", "aws-kb"]

    def test_put_server_group_without_members_rejected(
        self, client, mock_auth_admin, mock_repository
    ):
        """A server_group with no members is a 400 (model validation)."""
        body = {
            "axis": "target",
            "entity_type": "server_group",
            "name": "empty",
            "max_requests": 100,
            "window_seconds": 60,
        }
        resp = client.put("/api/rate-limits/target:server_group:empty:60", json=body)
        assert resp.status_code == 400

    def test_put_single_target_with_members_rejected(
        self, client, mock_auth_admin, mock_repository
    ):
        """A plain mcp_server def carrying members is a 400 (members is server_group-only)."""
        body = {
            "axis": "target",
            "entity_type": "mcp_server",
            "name": "mcpgw",
            "max_requests": 500,
            "window_seconds": 60,
            "members": ["mcpgw"],
        }
        resp = client.put("/api/rate-limits/target:mcp_server:mcpgw:60", json=body)
        assert resp.status_code == 400

    def test_put_url_id_mismatch_rejected(self, client, mock_auth_admin, mock_repository):
        """A body that builds a different _id than the URL id is a 400."""
        body = {
            "axis": "caller",
            "entity_type": "group",
            "name": "developers",
            "user_max_requests": 25,
            "window_seconds": 60,
        }
        # URL says window 30, body says 60 -> mismatch.
        resp = client.put("/api/rate-limits/caller:group:developers:30", json=body)
        assert resp.status_code == 400
        assert "does not match" in resp.json()["detail"]

    def test_put_invalid_axis_rejected(self, client, mock_auth_admin, mock_repository):
        """An invalid axis fails model validation with a 400."""
        body = {
            "axis": "sideways",
            "entity_type": "group",
            "name": "x",
            "user_max_requests": 25,
            "window_seconds": 60,
        }
        resp = client.put("/api/rate-limits/sideways:group:x:60", json=body)
        assert resp.status_code == 400

    def test_put_unenforced_entity_type_rejected(self, client, mock_auth_admin, mock_repository):
        """A modeled-but-unenforced entity type (mcp_tool) is rejected with a clear 400."""
        body = {
            "axis": "target",
            "entity_type": "mcp_tool",
            "name": "mcpgw:search",
            "max_requests": 5,
            "window_seconds": 60,
        }
        resp = client.put("/api/rate-limits/target:mcp_tool:mcpgw:search:60", json=body)
        assert resp.status_code == 400
        assert "not enforced" in resp.json()["detail"]

    def test_put_user_below_floor_rejected(self, client, mock_auth_admin, mock_repository):
        """A short-window group user limit below the user floor (20/min) is rejected."""
        body = {
            "axis": "caller",
            "entity_type": "group",
            "name": "devs",
            "user_max_requests": 3,  # below the 20/min floor on a 60s window
            "window_seconds": 60,
        }
        resp = client.put("/api/rate-limits/caller:group:devs:60", json=body)
        assert resp.status_code == 400
        assert "below the user floor" in resp.json()["detail"]

    def test_put_agent_below_floor_rejected(self, client, mock_auth_admin, mock_repository):
        """A short-window group agent limit below the agent floor (10/min) is rejected."""
        body = {
            "axis": "caller",
            "entity_type": "group",
            "name": "agents",
            "agent_max_requests": 2,  # below the 10/min floor on a 60s window
            "window_seconds": 60,
        }
        resp = client.put("/api/rate-limits/caller:group:agents:60", json=body)
        assert resp.status_code == 400
        assert "below the agent floor" in resp.json()["detail"]

    def test_put_low_limit_allowed_on_daily_window(self, client, mock_auth_admin, mock_repository):
        """A low limit is allowed on a long (daily) window -- floor only applies to <=60s."""
        from registry.rate_limiting.models import RateLimitDefinition

        definition = RateLimitDefinition(
            axis="caller",
            entity_type="group",
            name="devs",
            user_max_requests=5000,
            window_seconds=86400,
        )
        mock_repository.upsert.return_value = definition
        body = {
            "axis": "caller",
            "entity_type": "group",
            "name": "devs",
            "user_max_requests": 5000,
            "window_seconds": 86400,
        }
        resp = client.put("/api/rate-limits/caller:group:devs:86400", json=body)
        assert resp.status_code == 200

    def test_put_requires_admin(self, client, mock_auth_regular, mock_repository):
        """A non-admin gets 403."""
        body = {
            "axis": "caller",
            "entity_type": "group",
            "name": "developers",
            "max_requests": 5,
            "window_seconds": 60,
        }
        resp = client.put("/api/rate-limits/caller:group:developers:60", json=body)
        assert resp.status_code == 403


@pytest.mark.unit
class TestListAndDelete:
    """Tests for GET and DELETE."""

    def test_list_returns_definitions(self, client, mock_auth_admin, mock_repository):
        """List returns the repository's definitions."""
        from registry.rate_limiting.models import RateLimitDefinition

        mock_repository.list_all.return_value = [
            RateLimitDefinition(
                axis="target",
                entity_type="a2a_agent",
                name="booking",
                max_requests=2,
                window_seconds=60,
            )
        ]
        resp = client.get("/api/rate-limits")
        assert resp.status_code == 200
        assert len(resp.json()["definitions"]) == 1

    def test_delete_present(self, client, mock_auth_admin, mock_repository):
        """Deleting an existing definition returns deleted=true."""
        mock_repository.delete.return_value = True
        resp = client.request("DELETE", "/api/rate-limits/caller:group:developers:60")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_absent_is_404(self, client, mock_auth_admin, mock_repository):
        """Deleting a missing definition returns 404."""
        mock_repository.delete.return_value = False
        resp = client.request("DELETE", "/api/rate-limits/caller:group:ghost:60")
        assert resp.status_code == 404


@pytest.mark.unit
class TestGetAndToggle:
    """Tests for single-read GET and enable/disable toggle."""

    def test_get_present(self, client, mock_auth_admin, mock_repository):
        """Reading an existing group definition returns it."""
        from registry.rate_limiting.models import RateLimitDefinition

        mock_repository.get_by_id.return_value = RateLimitDefinition(
            axis="caller",
            entity_type="group",
            name="developers",
            user_max_requests=25,
            window_seconds=60,
        )
        resp = client.get("/api/rate-limits/caller:group:developers:60")
        assert resp.status_code == 200
        assert resp.json()["entity_type"] == "group"
        assert resp.json()["name"] == "developers"

    def test_get_absent_is_404(self, client, mock_auth_admin, mock_repository):
        """Reading a missing definition returns 404."""
        mock_repository.get_by_id.return_value = None
        resp = client.get("/api/rate-limits/caller:group:ghost:60")
        assert resp.status_code == 404

    def test_disable_toggles_in_place(self, client, mock_auth_admin, mock_repository):
        """Disabling flips enabled=false without re-specifying the definition."""
        from registry.rate_limiting.models import RateLimitDefinition

        mock_repository.set_enabled.return_value = RateLimitDefinition(
            axis="target",
            entity_type="mcp_server",
            name="mcpgw",
            max_requests=3,
            window_seconds=60,
            enabled=False,
        )
        resp = client.post("/api/rate-limits-enabled/target:mcp_server:mcpgw:60?enabled=false")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False
        mock_repository.set_enabled.assert_awaited_once()

    def test_enable_absent_is_404(self, client, mock_auth_admin, mock_repository):
        """Toggling a missing definition returns 404."""
        mock_repository.set_enabled.return_value = None
        resp = client.post("/api/rate-limits-enabled/caller:group:ghost:60?enabled=true")
        assert resp.status_code == 404

    def test_get_requires_admin(self, client, mock_auth_regular, mock_repository):
        """A non-admin gets 403 on read."""
        resp = client.get("/api/rate-limits/caller:user:alice:60")
        assert resp.status_code == 403


@pytest.mark.unit
class TestMemberships:
    """Tests for /api/rate-limit-memberships CRUD."""

    def test_put_valid_membership(self, client, mock_auth_admin, mock_memberships_repository):
        """A valid membership whose body matches the URL id is stored."""
        from registry.rate_limiting.models import RateLimitMembership

        membership = RateLimitMembership(subject_type="user", subject="alice", groups=["devs"])
        mock_memberships_repository.upsert.return_value = membership
        body = {"subject_type": "user", "subject": "alice", "groups": ["devs"]}
        resp = client.put("/api/rate-limit-memberships/user:alice", json=body)
        assert resp.status_code == 200
        assert resp.json()["groups"] == ["devs"]

    def test_put_url_id_mismatch_rejected(
        self, client, mock_auth_admin, mock_memberships_repository
    ):
        """A body building a different _id than the URL is a 400."""
        body = {"subject_type": "user", "subject": "alice", "groups": ["devs"]}
        resp = client.put("/api/rate-limit-memberships/user:bob", json=body)
        assert resp.status_code == 400
        assert "does not match" in resp.json()["detail"]

    def test_put_invalid_subject_type_rejected(
        self, client, mock_auth_admin, mock_memberships_repository
    ):
        """An invalid subject_type fails model validation with a 400."""
        body = {"subject_type": "group", "subject": "x", "groups": ["g"]}
        resp = client.put("/api/rate-limit-memberships/group:x", json=body)
        assert resp.status_code == 400

    def test_list_memberships(self, client, mock_auth_admin, mock_memberships_repository):
        """List returns the repository's memberships."""
        from registry.rate_limiting.models import RateLimitMembership

        mock_memberships_repository.list_all.return_value = [
            RateLimitMembership(subject_type="client", subject="agent-1", groups=["agents"])
        ]
        resp = client.get("/api/rate-limit-memberships")
        assert resp.status_code == 200
        assert len(resp.json()["memberships"]) == 1

    def test_delete_present(self, client, mock_auth_admin, mock_memberships_repository):
        """Deleting an existing membership returns deleted=true."""
        mock_memberships_repository.delete.return_value = True
        resp = client.request("DELETE", "/api/rate-limit-memberships/user:alice")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_absent_is_404(self, client, mock_auth_admin, mock_memberships_repository):
        """Deleting a missing membership returns 404."""
        mock_memberships_repository.delete.return_value = False
        resp = client.request("DELETE", "/api/rate-limit-memberships/user:ghost")
        assert resp.status_code == 404

    def test_membership_requires_admin(
        self, client, mock_auth_regular, mock_memberships_repository
    ):
        """A non-admin gets 403 on membership list."""
        resp = client.get("/api/rate-limit-memberships")
        assert resp.status_code == 403


@pytest.mark.unit
class TestCallerTargetAxisRoute:
    """PUT accepts the caller_target axis and applies the caller floor."""

    def test_put_caller_target_definition(self, client, mock_auth_admin, mock_repository):
        from registry.rate_limiting.models import RateLimitDefinition

        definition = RateLimitDefinition(
            axis="caller_target",
            entity_type="group",
            name="per-server-cap",
            user_max_requests=60,
            window_seconds=60,
        )
        mock_repository.upsert.return_value = definition
        body = {
            "axis": "caller_target",
            "entity_type": "group",
            "name": "per-server-cap",
            "user_max_requests": 60,
            "window_seconds": 60,
        }
        resp = client.put("/api/rate-limits/caller_target:group:per-server-cap:60", json=body)
        assert resp.status_code == 200
        assert resp.json()["axis"] == "caller_target"

    def test_caller_target_floor_applied(self, client, mock_auth_admin, mock_repository):
        """A caller_target group below the user floor on a short window is rejected (400)."""
        body = {
            "axis": "caller_target",
            "entity_type": "group",
            "name": "tiny",
            "user_max_requests": 1,
            "window_seconds": 60,
        }
        resp = client.put("/api/rate-limits/caller_target:group:tiny:60", json=body)
        assert resp.status_code == 400


@pytest.mark.unit
class TestReservedGroupProtection:
    """A reserved kill-switch group cannot be shadowed by a rate def, nor deleted."""

    def test_reserved_name_rejected_on_rate_axis(self, client, mock_auth_admin, mock_repository):
        body = {
            "axis": "caller",
            "entity_type": "group",
            "name": "quarantine-callers",
            "user_max_requests": 25,
            "window_seconds": 60,
        }
        resp = client.put("/api/rate-limits/caller:group:quarantine-callers:60", json=body)
        assert resp.status_code == 400

    def test_reserved_group_cannot_be_deleted(self, client, mock_auth_admin, mock_repository):
        resp = client.delete("/api/rate-limits/quarantine:group:quarantine-callers:1")
        assert resp.status_code == 409


@pytest.mark.unit
class TestQuarantineEndpoints:
    """Quarantine add/remove/list convenience endpoints."""

    def test_add_caller_quarantine(self, client, mock_auth_admin, mock_memberships_repository):
        from registry.rate_limiting.models import RateLimitMembership

        mock_memberships_repository.get_by_id.return_value = None
        stored = RateLimitMembership(
            subject_type="user", subject="alice", groups=["quarantine-callers"]
        )
        mock_memberships_repository.upsert.return_value = stored
        mock_memberships_repository.count_group_members.return_value = 1

        # alice is not an admin -> guard resolves her as non-admin.
        with patch(
            "registry.api.rate_limit_routes._subject_is_admin",
            new=AsyncMock(return_value=False),
        ):
            resp = client.post("/api/rate-limit-quarantine/user:alice")
        assert resp.status_code == 200
        assert resp.json()["groups"] == ["quarantine-callers"]

    def test_add_caller_quarantine_refuses_admin(
        self, client, mock_auth_admin, mock_memberships_repository
    ):
        """An admin caller must not be quarantinable (self-lockout guard)."""
        with patch(
            "registry.api.rate_limit_routes._subject_is_admin",
            new=AsyncMock(return_value=True),
        ):
            resp = client.post("/api/rate-limit-quarantine/user:alice")
        assert resp.status_code == 403
        assert "administrator" in resp.json()["detail"].lower()
        mock_memberships_repository.upsert.assert_not_called()

    def test_add_caller_quarantine_fails_closed_on_unresolvable_admin(
        self, client, mock_auth_admin, mock_memberships_repository
    ):
        """If admin status can't be verified, the quarantine is refused (fail closed)."""
        from fastapi import HTTPException

        with patch(
            "registry.api.rate_limit_routes._subject_is_admin",
            new=AsyncMock(side_effect=HTTPException(status_code=503, detail="unverifiable")),
        ):
            resp = client.post("/api/rate-limit-quarantine/user:alice")
        assert resp.status_code == 503
        mock_memberships_repository.upsert.assert_not_called()

    def test_add_target_quarantine(self, client, mock_auth_admin, mock_memberships_repository):
        from registry.rate_limiting.models import RateLimitMembership

        stored = RateLimitMembership(
            subject_type="server", subject="mcpgw", groups=["quarantine-targets"]
        )
        mock_memberships_repository.upsert.return_value = stored
        mock_memberships_repository.count_group_members.return_value = 1

        resp = client.post("/api/rate-limit-quarantine/server:mcpgw")
        assert resp.status_code == 200
        assert resp.json()["groups"] == ["quarantine-targets"]

    def test_add_target_quarantine_normalizes_server_slashes(
        self, client, mock_auth_admin, mock_memberships_repository
    ):
        """A server subject with leading/trailing slashes is normalized to the bare name.

        Guards against a silently-inert quarantine: the request classifier keys on
        'aws-kb', so a subject stored as '/aws-kb' would never match. The route must
        upsert the normalized 'aws-kb'.
        """
        from registry.rate_limiting.models import RateLimitMembership

        mock_memberships_repository.upsert.return_value = RateLimitMembership(
            subject_type="server", subject="aws-kb", groups=["quarantine-targets"]
        )
        mock_memberships_repository.count_group_members.return_value = 1

        resp = client.post("/api/rate-limit-quarantine/server:/aws-kb/")
        assert resp.status_code == 200
        # The membership actually upserted must carry the BARE name.
        upserted = mock_memberships_repository.upsert.call_args.args[0]
        assert upserted.subject == "aws-kb"

    def test_add_target_quarantine_normalizes_agent_path(
        self, client, mock_auth_admin, mock_memberships_repository
    ):
        """An agent subject is normalized to the leading-slash form the classifier uses."""
        from registry.rate_limiting.models import RateLimitMembership

        mock_memberships_repository.upsert.return_value = RateLimitMembership(
            subject_type="agent", subject="/booking-agent", groups=["quarantine-targets"]
        )
        mock_memberships_repository.count_group_members.return_value = 1

        resp = client.post("/api/rate-limit-quarantine/agent:booking-agent")
        assert resp.status_code == 200
        upserted = mock_memberships_repository.upsert.call_args.args[0]
        assert upserted.subject == "/booking-agent"

    def test_add_quarantine_bad_subject_type(
        self, client, mock_auth_admin, mock_memberships_repository
    ):
        resp = client.post("/api/rate-limit-quarantine/bogus:x")
        assert resp.status_code == 400

    def test_remove_quarantine_target_deletes(
        self, client, mock_auth_admin, mock_memberships_repository
    ):
        from registry.rate_limiting.models import RateLimitMembership

        mock_memberships_repository.get_by_id.return_value = RateLimitMembership(
            subject_type="server", subject="mcpgw", groups=["quarantine-targets"]
        )
        mock_memberships_repository.delete.return_value = True
        mock_memberships_repository.count_group_members.return_value = 0

        resp = client.delete("/api/rate-limit-quarantine/server:mcpgw")
        assert resp.status_code == 200
        assert resp.json()["removed"] is True

    def test_remove_quarantine_not_present(
        self, client, mock_auth_admin, mock_memberships_repository
    ):
        mock_memberships_repository.get_by_id.return_value = None
        resp = client.delete("/api/rate-limit-quarantine/user:nobody")
        assert resp.status_code == 200
        assert resp.json()["removed"] is False

    def test_list_quarantine(self, client, mock_auth_admin, mock_memberships_repository):
        from registry.rate_limiting.models import RateLimitMembership

        mock_memberships_repository.list_group_members.side_effect = [
            [
                RateLimitMembership(
                    subject_type="client", subject="c1", groups=["quarantine-callers"]
                )
            ],
            [
                RateLimitMembership(
                    subject_type="server", subject="mcpgw", groups=["quarantine-targets"]
                )
            ],
        ]
        resp = client.get("/api/rate-limit-quarantine")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["callers"]) == 1
        assert len(body["targets"]) == 1

    def test_quarantine_requires_admin(
        self, client, mock_auth_regular, mock_memberships_repository
    ):
        resp = client.post("/api/rate-limit-quarantine/user:alice")
        assert resp.status_code == 403
