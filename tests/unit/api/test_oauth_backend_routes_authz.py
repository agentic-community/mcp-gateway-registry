"""Access-control tests for the OAuth backend-auth / discovery-identity routes.

These five operations configure or expose the credentials the registry uses for
its OWN headless calls (health checks, tool discovery, security scans):

- ``GET|PUT  /api/servers/{path}/oauth-config``    -- client_credentials config
- ``GET|PUT|DELETE /api/servers/{path}/oauth-discovery`` -- designated identity

The writes were always owner-or-admin + CSRF. The two GETs originally enforced
only ``_check_server_permission("modify", ...)``, which this file exists to
prevent regressing: ``modify_service`` is granted to any caller holding an
``/execute`` scope on the server, so on its own it let a non-owner read another
owner's ``token_url`` / ``client_id`` / custom authorize+token endpoints and the
identity the registry borrows. The sibling write route documents exactly this
("modify_service alone ... is not sufficient"); the reads now match.

Mounted against the REAL ``registry.main.app`` rather than a locally-assembled
FastAPI instance, so router topology and prefixes are exercised as deployed.
"""

import logging
from contextlib import contextmanager
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


OWNER = "owner-user"
OTHER = "other-user"


def _ctx(username: str, is_admin: bool = False) -> dict[str, Any]:
    """A user context that passes _check_server_permission('modify', ...)."""
    return {
        "username": username,
        "is_admin": is_admin,
        "groups": ["mcp-registry-admin"] if is_admin else ["test-group"],
        "scopes": ["mcp-servers-unrestricted/execute"],
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "accessible_agents": ["all"],
        # modify_service granted -- this is the permission the GETs used to trust alone.
        "ui_permissions": {"modify_service": ["all"], "list_service": ["all"]},
        "auth_method": "session",
    }


@pytest.fixture
def server_record() -> dict[str, Any]:
    """A remote server owned by OWNER, with both OAuth config blocks populated."""
    return {
        "server_name": "test-server",
        "path": "/test-server",
        "registered_by": OWNER,
        "deployment": "remote",
        "proxy_pass_url": "https://upstream.example.com/mcp",
        "auth_scheme": "oauth",
        "backend_oauth": {
            "token_url": "https://idp.example.com/oauth2/token",
            "client_id": "cc-client",
            "client_secret_encrypted": "ciphertext-must-not-leak",
            "scopes": ["mcp.read"],
        },
        "oauth_discovery": {
            "enabled": True,
            "oauth": {
                "provider": "custom",
                "client_id": "disc-client",
                "client_secret_encrypted": "ciphertext-must-not-leak",
                "custom_authorize_url": "https://idp.example.com/authorize",
                "custom_token_url": "https://idp.example.com/token",
            },
            "auth_method": "oauth2",
            "user_id": "oidc-sub-of-the-designated-admin",
            "designated_by": OWNER,
            "designated_at": "2026-01-01T00:00:00+00:00",
        },
    }


@pytest.fixture
def mock_server_service(server_record):
    svc = MagicMock()

    async def _get_server_info(path, include_credentials=False):
        """Mirror the real service: ciphertext is only present when asked for.

        `_prepare_server_dict` strips the encrypted fields unless
        `include_credentials=True`, and `GET /server_details` deliberately calls
        without it. A mock that always returned ciphertext would make the
        "no ciphertext in the response" assertions vacuous on that endpoint.
        """
        record = deepcopy(server_record)
        if include_credentials:
            return record
        record.get("backend_oauth", {}).pop("client_secret_encrypted", None)
        (record.get("oauth_discovery") or {}).get("oauth", {}).pop("client_secret_encrypted", None)
        record.pop("auth_credential_encrypted", None)
        return record

    svc.get_server_info = AsyncMock(side_effect=_get_server_info)
    svc.update_server = AsyncMock(return_value=True)
    svc.user_can_access_server_path = AsyncMock(return_value=True)
    # $unset companion to update_server, and the enabled-state read the edit route makes.
    svc.remove_server_fields = AsyncMock(return_value=None)
    svc.is_service_enabled = AsyncMock(return_value=True)
    return svc


@contextmanager
def _client(user_context: dict[str, Any], mock_server_service: MagicMock):
    """Yield a TestClient bound to the real app with auth + services overridden.

    A context manager rather than a bare generator so tests that build a client for a
    non-default principal can `with` it; entering via next() would let the patches
    unwind as soon as the generator was collected, and every request would then see
    the unpatched server_service and 404.
    """
    from registry.auth.dependencies import enhanced_auth, nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context
    app.dependency_overrides[enhanced_auth] = lambda: user_context
    try:
        with (
            patch("registry.api.server_routes.server_service", mock_server_service),
            patch("registry.health.service.health_service", MagicMock()),
            patch("registry.core.nginx_service.nginx_service", MagicMock()),
        ):
            yield TestClient(app, cookies={"mcp_gateway_session": "test-session"})
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client_owner(mock_server_service):
    with _client(_ctx(OWNER), mock_server_service) as c:
        yield c


@pytest.fixture
def client_other(mock_server_service):
    with _client(_ctx(OTHER), mock_server_service) as c:
        yield c


@pytest.fixture
def client_admin(mock_server_service):
    with _client(_ctx("admin", is_admin=True), mock_server_service) as c:
        yield c


READ_PATHS = [
    "/api/servers/test-server/oauth-config",
    "/api/servers/test-server/oauth-discovery",
]


@pytest.mark.unit
class TestOAuthConfigReadAuthz:
    @pytest.mark.parametrize("path", READ_PATHS)
    def test_non_owner_with_modify_is_refused(self, client_other, path):
        """modify_service alone must NOT be enough to read another owner's config."""
        resp = client_other.get(path)
        assert resp.status_code == 403, resp.text

    @pytest.mark.parametrize("path", READ_PATHS)
    def test_owner_allowed(self, client_owner, path):
        assert client_owner.get(path).status_code == 200

    @pytest.mark.parametrize("path", READ_PATHS)
    def test_admin_allowed(self, client_admin, path):
        assert client_admin.get(path).status_code == 200

    @pytest.mark.parametrize("path", READ_PATHS)
    def test_no_ciphertext_in_any_view(self, client_owner, path):
        """Even for the owner, the non-secret projections must omit ciphertext."""
        assert "ciphertext-must-not-leak" not in client_owner.get(path).text

    def test_discovery_view_omits_the_borrowed_principal_sub(self, client_owner):
        """The designated principal's OIDC sub is a vault-address component.

        No client needs it; `designated_by` already names the identity for display.
        The boolean is `identity_designated`, not `_connected`: PUT /oauth-discovery
        records the principal BEFORE it completes consent, so a true value does not
        imply a vaulted token exists.
        """
        body = client_owner.get("/api/servers/test-server/oauth-discovery").json()
        assert "oidc-sub-of-the-designated-admin" not in str(body)
        assert body["identity_designated"] is True
        assert body["designated_by"] == OWNER

    def test_oauth_config_view_omits_the_inert_header_name(self, client_owner):
        """A resolved OAuth bearer always uses Authorization, so the view must not
        advertise a configurable header name the code no longer honours."""
        body = client_owner.get("/api/servers/test-server/oauth-config").json()
        assert "auth_header_name" not in body

    def test_delete_is_refused_for_non_owner(self, client_other, mock_server_service):
        """Exactly 403, and no write attempted.

        Asserting a set like (401, 403) cannot distinguish an authz refusal from a
        CSRF rejection, so a future CSRF change could hollow this out while it still
        passes. Pinning the write is what actually proves the guard ran.
        """
        resp = client_other.request("DELETE", "/api/servers/test-server/oauth-discovery")
        assert resp.status_code == 403, resp.text
        mock_server_service.update_server.assert_not_awaited()

    @pytest.mark.parametrize(
        "leaked",
        [
            "https://idp.example.com/oauth2/token",  # backend_oauth.token_url
            "cc-client",  # backend_oauth.client_id
            "disc-client",  # oauth_discovery.oauth.client_id
            "https://idp.example.com/authorize",  # oauth_discovery.oauth.custom_authorize_url
            "oidc-sub-of-the-designated-admin",  # oauth_discovery.user_id
        ],
    )
    @pytest.mark.parametrize(
        "path",
        [
            "/api/server_details/test-server",
            "/api/server_details/all",
            # Sibling read endpoint ~4000 lines away in the same file, with a weaker
            # access check. Counting the returns inside get_server_details missed it.
            "/api/servers/test-server",
        ],
    )
    def test_server_details_does_not_bypass_the_ownership_guard(
        self, client_other, mock_server_service, server_record, leaked, path
    ):
        """GET /server_details must not hand a non-owner what the dedicated GETs refuse.

        That endpoint gates only on user_can_access_server_path -- weaker than
        modify_service, let alone ownership -- and the recursive credential projection
        strips only *_encrypted / client_secret, so every non-secret field of both
        config blocks survived it. Without this the ownership guards on
        GET /oauth-config and GET /oauth-discovery were decorative.

        BOTH path forms are covered. `/all` is the worse one: it returns every server
        the caller can reach in a single response, and it returns from its own branch
        before the single-server redaction runs, so an earlier revision closed only
        half the hole.
        """
        mock_server_service.get_all_servers_with_permissions = AsyncMock(
            return_value={"/test-server": deepcopy(server_record)}
        )
        resp = client_other.get(path)
        assert resp.status_code == 200, resp.text
        assert leaked not in resp.text

    def test_server_details_still_gives_the_owner_the_config(self, client_owner):
        """The edit modal populates from this endpoint, so the owner must keep it."""
        body = client_owner.get("/api/server_details/test-server").json()
        assert body["backend_oauth"]["client_id"] == "cc-client"
        assert body["oauth_discovery"]["oauth"]["client_id"] == "disc-client"
        # Ciphertext is stripped for everyone by the pre-existing projection.
        assert "ciphertext-must-not-leak" not in str(body)


@pytest.mark.unit
class TestSchemeChangeIsOwnerOrAdmin:
    """Changing auth_scheme via the edit form DESTROYS credential config.

    It nulls `backend_oauth` for every non-oauth scheme and clears the static
    credential for `none`. That handler authorizes with modify_service only, which is
    granted to any caller holding an /execute scope -- the same reason every dedicated
    credential route pairs it with an ownership check. Without the guard a non-owner
    could wipe the owner's backend OAuth config by saving the form, and the non-owner
    OAuth redaction makes it an easy ACCIDENT: they open the modal, see the OAuth
    fields blank, change the dropdown, save.
    """

    FORM = {
        "name": "test-server",
        "description": "d",
        "deployment": "remote",
        "proxy_pass_url": "https://upstream.example.com/mcp",
        "tags": "",
        "auth_scheme": "bearer",
    }

    def _post(self, client, **overrides):
        return client.post(
            "/api/edit/test-server",
            headers={"accept": "application/json"},
            data={**self.FORM, **overrides},
        )

    def test_non_owner_cannot_change_the_scheme(self, mock_server_service):
        with _client(_ctx(OTHER), mock_server_service) as client:
            resp = self._post(client)
        assert resp.status_code == 403, resp.text
        mock_server_service.update_server.assert_not_awaited()

    def test_owner_can(self, mock_server_service):
        with _client(_ctx(OWNER), mock_server_service) as client:
            resp = self._post(client)
        assert resp.status_code == 200, resp.text
        written = mock_server_service.update_server.await_args.args[1]
        assert written["auth_scheme"] == "bearer"
        # ...and it nulled the now-stale OAuth config, which is the point.
        assert written["backend_oauth"] is None

    def test_non_owner_may_edit_without_changing_the_scheme(self, mock_server_service):
        """The guard is scoped to a scheme CHANGE, not to editing generally.

        The stored scheme is already `oauth`, so submitting it unchanged is not a
        credential-destroying edit.
        """
        with _client(_ctx(OTHER), mock_server_service) as client:
            resp = self._post(client, auth_scheme="oauth", description="just a description")
        assert resp.status_code == 200, resp.text


@pytest.mark.unit
class TestDiscoveryDesignationIsNotStolen:
    """PUT /oauth-discovery must not silently move the borrowed identity.

    The frontend PUTs this endpoint on EVERY server save, so an owner-or-admin
    editing an unrelated field would otherwise re-point the designation at
    themselves. The borrow addresses the vault by
    (auth_method, user_id, provider, server_path), so the new principal has no
    vaulted token: get_valid_token returns None and headless discovery silently
    degrades to unauthenticated. Handing it over is a deliberate DELETE + PUT.
    """

    # Mirrors the stored config, which is what the edit form round-trips. A no-op save
    # must not trip the stranding guard; the stranding tests override one bound field.
    BODY = {
        "provider": "custom",
        "client_id": "disc-client",
        "scopes": ["repo"],
        "custom_authorize_url": "https://idp.example.com/authorize",
        "custom_token_url": "https://idp.example.com/token",
    }
    URL = "/api/servers/test-server/oauth-discovery"

    @pytest.fixture(autouse=True)
    def _egress_on(self, monkeypatch):
        """PUT /oauth-discovery requires EGRESS_AUTH_ENABLED (it designates a vault
        principal), so every designation case runs with the feature on."""
        from registry.core.config import settings

        monkeypatch.setattr(settings, "egress_auth_enabled", True, raising=False)

    def test_rejected_when_the_egress_feature_is_off(self, mock_server_service, monkeypatch):
        """Fail closed rather than storing a designation the deployment cannot honour.

        Discovery borrows a vaulted per-user token. With the egress feature off there
        is no vend path and no mounted consent route, so accepting the designation
        would just persist config that silently never authenticates.
        """
        from registry.core.config import settings

        monkeypatch.setattr(settings, "egress_auth_enabled", False, raising=False)
        ctx = _ctx(OWNER)
        ctx["auth_method"] = "oauth2"
        ctx["egress_user"] = "oidc-sub-of-owner"
        with _client(ctx, mock_server_service) as client:
            resp = client.put(self.URL, json=self.BODY)
        assert resp.status_code == 400, resp.text
        assert "EGRESS_AUTH_ENABLED" in resp.text
        mock_server_service.update_server.assert_not_awaited()

    def _put(self, client, **overrides):
        return client.put(self.URL, json={**self.BODY, **overrides})

    def test_a_different_admin_saving_keeps_the_original_designation(
        self, mock_server_service, server_record
    ):
        """A non-designee edits a non-binding field; the designation survives.

        `scopes` does not affect which vault entry the borrow resolves to, so editing
        it is safe under a preserved designation. provider/client_id are not -- see the
        stranding test below.
        """
        ctx = _ctx("admin", is_admin=True)
        ctx["auth_method"] = "oauth2"
        ctx["egress_user"] = "oidc-sub-of-some-other-admin"
        with _client(ctx, mock_server_service) as client:
            # provider/client_id match what is stored; only scopes change.
            resp = self._put(client, scopes=["repo", "read:user"])
        assert resp.status_code == 200, resp.text
        written = mock_server_service.update_server.await_args.args[1]["oauth_discovery"]
        assert written["user_id"] == "oidc-sub-of-the-designated-admin"
        assert written["designated_by"] == OWNER
        # The original designation timestamp is preserved, not refreshed.
        assert written["designated_at"] == server_record["oauth_discovery"]["designated_at"]
        # The safe edit still applied.
        assert written["oauth"]["scopes"] == ["repo", "read:user"]

    @pytest.mark.parametrize(
        "override",
        [
            {"client_id": "a-new-client"},
            {"provider": "google"},
            {"custom_token_url": "https://other-idp.example.com/token"},
        ],
    )
    def test_changing_the_bound_client_under_a_foreign_designation_is_refused(
        self, mock_server_service, override
    ):
        """409 rather than stranding the designee's vaulted credential.

        The vaulted token records the client_id it was minted under and the vend
        refuses a mismatch, so re-pointing provider/client_id while someone else's
        designation stands leaves their token behind a config it no longer matches --
        discovery degrades to unauthenticated with no signal. That is the same end
        state the designation guard prevents, reached through the other field.
        """
        ctx = _ctx("admin", is_admin=True)
        ctx["auth_method"] = "oauth2"
        ctx["egress_user"] = "oidc-sub-of-some-other-admin"
        with _client(ctx, mock_server_service) as client:
            resp = self._put(client, **override)
        assert resp.status_code == 409, resp.text
        mock_server_service.update_server.assert_not_awaited()

    def test_the_designee_may_change_their_own_bound_client(self, mock_server_service):
        """The refusal protects a FOREIGN designation, not the designee's own config."""
        ctx = _ctx(OWNER)
        ctx["auth_method"] = "oauth2"
        ctx["egress_user"] = "oidc-sub-of-the-designated-admin"
        with _client(ctx, mock_server_service) as client:
            resp = self._put(client, client_id="rotated-client", client_secret="s3cret")
        assert resp.status_code == 200, resp.text
        written = mock_server_service.update_server.await_args.args[1]["oauth_discovery"]
        assert written["oauth"]["client_id"] == "rotated-client"

    def test_first_designation_records_the_caller(self, mock_server_service, server_record):
        """With no prior designation there is nothing to preserve.

        A client_secret is required here because there is no stored ciphertext to
        inherit -- a blank secret only means "keep the existing one" on an edit.
        """
        server_record["oauth_discovery"] = None
        ctx = _ctx(OWNER)
        ctx["auth_method"] = "oauth2"
        ctx["egress_user"] = "oidc-sub-of-owner"
        with _client(ctx, mock_server_service) as client:
            resp = self._put(client, client_secret="s3cret")
        assert resp.status_code == 200, resp.text
        written = mock_server_service.update_server.await_args.args[1]["oauth_discovery"]
        assert written["user_id"] == "oidc-sub-of-owner"
        assert written["designated_by"] == OWNER

    def test_the_designee_re_saving_refreshes_their_own_designation(
        self, mock_server_service, server_record
    ):
        """Same principal: not a takeover, so the timestamp may refresh."""
        ctx = _ctx(OWNER)
        ctx["auth_method"] = "oauth2"
        ctx["egress_user"] = "oidc-sub-of-the-designated-admin"
        with _client(ctx, mock_server_service) as client:
            assert self._put(client).status_code == 200
        written = mock_server_service.update_server.await_args.args[1]["oauth_discovery"]
        assert written["user_id"] == "oidc-sub-of-the-designated-admin"
        assert written["designated_by"] == OWNER


@pytest.mark.unit
class TestRouterTopology:
    """Pin what the REAL app publishes, which a locally-built FastAPI cannot catch.

    Read through ``app.openapi()`` rather than walking ``app.routes``: this FastAPI
    wraps included routers in opaque ``_IncludedRouter`` objects that carry no
    ``.path``, and the generated schema is the published contract anyway -- the same
    document ``api/openapi.json`` is refreshed from.
    """

    def test_new_routes_are_published_under_api(self):
        from registry.main import app

        paths = app.openapi()["paths"]
        for path, methods in (
            ("/api/servers/{server_path}/oauth-config", {"get", "put"}),
            ("/api/servers/{server_path}/oauth-discovery", {"get", "put", "delete"}),
        ):
            assert path in paths, f"{path} is not published"
            assert methods <= set(paths[path]), (
                f"{path} publishes {sorted(paths[path])}, expected at least {sorted(methods)}"
            )

    def test_discovery_consent_route_presence_tracks_the_egress_flag(self):
        """The discovery consent front door rides the egress facade router.

        `GET /oauth2/egress/connect` serves purpose=discovery, and BOTH purposes are
        gated on EGRESS_AUTH_ENABLED -- as is the router that carries the route
        (registry/main.py). This pins the coupling so it stays a decision rather than
        drifting back to a handler that claims independence the deployment cannot honour.
        This records that coupling so it stays a decision rather than a surprise.
        """
        from registry.core.config import settings
        from registry.main import app

        published = "/oauth2/egress/connect" in app.openapi()["paths"]
        assert published == bool(settings.egress_auth_enabled)
