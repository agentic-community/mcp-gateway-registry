"""Deleting a server must revoke the delegated credential its designation borrowed.

A discovery designation points at a vault entry holding a real person's OAuth access and
refresh tokens. That entry is NOT recoverable once the server document is gone:

- it is deliberately absent from the consenting user's Connected Accounts, because
  ``list_for_user`` reads the egress space only,
- the user-facing disconnect is pinned to the egress purpose, so they cannot reach it,
- and the designation that recorded the vault address dies with the server document.

So a missed revoke here is a permanent, invisible, un-revocable credential for a real
person. The revoke lives in ``ServerService.remove_server`` rather than in the delete
routes precisely because eight call sites reach it -- two API routes plus six
federation/reconciliation paths -- and a route-level fix would have covered two.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.services.server_service import ServerService

DESIGNATION = {
    "enabled": True,
    "oauth": {"provider": "github", "client_id": "disc-client"},
    "auth_method": "oauth2",
    "user_id": "oidc-sub-of-the-designated-admin",
    "designated_by": "owner-user",
}


def _service(record: dict | None) -> tuple[ServerService, MagicMock]:
    """ServerService resolves its repositories from the factory in __init__."""
    repo = MagicMock()
    repo.delete_with_versions = AsyncMock(return_value=1)
    with (
        patch("registry.services.server_service.get_server_repository", return_value=repo),
        patch("registry.repositories.factory.get_search_repository", return_value=MagicMock()),
    ):
        svc = ServerService()
    svc.get_server_info = AsyncMock(return_value=record)
    return svc, repo


@pytest.mark.unit
@pytest.mark.asyncio
class TestServerDeleteRevokesDiscoveryCredential:
    async def test_delete_revokes_using_the_designation_not_the_caller(self):
        """The address must come from the stored designation.

        A federation path has no acting principal at all, and an admin may delete someone
        else's server, so rebuilding the address from "whoever is calling" would target
        the wrong vault entry (or none) and silently orphan the real one.
        """
        svc, _ = _service({"path": "/srv", "oauth_discovery": dict(DESIGNATION)})
        with (
            patch(
                "registry.services.search_index_cleanup.remove_from_search_index_with_retry",
                AsyncMock(return_value=True),
            ),
            patch("registry.services.discovery_credential.revoke_discovery_credential") as revoke,
        ):
            revoke.return_value = None
            assert await svc.remove_server("/srv") is True

        revoke.assert_awaited_once()
        path, prior = revoke.await_args.args
        assert path == "/srv"
        assert prior["auth_method"] == DESIGNATION["auth_method"]
        assert prior["user_id"] == DESIGNATION["user_id"]
        assert prior["oauth"]["provider"] == "github"

    async def test_designation_is_read_with_credentials_included(self):
        """`oauth_discovery` is only present when credentials are requested.

        Without the flag the block is stripped, the designation looks absent, and the
        revoke silently no-ops -- the exact shape of the bug this guards.
        """
        svc, _ = _service({"path": "/srv", "oauth_discovery": dict(DESIGNATION)})
        with (
            patch(
                "registry.services.search_index_cleanup.remove_from_search_index_with_retry",
                AsyncMock(return_value=True),
            ),
            patch(
                "registry.services.discovery_credential.revoke_discovery_credential", AsyncMock()
            ),
        ):
            await svc.remove_server("/srv")

        svc.get_server_info.assert_awaited_once()
        assert svc.get_server_info.await_args.kwargs["include_credentials"] is True

    async def test_no_revoke_when_the_document_delete_did_not_happen(self):
        """Nothing was deleted, so the designation still owns the credential."""
        svc, repo = _service({"path": "/srv", "oauth_discovery": dict(DESIGNATION)})
        repo.delete_with_versions = AsyncMock(return_value=0)
        with (
            patch(
                "registry.services.search_index_cleanup.remove_from_search_index_with_retry",
                AsyncMock(return_value=True),
            ),
            patch("registry.services.discovery_credential.revoke_discovery_credential") as revoke,
        ):
            assert await svc.remove_server("/srv") is False

        revoke.assert_not_awaited()

    async def test_no_revoke_when_the_search_index_removal_aborts_the_delete(self):
        """The delete failed closed, so the server and its designation both survive."""
        svc, _ = _service({"path": "/srv", "oauth_discovery": dict(DESIGNATION)})
        with (
            patch(
                "registry.services.search_index_cleanup.remove_from_search_index_with_retry",
                AsyncMock(return_value=False),
            ),
            patch("registry.services.discovery_credential.revoke_discovery_credential") as revoke,
        ):
            assert await svc.remove_server("/srv") is False

        revoke.assert_not_awaited()

    async def test_server_without_a_designation_is_unaffected(self):
        """The overwhelmingly common case must not gain a failure mode."""
        svc, _ = _service({"path": "/srv"})
        with (
            patch(
                "registry.services.search_index_cleanup.remove_from_search_index_with_retry",
                AsyncMock(return_value=True),
            ),
            patch("registry.services.discovery_credential.revoke_discovery_credential") as revoke,
        ):
            revoke.return_value = None
            assert await svc.remove_server("/srv") is True
        # The helper itself no-ops on an empty designation, so it may be called with {}.
        if revoke.await_count:
            assert revoke.await_args.args[1] == {}
