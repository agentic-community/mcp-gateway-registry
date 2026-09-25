"""Revoking the delegated credential a discovery designation borrowed.

Lives in its own module because it is needed from two layers that must not import each
other: the ``DELETE /servers/{path}/oauth-discovery`` route (clearing a designation) and
``ServerService.remove_server`` (deleting the server that held it). Putting it in either
one would force an API module into the service layer or the reverse.

Both callers matter. Missing either leaves a live delegated credential for a real person
in the vault, and it is not a recoverable orphan: discovery entries are deliberately
absent from the consenting user's Connected Accounts (``list_for_user`` reads the egress
space only), and the designation that named the vault address is destroyed along with it,
so nothing can locate it afterwards.
"""

import logging

logger = logging.getLogger(__name__)


async def revoke_discovery_credential(server_path: str, prior_disc: dict) -> None:
    """Delete the vault entry a discovery designation was borrowing from.

    ``prior_disc`` is the ``oauth_discovery`` block as it stood BEFORE the designation was
    cleared -- callers must capture it first, since it is the only record of the address.

    Best-effort and idempotent: a designation may never have been consented, so there is
    often nothing to delete. A failure must not fail the caller's operation -- the
    designation is going away regardless, so discovery stops borrowing either way -- but
    it IS logged at exception level, because the residue is invisible to every UI.

    The address is rebuilt from the designation, never from the acting principal: an admin
    may revoke on someone else's behalf, and a federation path has no principal at all.
    """
    auth_method = prior_disc.get("auth_method")
    user_id = prior_disc.get("user_id")
    provider = (prior_disc.get("oauth") or {}).get("provider")
    if not (auth_method and user_id and provider):
        return
    try:
        from ..egress_auth.factory import get_egress_auth_service
        from ..secrets import keys

        await get_egress_auth_service().disconnect(
            auth_method=auth_method,
            user_id=user_id,
            provider=provider,
            server_path=server_path,
            purpose=keys.DISCOVERY_PURPOSE,
        )
        logger.info("revoked discovery credential for server=%s provider=%s", server_path, provider)
    except Exception:
        logger.exception(
            "FAILED to revoke discovery credential for server=%s provider=%s -- a delegated "
            "token may remain in the vault with no UI surface; remove it manually",
            server_path,
            provider,
        )
