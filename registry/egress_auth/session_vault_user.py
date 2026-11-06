"""The vault ``user_id`` of a browser session, read from its stored record.

The auth server computes a session's vault key once, at login, and stores it on
the ``oauth_sessions`` document as ``egress_user`` (see
``auth_server/session_store.py`` ``create_session``). Both services read that
document through their own projection, so every reader of a session's key goes
through :func:`session_vault_user_id` and reads only ``egress_user``. No other
session field can split the key between the auth server and the registry.

Readers: the auth server's ``_canonical_egress_user`` for a session cookie, the
token mint's ``_reconcile_context_against_session``, and the registry's
``enhanced_auth`` (the raw cookie path behind the consent callback and
``/oauth2/egress/connect``).
"""

from collections.abc import Mapping
from typing import Any


def session_vault_user_id(
    session: Mapping[str, Any],
) -> str:
    """Return the vault ``user_id`` a session record owns.

    A stored ``egress_user`` that is a non-empty string is the key, including a
    username-fallback key the login computed for a provider with no account id.
    Any other value (an empty string, a number, a list, or None, which both
    stores project for an absent field) owns nothing and is never coerced. A
    session without a stored key owns no vault key whatever its ``subject``,
    and no key is rebuilt from its ``provider`` and ``username``.

    Args:
        session: A session record as either session store projects it.

    Returns:
        The vault ``user_id``, or ``""`` when the session owns no bucket.

    Example:
        >>> session_vault_user_id({"egress_user": "name:github:alice", "provider": "google"})
        'name:github:alice'
        >>> session_vault_user_id({"subject": "00u-alice"})
        ''
    """
    stored = session.get("egress_user")
    return stored if isinstance(stored, str) and stored else ""
