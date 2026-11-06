"""The vault ``user_id`` for a login that carries no account id.

Every identity provider shares one vault namespace, ``(oauth2, user_id)``. An
account id (an OIDC ``sub``, a GitHub numeric id, an Okta ``uid``) is unique
within its provider and does not collide with another provider's in practice.
A username is different: a GitHub, Google or custom login that maps no subject
keys the vault on whatever the provider's ``username_claim`` returns, and that
value can equal an account id another provider issues (a PingFederate ``sub``
that holds an email, for example). Only those fallback keys are namespaced
here, so no account-id key changes.

The auth server's login callback (``_login_vault_user_id``) is the only code
that produces a fallback key: it computes one once for a session and stores
it on the session record as ``egress_user``, and every later reader of that
session's key takes the stored value through
``registry.egress_auth.session_vault_user.session_vault_user_id`` rather than
rebuilding it. So the ``<provider>`` segment is always a login provider key
from ``oauth2_providers.yml``, and two providers' users cannot share a key. An
IdP bearer with no account id gets no fallback key (``_canonical_egress_user``
returns ``""``). A token ``/internal/tokens`` mints from a subject-less session
carries that session's stored fallback key as ``egress_user``, so the MCP
client holding it uses the connections the browser saved.
"""

from urllib.parse import quote

# Marks a username-fallback key. An account id that starts with it is refused
# (no login, no vault key; see ``is_fallback_shaped``): a provider whose
# ``subject_claim`` names a user-chosen value could otherwise return the
# fallback key of another user.
FALLBACK_KEY_PREFIX: str = "name:"


def is_fallback_shaped(
    account_id: object,
) -> bool:
    """Return True when an account id has the shape of a username-fallback key.

    Every outside value that would key the vault passes through this check: a
    login's subject (``_require_login_subject``), an IdP bearer's account id in
    the auth server's ``_canonical_egress_user``, and the ``egress_user`` or
    ``subject`` of a mint request that names no session. Such an id keys
    nothing, so it can never read the connections of the subject-less login
    whose fallback key it equals.
    Values the gateway wrote (a session's stored key, and the ``egress_user`` of
    a token minted from that session) are not checked.

    Args:
        account_id: The account id an identity provider returned.

    Returns:
        True for a string that starts with ``FALLBACK_KEY_PREFIX``.
    """
    return isinstance(account_id, str) and account_id.startswith(FALLBACK_KEY_PREFIX)


def username_fallback_key(
    provider: object,
    username: object,
) -> str:
    """Return the namespaced vault ``user_id`` for a login with no account id.

    The key is ``name:<quoted provider>:<username>``. The provider is
    percent-encoded with no safe characters, so it never contains ``:`` and the
    first ``:`` after the prefix always ends it. That makes the key injective
    over ``(provider, username)``: a provider name containing ``:`` cannot
    produce another provider's key. The username follows verbatim; the vault
    encodes each key segment separately (``registry/secrets/keys.py``), so the
    ``:`` characters are safe there.

    Args:
        provider: The login provider key from the callback path (``github``,
            ``google`` or an operator-added name). A non-string counts as
            ``""``.
        username: The login's username.

    Returns:
        The fallback key, or ``""`` when ``username`` is not a non-empty string,
        so the caller owns no per-user bucket.

    Example:
        >>> username_fallback_key("github", "alice")
        'name:github:alice'
    """
    if not isinstance(username, str) or not username:
        return ""
    provider_name = provider if isinstance(provider, str) else ""
    return f"{FALLBACK_KEY_PREFIX}{quote(provider_name, safe='')}:{username}"
