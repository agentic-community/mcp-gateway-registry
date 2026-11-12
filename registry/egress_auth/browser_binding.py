"""Bind each egress consent to the browser that started it.

The consent callback stores the provider token in the bucket of the principal
the signed state names, after checking the completing browser is live as that
principal. That check alone does not stop account injection: whoever holds a
state bound to user V (an agent holding V's bearer, or V's browser history)
can finish the provider step with their own account and make V's browser load
``/oauth2/egress/callback?code=<theirs>&state=<V's>``. V's vault then holds the
attacker's provider account. RFC 9700 section 4.7 and RFC 6749 section 10.12
require binding the authorization response to the user agent that started the
request.

So a consent that starts in a browser also sets a cookie there: a fresh
32-byte secret, under a name derived from its own digest, so parallel consents
in different tabs each keep their own cookie. The state carries the SHA-256 of
the secret, and the callback refuses unless the completing browser presents a
secret with that digest. A consent built where no browser exists has no cookie
to set, so it can never complete; the vend therefore builds none.
"""

import hashlib
import hmac
import secrets
from typing import NamedTuple
from urllib.parse import urlsplit

from fastapi import (
    Request,
    Response,
)

from registry.common.session_cookie import session_cookie_secure_flag
from registry.core.config import settings
from registry.utils.request_utils import is_request_https

# Every consent cookie's name starts with this; the rest is a short digest.
CONSENT_COOKIE_PREFIX: str = "egress_consent_"
# The route the provider redirects the browser to, under the callback base URL.
CALLBACK_ROUTE: str = "/oauth2/egress/callback"
_SECRET_BYTES: int = 32
# Hex characters of the digest used in the cookie name: 64 bits keeps the names
# of parallel consents apart, and reveals nothing usable about the secret.
_COOKIE_NAME_DIGEST_CHARS: int = 16


class BrowserBinding(NamedTuple):
    """One consent's browser binding.

    Attributes:
        secret: The cookie value; only the starting browser holds it.
        digest: SHA-256 hex of ``secret``; the consent state carries this.
    """

    secret: str
    digest: str


def _consent_cookie_path() -> str:
    """Return the browser-visible path of the consent callback.

    The provider redirects to exactly ``{egress_oauth_callback_base}`` plus
    ``CALLBACK_ROUTE`` (the ``redirect_uri`` the service sends), so scoping the
    cookie to that URL's path covers the callback and no other route, whatever
    ``ROOT_PATH`` or callback base the deployment uses.
    """
    return urlsplit(settings.egress_oauth_callback_base.rstrip("/") + CALLBACK_ROUTE).path


def _consent_cookie_secure(
    request: Request,
) -> bool:
    """Return whether a set or expiry of the consent cookie carries Secure.

    The session cookie's rule: on HTTPS when ``SESSION_COOKIE_SECURE`` is
    true, never on plain HTTP. The consent cookie has no ``__Host-`` name, so
    the prefix plays no part.
    """
    return session_cookie_secure_flag(
        is_https=is_request_https(request),
        setting_secure=settings.session_cookie_secure,
        host_prefix=False,
    )


def binding_digest(
    secret: str,
) -> str:
    """Return the SHA-256 hex digest the consent state carries for ``secret``.

    Args:
        secret: A consent cookie value.

    Returns:
        The lowercase hex digest.
    """
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def new_browser_binding() -> BrowserBinding:
    """Create the secret and digest for a consent that starts in a browser.

    Returns:
        A fresh binding.
    """
    secret = secrets.token_urlsafe(_SECRET_BYTES)
    return BrowserBinding(secret=secret, digest=binding_digest(secret))


def consent_cookie_name(
    digest: str,
) -> str:
    """Return the cookie name of the consent whose state carries ``digest``.

    Args:
        digest: The state's browser-binding digest.

    Returns:
        The cookie name.
    """
    return CONSENT_COOKIE_PREFIX + digest[:_COOKIE_NAME_DIGEST_CHARS]


def read_browser_secret(
    request: Request,
    digest: str,
) -> str | None:
    """Return the consent cookie the request carries for ``digest``, if any.

    Args:
        request: The callback request.
        digest: The state's browser-binding digest; empty for a state that
            binds no browser.

    Returns:
        The cookie value, or None when the state binds no browser or the
        browser holds no cookie for it.
    """
    if not digest:
        return None
    return request.cookies.get(consent_cookie_name(digest))


def browser_secret_matches(
    digest: str,
    secret: str | None,
) -> bool:
    """Return True when ``secret`` is the one whose digest the state carries.

    Compares in constant time. A state without a digest (minted before the
    binding existed, or by a path with no browser) never matches.

    Args:
        digest: The state's browser-binding digest.
        secret: The consent cookie the completing browser presented.

    Returns:
        True only for a non-empty digest and a secret that hashes to it.
    """
    if not digest or not secret:
        return False
    return hmac.compare_digest(binding_digest(secret), digest)


def set_consent_cookie(
    response: Response,
    request: Request,
    binding: BrowserBinding,
) -> None:
    """Set the consent cookie on the response that starts a consent.

    Attributes follow the session cookie and ``oauth2_temp_session``: HttpOnly;
    SameSite=Lax, because the provider's redirect back is a top-level
    cross-site GET; the session cookie's Secure rule without the ``__Host-``
    prefix (``_consent_cookie_secure``); the session cookie's
    ``SESSION_COOKIE_DOMAIN``, so the cookie reaches the callback host
    whenever the session cookie does. Path is the callback's path and Max-Age
    is the state TTL, after which the state is refused anyway.

    Args:
        response: The response that starts the consent.
        request: The request it answers.
        binding: The consent's browser binding.
    """
    response.set_cookie(
        key=consent_cookie_name(binding.digest),
        value=binding.secret,
        max_age=settings.egress_state_ttl_seconds,
        path=_consent_cookie_path(),
        domain=settings.session_cookie_domain,
        secure=_consent_cookie_secure(request),
        httponly=True,
        samesite="lax",
    )


def clear_consent_cookie(
    response: Response,
    request: Request,
    digest: str,
) -> None:
    """Expire the consent cookie for ``digest`` on a callback response.

    The name, path, domain and Secure flag match :func:`set_consent_cookie`,
    or the browser ignores the deletion. A state that binds no browser has no
    cookie to clear.

    Args:
        response: The callback response.
        request: The callback request; it decides whether the browser reached
            the gateway over HTTPS.
        digest: The state's browser-binding digest.
    """
    if not digest:
        return
    response.delete_cookie(
        key=consent_cookie_name(digest),
        path=_consent_cookie_path(),
        domain=settings.session_cookie_domain,
        secure=_consent_cookie_secure(request),
        httponly=True,
        samesite="lax",
    )
