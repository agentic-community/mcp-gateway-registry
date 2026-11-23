"""The session cookie name, attributes and reader, shared by the registry and the auth server.

The auth server sets the session cookie at login and reads it at ``/validate``;
the registry reads it on every browser request and deletes it at logout. Both
must use one name, so both take it from ``Settings.session_cookie_effective_name``,
which ``effective_session_cookie_name`` computes from ``SESSION_COOKIE_NAME`` and
``SESSION_COOKIE_HOST_PREFIX``. Both set and expire it with the attributes
``session_cookie_attributes`` returns, so a browser matches every expiry to the
cookie the login set.

Both must also agree on which cookie is the session when the ``Cookie`` header
carries the name more than once, so both read it with ``single_session_cookie``
over every ``Cookie`` header field joined by ``join_cookie_fields``. HTTP/2
lets a client split its cookies over several fields, so a reader of the first
field alone would miss a duplicate in a later one.

Another host under the parent domain can set a cookie of that name for the
parent domain, and a browser sends it beside the real one. Browsers order the
two by longer Path first and then older creation time, so the host that set the
second cookie decides which one comes first and which one comes last. Taking
either is unsafe, so a duplicated session cookie counts as no session cookie.

This module imports no settings, so it works from its inputs only.
"""

import logging
from typing import (
    Any,
    NamedTuple,
)

logger = logging.getLogger(__name__)

DEFAULT_SESSION_COOKIE_NAME: str = "mcp_gateway_session"
# A browser stores a cookie whose name starts with this only when it is Secure,
# has Path=/ and has no Domain, so no sibling subdomain can set or overwrite it.
HOST_COOKIE_PREFIX: str = "__Host-"
# With the __Host- prefix on, the session cookie value is signed with this plus
# the cookie's effective name as the itsdangerous salt, so a value signed for
# one cookie name fails the check under another.
SESSION_COOKIE_SIGNING_SALT_PREFIX: str = "mcp-gateway-session-cookie:"
# The session cookie's SameSite and HttpOnly attributes are fixed, not settings.
# SameSite must be Lax: the login callback sets the cookie on a top-level
# cross-site navigation back from the identity provider, and a browser drops a
# Strict cookie there, so nobody could sign in. CSRF is defended by the CSRF
# token, not by SameSite=Strict. HttpOnly keeps the cookie away from scripts on
# the page.
SESSION_COOKIE_SAMESITE: str = "lax"
SESSION_COOKIE_HTTPONLY: bool = True
# Browsers give names starting with these special rules (matched without regard
# to case), so a configured name must not start with one.
_RESERVED_COOKIE_PREFIXES: tuple[str, ...] = ("__host-", "__secure-")
# RFC 6265 section 4.1.1: a cookie name is an RFC 2616 token, which is printable
# US-ASCII ("!" to "~") without these separators.
_COOKIE_NAME_SEPARATORS: frozenset[str] = frozenset('()<>@,;:\\"/[]?={}')


def check_session_cookie_name(
    configured_name: str,
) -> str:
    """Refuse a configured session cookie name a browser would not keep as is.

    Args:
        configured_name: The ``SESSION_COOKIE_NAME`` value.

    Returns:
        The name without surrounding whitespace.

    Raises:
        ValueError: The name is empty, starts with ``__Host-`` or ``__Secure-``
            in any case, or contains a character that RFC 6265 does not allow
            in a cookie name.

    Example:
        >>> check_session_cookie_name(" custom_session ")
        'custom_session'
    """
    name = configured_name.strip()
    if not name:
        raise ValueError(
            "SESSION_COOKIE_NAME is empty. Unset it to use mcp_gateway_session, or set "
            "it to a cookie name, on the registry and the auth server."
        )
    if name.lower().startswith(_RESERVED_COOKIE_PREFIXES):
        raise ValueError(
            f"SESSION_COOKIE_NAME={name!r} starts with a reserved cookie prefix. "
            "Browsers apply special rules to __Host- and __Secure- names that this "
            "setting does not meet. Remove the prefix from the name, and set "
            "SESSION_COOKIE_HOST_PREFIX=true for a __Host- session cookie."
        )
    invalid = sorted({ch for ch in name if not "!" <= ch <= "~" or ch in _COOKIE_NAME_SEPARATORS})
    if invalid:
        raise ValueError(
            f"SESSION_COOKIE_NAME={name!r} contains {', '.join(map(repr, invalid))}, a "
            "character a cookie name cannot carry (RFC 6265: no spaces, control or "
            'non-ASCII characters, or any of ()<>@,;:\\"/[]?={}). Use letters, digits, '
            "underscores, hyphens and dots."
        )
    return name


def effective_session_cookie_name(
    name: str,
    host_prefix: bool,
) -> str:
    """Return the name the session cookie is set and read under.

    Args:
        name: The ``SESSION_COOKIE_NAME`` value ``check_session_cookie_name`` accepted.
        host_prefix: The ``SESSION_COOKIE_HOST_PREFIX`` value.

    Returns:
        ``name``, with ``__Host-`` in front when ``host_prefix`` is true.

    Example:
        >>> effective_session_cookie_name("custom_session", host_prefix=True)
        '__Host-custom_session'
    """
    if host_prefix:
        return HOST_COOKIE_PREFIX + name
    return name


def session_cookie_signing_salt(
    effective_name: str,
    host_prefix: bool,
) -> str | None:
    """Return the salt the session cookie value is signed and checked with.

    Turning the ``__Host-`` prefix on renames the cookie but leaves the previous
    one in browsers until it expires, with the session it names still live. The
    previous cookie carries the parent ``Domain``, so every host under it keeps
    receiving the value, and the value alone is a signed session id. With the
    prefix on the salt names the cookie, so that value, or one signed for any
    other name, fails the check when presented as the ``__Host-`` cookie.

    With the prefix off this returns None: itsdangerous keeps its default salt,
    and every cookie signed before the prefix existed stays valid.

    Args:
        effective_name: The name the session cookie is set and read under.
        host_prefix: The ``SESSION_COOKIE_HOST_PREFIX`` value.

    Returns:
        The salt for the serializer's ``dumps`` and ``loads``, or None for its default.

    Example:
        >>> session_cookie_signing_salt("__Host-mcp_gateway_session", host_prefix=True)
        'mcp-gateway-session-cookie:__Host-mcp_gateway_session'
    """
    if not host_prefix:
        return None
    return SESSION_COOKIE_SIGNING_SALT_PREFIX + effective_name


def check_host_prefix_settings(
    host_prefix: bool,
    secure: bool,
    domain: str | None,
) -> None:
    """Refuse a ``__Host-`` session cookie that browsers would not store.

    Args:
        host_prefix: The ``SESSION_COOKIE_HOST_PREFIX`` value.
        secure: The ``SESSION_COOKIE_SECURE`` value.
        domain: The ``SESSION_COOKIE_DOMAIN`` value.

    Raises:
        ValueError: ``host_prefix`` is true while ``secure`` is false or
            ``domain`` is set.
    """
    if not host_prefix:
        return
    problems = []
    if not secure:
        problems.append("SESSION_COOKIE_SECURE must be true")
    if (domain or "").strip():
        problems.append("SESSION_COOKIE_DOMAIN must be empty")
    if problems:
        raise ValueError(
            "SESSION_COOKIE_HOST_PREFIX=true needs a Secure session cookie with no "
            f"Domain, but {' and '.join(problems)}. Browsers drop a __Host- cookie "
            "otherwise, so nobody could sign in. Fix those variables on the registry "
            "and the auth server, or set SESSION_COOKIE_HOST_PREFIX=false."
        )


def session_cookie_secure_flag(
    is_https: bool,
    setting_secure: bool,
    host_prefix: bool,
) -> bool:
    """Return whether a Set-Cookie of the session cookie, or its expiry, carries Secure.

    Both services send every set and every expiry of the session cookie with
    this flag. On HTTPS the cookie is Secure exactly when
    ``SESSION_COOKIE_SECURE`` is true. On plain HTTP it is never Secure,
    because a browser drops a Secure Set-Cookie sent over HTTP, except with
    the ``__Host-`` prefix, which a browser accepts only with Secure.

    Args:
        is_https: Whether the browser reached the gateway over HTTPS.
        setting_secure: The ``SESSION_COOKIE_SECURE`` value.
        host_prefix: The ``SESSION_COOKIE_HOST_PREFIX`` value.

    Returns:
        True when the cookie is Secure.
    """
    if host_prefix:
        return True
    return is_https and setting_secure


class SessionCookieAttributes(NamedTuple):
    """How to set the session cookie, and how to expire every copy a browser may hold.

    Attributes:
        set_cookie: Keyword arguments for ``Response.set_cookie``; add ``value``.
        expiries: Keyword arguments for ``Response.delete_cookie``, one per
            cookie a browser keeps under the name at Path=/: the host-only one,
            then the one on ``SESSION_COOKIE_DOMAIN`` when it is set.
    """

    set_cookie: dict[str, Any]
    expiries: tuple[dict[str, Any], ...]

    def expiries_of_other_variants(self) -> tuple[dict[str, Any], ...]:
        """Return the expiries of every variant except the one ``set_cookie`` sets.

        A login sends these beside its cookie, so a browser that kept the other
        variant from before a ``SESSION_COOKIE_DOMAIN`` change does not send two.
        """
        return tuple(
            expiry for expiry in self.expiries if expiry["domain"] != self.set_cookie["domain"]
        )


def session_cookie_attributes(
    name: str,
    is_https: bool,
    secure: bool,
    domain: str | None,
    host_prefix: bool,
    max_age: int,
) -> SessionCookieAttributes:
    """Return the attributes both services set and expire the session cookie with.

    A browser ignores an expiry unless its name, Domain and Path match the
    cookie's, and keeps a host-only cookie and one with a Domain as two cookies.
    The cookie has Path=/, the Domain ``SESSION_COOKIE_DOMAIN`` names (none with
    the ``__Host-`` prefix), SameSite=Lax, HttpOnly, and the Secure flag of
    ``session_cookie_secure_flag``. Each expiry carries the same attributes.

    Args:
        name: The effective session cookie name.
        is_https: Whether the browser reached the gateway over HTTPS.
        secure: The ``SESSION_COOKIE_SECURE`` value.
        domain: The ``SESSION_COOKIE_DOMAIN`` value, or None when unset.
        host_prefix: The ``SESSION_COOKIE_HOST_PREFIX`` value.
        max_age: The ``SESSION_MAX_AGE_SECONDS`` value.

    Returns:
        The set attributes and the expiry of every variant.

    Raises:
        ValueError: ``host_prefix`` is true with ``secure`` false or a domain,
            which settings already refuse at startup.
    """
    check_host_prefix_settings(host_prefix, secure, domain)
    cookie_domain = (domain or "").strip() or None
    shared: dict[str, Any] = {
        "key": name,
        "path": "/",
        "secure": session_cookie_secure_flag(is_https, secure, host_prefix),
        "httponly": SESSION_COOKIE_HTTPONLY,
        "samesite": SESSION_COOKIE_SAMESITE,
    }
    expiries = [{**shared, "domain": None}]
    if cookie_domain is not None:
        expiries.append({**shared, "domain": cookie_domain})
    return SessionCookieAttributes(
        set_cookie={**shared, "domain": cookie_domain, "max_age": max_age},
        expiries=tuple(expiries),
    )


def join_cookie_fields(
    cookie_fields: list[str],
) -> str:
    """Join every ``Cookie`` request header field into one header, in arrival order.

    HTTP/2 (RFC 9113 section 8.2.3) lets a client send each cookie in its own
    ``Cookie`` field, and a proxy that does not merge them passes them on
    separately. A reader of only the first field would miss a session cookie,
    or a duplicate of it, in a later one. Pass every field, for example
    ``request.headers.getlist("cookie")``.

    Args:
        cookie_fields: The values of every ``Cookie`` header field, in order.

    Returns:
        The fields joined with ``"; "``; empty when there are none.

    Example:
        >>> join_cookie_fields(["a=1", "s=abc"])
        'a=1; s=abc'
    """
    return "; ".join(cookie_fields)


def session_cookie_values(
    cookie_header: str,
    name: str,
) -> list[str]:
    """Return every value the ``Cookie`` header carries under the session cookie name.

    Only a pair whose name equals ``name`` exactly counts, so a cookie such as
    ``old_mcp_gateway_session`` is not taken for the session. Pairs split on
    ``;`` and each pair on its first ``=``; names and values lose their
    surrounding whitespace, and values are not unquoted. A browser sends one
    pair per cookie in its jar, so two cookies of that name set for different
    domains or paths both appear, in the browser's order.

    Args:
        cookie_header: The raw ``Cookie`` request header, possibly empty.
        name: The session cookie name.

    Returns:
        The values in header order; empty when the header has none.
    """
    values: list[str] = []
    for pair in cookie_header.split(";"):
        pair_name, separator, value = pair.strip().partition("=")
        if separator and pair_name.strip() == name:
            values.append(value.strip())
    return values


def log_refused_session_cookie(
    count: int,
) -> None:
    """Log the WARNING for a session cookie the request carried ``count`` times.

    The log never includes a value.

    Args:
        count: How many times the ``Cookie`` header carried the session cookie name.
    """
    logger.warning(
        "Refused the session cookie: the Cookie header carries it %d times. Another "
        "host under the parent domain may have set one, or the browser kept one from "
        "before a SESSION_COOKIE_DOMAIN change. The request is treated as signed out.",
        count,
    )


def single_session_cookie(
    cookie_header: str | None,
    name: str,
) -> str | None:
    """Return the session cookie when the ``Cookie`` header carries it exactly once.

    A header that carries the name twice or more, even with equal values, is
    refused: None, as when it carries none, plus one WARNING with the count
    (``log_refused_session_cookie``).

    Args:
        cookie_header: The raw ``Cookie`` request header, or None when absent.
        name: The session cookie name.

    Returns:
        The one value (possibly empty), or None when the name is absent or
        appears more than once.

    Example:
        >>> single_session_cookie("theme=dark; s=abc", "s")
        'abc'
        >>> single_session_cookie("s=abc; s=xyz", "s") is None
        True
    """
    values = session_cookie_values(cookie_header or "", name)
    if len(values) == 1:
        return values[0]
    if values:
        log_refused_session_cookie(len(values))
    return None
