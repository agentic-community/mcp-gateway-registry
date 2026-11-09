"""Parse and check the Origin of a browser request.

Two callers share this parser:

- ``registry/core/config.py`` builds the CORS allowlist from configured origins.
- The auth server refuses an MCP request signed in with the browser session
  cookie unless it comes from the gateway's own origin (issue #1996). A browser
  attaches that cookie to a request any other site makes, so the cookie alone
  does not show the user meant to send the request.

An origin is ``scheme://host[:port]`` with an ``http`` or ``https`` scheme. The
parser rejects anything else, so a malformed value can never widen a list.
"""

from urllib.parse import urlparse

# Verbs a browser sends without an Origin header on a same-origin request, and
# which do not change state on an MCP route.
SAFE_HTTP_VERBS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})

DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}

# Reasons origin_refusal() returns; the auth server logs them and puts them in
# the JSON-RPC error data.
ORIGIN_MISSING: str = "origin_missing"
ORIGIN_NULL: str = "origin_null"
ORIGIN_MALFORMED: str = "origin_malformed"
ORIGIN_UNTRUSTED: str = "origin_untrusted"


def _drop_default_port(
    origin: str,
) -> str:
    """Remove :80 from an http origin and :443 from an https one.

    Browsers leave the default port out of the Origin header, so a configured
    https://gw.example.com:443 must compare equal to https://gw.example.com.
    """
    scheme, _, rest = origin.partition("://")
    default_port = DEFAULT_PORTS.get(scheme)
    suffix = f":{default_port}"
    if default_port is not None and rest.endswith(suffix):
        return f"{scheme}://{rest[: -len(suffix)]}"
    return origin


def normalize_origin(
    value: str,
) -> str | None:
    """Validate and normalize one origin string.

    Args:
        value: Candidate origin, already whitespace-stripped.

    Returns:
        ``scheme://host[:port]`` with the scheme and host lower-cased, or None
        if the value is not exactly an http or https origin. A path, query,
        fragment, user name, password or wildcard makes it None.
    """
    if not value or "*" in value:
        return None
    try:
        parsed = urlparse(value)
        # Reading .port validates it; an out-of-range or non-numeric port
        # raises ValueError, which fails closed here.
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.hostname:
        return None
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        return None
    if parsed.username or parsed.password:
        return None
    host = parsed.hostname.lower()
    # Browsers never send a trailing-dot host, so drop one trailing dot to keep
    # a configured "example.com." matching real origins.
    if host.endswith(".") and not host.endswith(".."):
        host = host[:-1]
    if not host:
        return None
    # urlparse strips the brackets from an IPv6 literal; put them back so the
    # result matches the form a browser sends (http://[::1]:8080).
    if ":" in host:
        host = f"[{host}]"
    scheme = parsed.scheme.lower()
    if port is not None:
        return f"{scheme}://{host}:{port}"
    return f"{scheme}://{host}"


def origin_of_url(
    url: str,
) -> str | None:
    """Return the origin of a configured URL, ignoring its path.

    ``REGISTRY_URL`` may carry a path (``https://gw.example.com/registry``),
    which normalize_origin() rejects. This keeps only scheme, host and port,
    and drops a default port.

    Args:
        url: A configured absolute URL, such as REGISTRY_URL.

    Returns:
        The URL's origin, or None if it is not an absolute http or https URL.
    """
    if not url or not url.strip():
        return None
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    if not parsed.scheme or not parsed.netloc:
        return None
    origin = normalize_origin(f"{parsed.scheme}://{parsed.netloc}")
    return _drop_default_port(origin) if origin else None


def origin_refusal(
    origin_header: str | None,
    http_verb: str,
    trusted_origins: frozenset[str],
) -> str | None:
    """Decide whether a cookie-authenticated request comes from a trusted origin.

    A request with an Origin header must name a trusted origin. A request
    without one passes only for a safe verb (GET, HEAD, OPTIONS), because
    browsers omit Origin on a same-origin GET. A POST or DELETE without Origin
    is refused: a browser always sends Origin on those, so its absence means
    the request did not come from a browser page we can check.

    Args:
        origin_header: The request's Origin header, or None if absent.
        http_verb: The original request's HTTP method.
        trusted_origins: Normalized origins, without default ports.

    Returns:
        None to allow the request, or one of ORIGIN_MISSING, ORIGIN_NULL,
        ORIGIN_MALFORMED or ORIGIN_UNTRUSTED.
    """
    value = (origin_header or "").strip()
    if not value:
        if (http_verb or "").upper() in SAFE_HTTP_VERBS:
            return None
        return ORIGIN_MISSING
    if value.lower() == "null":
        return ORIGIN_NULL
    origin = normalize_origin(value)
    if origin is None:
        return ORIGIN_MALFORMED
    if _drop_default_port(origin) not in trusted_origins:
        return ORIGIN_UNTRUSTED
    return None
