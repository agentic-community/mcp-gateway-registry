"""An httpx2 transport that sends every backend call through the registry's SSRF guard.

FastMCP's OpenAPI provider wants an ``httpx2.AsyncClient``. The registry's guard
(``registry/utils/url_guard.py``) is an ``httpx`` transport. This adapter copies
each httpx2 request into an httpx request, hands it to ``GuardedAsyncTransport``
under ``PROXY_PROFILE``, and copies the response back, reading at most
``max_response_bytes``. It makes no trust decision of its own: validation, IP
pinning and the metadata-address block all stay in ``url_guard.py``.
"""

import base64
import json
import logging
import re
from dataclasses import dataclass
from urllib.parse import (
    quote,
    quote_plus,
)

import httpx
import httpx2
from fastmcp.server.dependencies import get_http_request

from registry.exceptions import UrlValidationError
from registry.utils import url_guard

logger = logging.getLogger(__name__)


class BackendResponseTooLargeError(httpx2.RequestError):
    """The backend response passed the size cap."""


class BackendUrlBlockedError(httpx2.RequestError):
    """The SSRF guard refused the backend URL."""


class BackendRequestFailedError(httpx2.RequestError):
    """The backend could not be reached (DNS, connect, TLS, timeout)."""


class MissingUserCredentialError(httpx2.RequestError):
    """A per-user target was called without the caller's key."""


# Some APIs echo the caller's credential back (api.nasa.gov puts api_key=<key>
# into every "links" URL, even when the key arrived in a header). A credential
# must never reach the agent, so every echo of it is replaced in the response.
REDACTED: bytes = b"[REDACTED]"
# Shorter values are not redacted: they would match ordinary text everywhere.
MIN_REDACT_CHARS: int = 6

# A dot segment, plain or percent-encoded, between separators that may also be
# encoded. A backend that decodes the path could resolve it out of the target's
# base path, so such a URL is refused before any credential is attached.
_DOT_SEGMENT_RE: re.Pattern[str] = re.compile(r"(^|/|%2f)(\.|%2e){1,2}(/|%2f|$)", re.IGNORECASE)
_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}

# Backends are asked for uncompressed responses, and a compressed one is refused
# before it is read: decoding happens before the size cap can apply, so a small
# compressed body could expand past it in memory.
IDENTITY_ENCODINGS: frozenset[str] = frozenset({"", "identity"})
_DROPPED_RESPONSE_HEADERS: frozenset[bytes] = frozenset(
    {b"content-encoding", b"transfer-encoding", b"content-length", b"set-cookie", b"set-cookie2"}
)

# Key under which the dispatcher stores the verified per-user keys on the ASGI
# scope. FastMCP rebuilds the request from the scope inside the tool call.
USER_CREDENTIALS_SCOPE_KEY: str = "openapi_mcp_user_credentials"


@dataclass(frozen=True)
class PerUserAuth:
    """How a per-user target presents the caller's key."""

    target_name: str
    scheme: str
    header_name: str | None
    base_url: str


def _request_credentials() -> dict:
    """The verified per-user keys of the request being served, or {}."""
    try:
        request = get_http_request()
    except RuntimeError:
        return {}
    return request.scope.get(USER_CREDENTIALS_SCOPE_KEY) or {}


def _same_destination(
    a: str,
    b: str,
) -> bool:
    try:
        return url_guard.normalize_url_identity(a).rstrip("/") == url_guard.normalize_url_identity(
            b
        ).rstrip("/")
    except UrlValidationError:
        return False


def _json_forms(
    secret: str,
) -> set[str]:
    """How a JSON encoder may write the secret inside a string: escaped quotes and
    backslashes, \\uXXXX for non-ASCII, and the optional \\/ escape for "/"."""
    forms = {json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1]}
    forms |= {form.replace("/", "\\/") for form in forms}
    return forms


def _origin(
    url: httpx2.URL,
) -> tuple[str, str, int | None]:
    scheme = url.scheme.lower()
    return scheme, (url.host or "").lower(), url.port or _DEFAULT_PORTS.get(scheme)


def _within_base(
    url: httpx2.URL,
    base: httpx2.URL,
) -> bool:
    """True when the URL keeps the base URL's origin and stays under its path."""
    if _origin(url) != _origin(base):
        return False
    raw_path = url.raw_path.split(b"?", 1)[0].decode("ascii", "replace")
    if _DOT_SEGMENT_RE.search(raw_path) or "\\" in raw_path or "%5c" in raw_path.lower():
        return False
    root = base.path.rstrip("/")
    return not root or url.path == root or url.path.startswith(root + "/")


def secret_forms(
    *secrets: str | None,
    basic_username: str | None = None,
) -> tuple[bytes, ...]:
    """Every form in which a backend is likely to echo a credential.

    The raw value, its URL-encoded forms, and for basic auth the base64 of
    "user:password". Longest first, so a longer form is replaced before a
    shorter one inside it.
    """
    forms: set[str] = set()
    for secret in secrets:
        if not secret or len(secret) < MIN_REDACT_CHARS:
            continue
        forms.update({secret, quote(secret, safe=""), quote_plus(secret)})
        forms.update(_json_forms(secret))
        if basic_username is not None:
            forms.add(base64.b64encode(f"{basic_username}:{secret}".encode()).decode())
    return tuple(sorted((f.encode() for f in forms), key=len, reverse=True))


def redact(
    data: bytes,
    forms: tuple[bytes, ...],
) -> bytes:
    """Replace every form of a credential in a response body or header value."""
    for form in forms:
        if form in data:
            data = data.replace(form, REDACTED)
    return data


def _user_key(
    per_user: PerUserAuth,
    request: httpx2.Request,
) -> str:
    """The caller's key for this per-user target. Fails closed when there is none."""
    entry = _request_credentials().get(per_user.target_name)
    if not entry or not _same_destination(entry.get("base_url", ""), per_user.base_url):
        raise MissingUserCredentialError(
            f"No API key for '{per_user.target_name}'. Add your own key for this API in the "
            "registry (Connected Accounts, or the server's details), then call the tool again.",
            request=request,
        )
    return entry["key"]


def _user_auth_header(
    per_user: PerUserAuth,
    key: str,
) -> tuple[str, str]:
    """The (header, value) carrying the caller's key."""
    if per_user.scheme == "bearer":
        return "Authorization", f"Bearer {key}"
    return per_user.header_name or "X-API-Key", key


async def _read_capped(
    response: httpx.Response,
    max_bytes: int,
    request: httpx2.Request,
) -> bytes:
    """Read a streamed response body, stopping once it passes max_bytes."""
    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise BackendResponseTooLargeError(
                    f"backend response is larger than {max_bytes} bytes",
                    request=request,
                )
            chunks.append(chunk)
    finally:
        await response.aclose()
    return b"".join(chunks)


def _to_httpx_request(
    request: httpx2.Request,
    body: bytes,
) -> httpx.Request:
    """Copy an httpx2 request into an httpx request with the same bytes on the wire."""
    return httpx.Request(
        method=request.method,
        url=str(request.url),
        headers=list(request.headers.raw),
        content=body,
        extensions={"timeout": request.extensions.get("timeout", {})},
    )


class GuardedHttpx2Transport(httpx2.AsyncBaseTransport):
    """Send each httpx2 request through the canonical SSRF guard."""

    def __init__(
        self,
        max_response_bytes: int,
        base_url: str,
        per_user: PerUserAuth | None = None,
        shared_secret_forms: tuple[bytes, ...] = (),
        credential_bearing: bool = False,
    ) -> None:
        self._inner = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        self._max_response_bytes = max_response_bytes
        self._base = httpx2.URL(base_url)
        self._per_user = per_user
        self._shared_secret_forms = shared_secret_forms
        self._credential_bearing = credential_bearing or per_user is not None

    def _check_destination(
        self,
        request: httpx2.Request,
    ) -> None:
        """Refuse a request outside the target's base URL, or a credentialed one over HTTP.

        Runs before a per-user key is attached; a shared credential is already on
        the request, and raising here keeps it from being sent.
        """
        if not _within_base(request.url, self._base):
            logger.warning("Backend call refused: the URL leaves the target's base URL")
            raise BackendUrlBlockedError(
                "backend URL is outside the target's base URL", request=request
            )
        insecure = request.url.scheme.lower() != "https"
        if (
            insecure
            and self._credential_bearing
            and not url_guard.is_operator_trusted_host(request.url.host)
        ):
            logger.warning("Backend call refused: a credentialed target must use HTTPS")
            raise BackendUrlBlockedError(
                "a credentialed backend call must use HTTPS", request=request
            )

    async def handle_async_request(
        self,
        request: httpx2.Request,
    ) -> httpx2.Response:
        self._check_destination(request)
        body = await request.aread()
        inner_request = _to_httpx_request(request, body)
        inner_request.headers["Accept-Encoding"] = "identity"
        forms = self._shared_secret_forms
        if self._per_user is not None:
            key = _user_key(self._per_user, request)
            name, value = _user_auth_header(self._per_user, key)
            inner_request.headers[name] = value
            forms = secret_forms(key)
        try:
            inner_response = await self._inner.handle_async_request(inner_request)
        except UrlValidationError as e:
            # The error text can embed the URL, and a query-string key with it.
            logger.warning("Backend call blocked by the SSRF guard")
            raise BackendUrlBlockedError(
                "backend URL blocked by the SSRF guard", request=request
            ) from e
        except httpx.TimeoutException as e:
            raise httpx2.TimeoutException(
                f"backend timed out ({type(e).__name__})", request=request
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestFailedError(
                f"backend request failed ({type(e).__name__})", request=request
            ) from e

        encoding = inner_response.headers.get("content-encoding", "").strip().lower()
        if encoding not in IDENTITY_ENCODINGS:
            await inner_response.aclose()
            raise BackendRequestFailedError(
                f"backend sent a compressed response ({encoding[:40]}), which is not accepted",
                request=request,
            )
        content = redact(
            await _read_capped(inner_response, self._max_response_bytes, request), forms
        )
        # Set-Cookie is dropped: the client is shared by every caller of this
        # target, so a stored backend session would ride along on the next
        # caller's request.
        headers = [
            (k, redact(v, forms))
            for k, v in inner_response.headers.raw
            if k.lower() not in _DROPPED_RESPONSE_HEADERS
        ]
        return httpx2.Response(
            status_code=inner_response.status_code,
            headers=headers,
            content=content,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()
