"""Security helpers for the AgentCore sync adapter.

The sync adapter runs as a standalone CLI / sidecar that talks to the registry
API and to external OIDC identity providers, and it writes bearer tokens and a
refresh manifest to disk. This module centralises the egress and secret-handling
guards so every call site shares one hardened implementation instead of a
copy-pasted snippet:

- ``guarded_oidc_get`` fetches an OIDC discovery/token document through the
  canonical SSRF/URL guard (``registry.utils.url_guard``), so a registrant- or
  manifest-supplied ``discovery_url`` cannot be steered at a private/loopback/
  link-local/metadata address or a non-http(s) scheme, and the resolved IP is
  re-validated at connect time (rebinding-safe).
- ``require_secure_registry_url`` refuses to send a bearer/OAuth credential to a
  cleartext ``http://`` registry unless it is loopback (local dev), so the
  PATCH auth-credential call never leaks tokens on the wire.
- ``write_secret_file`` writes token/manifest files ``0o600`` atomically.
- ``validate_account_ids`` enforces the 12-digit AWS account-id shape and an
  explicit allowlist (fail-closed empty) before any cross-account AssumeRole.

The URL guard is imported from ``registry.utils.url_guard`` on purpose: it is
the single hardened implementation the rest of the codebase uses. It was
decoupled from the registry's server config (lazy ``_get_settings``) so this
standalone tool can import it without needing SECRET_KEY / DocumentDB / etc.
The FEDERATION_PROFILE is used because it carries an empty bypass allowlist —
an external IdP endpoint must never be able to resolve to an internal target.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import tempfile
from typing import Any
from urllib.parse import urlparse

from registry.exceptions import UrlValidationError
from registry.utils.url_guard import FEDERATION_PROFILE, guarded_client, validate_url

logger = logging.getLogger(__name__)

# AWS account IDs are exactly 12 digits.
_ACCOUNT_ID_RE = re.compile(r"^\d{12}$")

# Owner-only file permissions for on-disk secrets (tokens, refresh manifest).
_SECRET_FILE_MODE = 0o600


class EgressSecurityError(Exception):
    """Raised when an egress target or credential transport fails a guard."""


def _is_loopback_host(host: str) -> bool:
    """Return True if the host is loopback (localhost / 127.0.0.0/8 / ::1)."""
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_secure_registry_url(registry_url: str) -> None:
    """Refuse to carry a credential to a cleartext non-loopback registry.

    The sync adapter sends a Bearer registry JWT (and PATCHes upstream OAuth2
    access tokens) to ``REGISTRY_URL``. Over plaintext ``http://`` to a remote
    host those credentials are exposed to any on-path attacker, so this fails
    closed: ``https://`` is always allowed; ``http://`` is allowed ONLY when the
    host is loopback (local development). Anything else raises.

    Args:
        registry_url: The configured registry base URL.

    Raises:
        EgressSecurityError: If a credential would be sent over cleartext to a
            non-loopback host, or the URL is malformed / non-http(s).
    """
    parsed = urlparse(registry_url)
    scheme = parsed.scheme.lower()
    if scheme == "https":
        return
    if scheme == "http" and _is_loopback_host(parsed.hostname or ""):
        return
    if scheme not in ("http", "https"):
        raise EgressSecurityError(
            f"REGISTRY_URL must use http(s): got scheme '{parsed.scheme}' in {registry_url!r}"
        )
    raise EgressSecurityError(
        "Refusing to send registry credentials over cleartext HTTP to a non-loopback "
        f"host ({registry_url!r}). Use https:// (or http://localhost for local dev)."
    )


def guarded_oidc_get(
    url: str,
    timeout: int,
) -> Any:
    """GET an OIDC discovery/token document through the canonical URL guard.

    Validates ``url`` (HTTPS-only + SSRF, default-deny private/loopback/link-
    local/reserved/metadata) and pins the resolved public IP at connect time via
    the shared guarded client, so a poisoned DNS record or a tampered manifest
    that points ``discovery_url`` at an internal/attacker target is refused before
    any request is sent. ``require_https=True`` because the OIDC exchange carries
    the OAuth2 client secret — proving the target is public is not enough; the
    transport must also be encrypted so an on-path attacker cannot read it.

    Args:
        url: The OIDC discovery or token endpoint URL (registrant-supplied).
        timeout: Connect/read timeout in seconds.

    Returns:
        The ``httpx.Response`` (caller invokes ``.json()`` / ``.raise_for_status()``).

    Raises:
        EgressSecurityError: If the URL fails SSRF/HTTPS/scheme validation.
    """
    try:
        validate_url(url, profile=FEDERATION_PROFILE, resolve=True, require_https=True)
    except UrlValidationError as e:
        raise EgressSecurityError(f"Blocked OIDC fetch to unsafe URL {url!r}: {e}") from e

    # guarded_client pins the validated IP for the life of the request and
    # re-validates on redirects, so it is rebinding-safe.
    with guarded_client(profile=FEDERATION_PROFILE, timeout=timeout) as client:
        return client.get(url)


def guarded_oidc_post(
    url: str,
    timeout: int,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
) -> Any:
    """POST to an OIDC token endpoint through the canonical URL guard.

    Same SSRF/scheme validation and connect-time IP pinning as
    ``guarded_oidc_get``. Used for the ``client_credentials`` grant, which sends
    the OAuth2 client secret in the body — so the destination (derived from a
    registrant/manifest ``discovery_url``) must be proven non-internal first.

    Args:
        url: The OAuth2 token endpoint URL.
        timeout: Connect/read timeout in seconds.
        headers: Optional request headers.
        data: Optional form body.

    Returns:
        The ``httpx.Response``.

    Raises:
        EgressSecurityError: If the URL fails SSRF/HTTPS/scheme validation.
    """
    try:
        validate_url(url, profile=FEDERATION_PROFILE, resolve=True, require_https=True)
    except UrlValidationError as e:
        raise EgressSecurityError(f"Blocked OIDC token POST to unsafe URL {url!r}: {e}") from e

    with guarded_client(profile=FEDERATION_PROFILE, timeout=timeout) as client:
        return client.post(url, headers=headers or {}, data=data or {})


def is_safe_egress_url(url: str) -> bool:
    """Return True if ``url`` passes the SSRF/scheme guard (no network I/O).

    Used at REGISTRATION time (and re-checked at manifest-read time) to reject a
    Bedrock-metadata ``discovery_url`` that is non-HTTPS or resolves to a private/
    loopback/link-local/metadata target BEFORE it is persisted or used as a
    credential-fetch destination (so a tampered gateway URL never becomes a stored
    or refresh-time egress target). ``require_https=True`` — the discovery_url is
    the endpoint the client secret is later sent to, so it must be encrypted, not
    merely public. Validates format + resolves and checks the IP, but sends no
    HTTP request. Fails closed on any error.

    Args:
        url: The candidate egress URL.

    Returns:
        True if the URL is a safe public HTTPS target, else False.
    """
    if not url:
        return False
    try:
        validate_url(url, profile=FEDERATION_PROFILE, resolve=True, require_https=True)
        return True
    except UrlValidationError:
        return False
    except Exception:  # defensive: any resolver/parse error fails closed
        return False


def write_secret_file(
    path: str,
    content: str,
) -> None:
    """Write ``content`` to ``path`` atomically with ``0o600`` permissions.

    Tokens and the refresh manifest carry live credentials, so they must not be
    world-readable. Writes to a temp file in the same directory (created with
    ``0o600`` via ``os.open``), then atomically replaces the target so a reader
    never observes a partial or default-mode file.

    Args:
        path: Destination path.
        content: Text to write.
    """
    abs_path = os.path.abspath(path)
    directory = os.path.dirname(abs_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".secret")
    try:
        os.fchmod(fd, _SECRET_FILE_MODE)
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, abs_path)
        # os.replace preserves the temp file's 0o600 mode on the new inode; if
        # the target pre-existed with looser perms it is now gone (replaced).
        os.chmod(abs_path, _SECRET_FILE_MODE)
    except BaseException:
        # Best-effort cleanup of the temp file on any failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def validate_account_ids(
    account_ids: list[str],
    allowlist: list[str] | None = None,
) -> list[str]:
    """Validate cross-account IDs against the 12-digit shape and an allowlist.

    Cross-account AssumeRole widens the blast radius to every listed account, so
    the account set is validated fail-closed:

    - Every ID must match ``^\\d{12}$`` (reject anything else).
    - When an allowlist is provided (non-empty), every requested ID must be a
      member; a request for an off-list account raises.

    This helper does NOT decide what happens when no allowlist is supplied — that
    policy lives at the call site (``sync._parse_account_ids``), which fails
    CLOSED for cross-account requests unless an allowlist is set or the explicit
    ``AGENTCORE_ALLOW_ANY_ACCOUNT`` opt-in is present. Callers that pass
    ``allowlist=None`` here have already made that authorization decision; this
    function then enforces shape only.

    Args:
        account_ids: Requested target account IDs.
        allowlist: Optional explicit allowlist of permitted account IDs.

    Returns:
        The validated list of account IDs.

    Raises:
        EgressSecurityError: On a malformed ID or an off-allowlist ID.
    """
    for acct in account_ids:
        if not _ACCOUNT_ID_RE.match(acct):
            raise EgressSecurityError(
                f"Invalid AWS account ID {acct!r}: expected exactly 12 digits."
            )
    if allowlist:
        allowed = set(allowlist)
        rejected = [a for a in account_ids if a not in allowed]
        if rejected:
            raise EgressSecurityError(
                f"Account ID(s) {rejected} are not in AGENTCORE_ALLOWED_ACCOUNTS; "
                "refusing cross-account AssumeRole (fail closed)."
            )
    return account_ids
