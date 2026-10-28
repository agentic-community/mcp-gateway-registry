"""On-Behalf-Of (OBO) token exchange for the egress hop.

This is the auth_server side of the same-IdP OBO flow. When a registered server
has ``egress_auth_mode == "obo_exchange"``, the gateway re-audiences the user's
ingress JWT to the internal MCP server's app via the gateway's OWN IdP client
credentials, preserving the user's ``sub``. The forwarded token is an IdP-issued
token audienced to the MCP server's app; what the MCP server does with it (call
its own downstream APIs, exchange it further, etc.) is out of scope here.

Security invariants:
- The minted token embeds the user's ``sub``; it is exchanged PER REQUEST and is
  NEVER cached or reused across users. This module holds no cache.
- The gateway authenticates with its OWN IdP client credentials (read from the
  provider object), not any per-server secret.
- A Keycloak-exchanged token is verified and refused if it would also be
  accepted by this gateway: legacy exchange keeps the audience client's
  default-scope audiences, so the forwarded token must not become a gateway
  credential in the upstream's hands.
"""

import asyncio
import logging
from collections.abc import Callable

import httpx

from registry.exceptions import UrlValidationError
from registry.utils.url_guard import CREDENTIALED_OAUTH_PROFILE, validate_url

logger = logging.getLogger(__name__)

# OAuth grant types for the two supported IdPs.
_ENTRA_JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_RFC8693_TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"  # nosec B105 - OAuth grant-type URN, not a secret
_RFC8693_ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"  # nosec B105 - OAuth token-type URN, not a secret

# Network timeout for the IdP token endpoint call (matches _vend_egress_token).
_TOKEN_EXCHANGE_TIMEOUT_SECONDS: float = 10.0

# Upper bound for IdP-supplied text (error code / error_description) that is
# written to the log or carried in an exception message.
_IDP_TEXT_MAX_CHARS: int = 200

# Where the Keycloak version differences behind the choices below are written
# down, so the code comments can point at one place instead of repeating them.
_KEYCLOAK_VERSION_NOTES: str = (
    "docs/design/egress-auth-design.md#keycloak-token-exchange-behavior-by-version"
)

# KeycloakProvider.exchanged_token_gateway_audiences(token, target_audience,
# subject_token) -> the gateway audiences the verified exchanged token carries.
_ExchangedTokenVerifier = Callable[[str, str, str], list[str]]


class OboExchangeError(Exception):
    """Base error for OBO token-exchange failures."""


class OboReauthRequired(OboExchangeError):
    """The IdP refused the exchange in a way the user can fix by re-authenticating.

    Covers ``invalid_grant`` (e.g. the ingress JWT expired between /validate and
    the exchange, or the user lacks permission on the target API).
    """


class OboConsentRequired(OboExchangeError):
    """The IdP needs admin/interactive consent (``interaction_required``).

    For a same-IdP internal server this means the MCP server's app has not been
    admin-consented in the tenant.
    """


class OboConfigError(OboExchangeError):
    """The exchange is misconfigured (e.g. ``invalid_grant`` due to the gateway
    app not being granted access to the target API, or a missing target audience).
    """


class OboUnsupportedIdpError(OboExchangeError):
    """The configured IdP does not (yet) support OBO exchange in this gateway."""


def _idp_kind(idp_provider: object) -> str:
    """Classify the gateway's IdP provider object as 'entra' or 'keycloak'.

    Detection is by class name so this module does not import the provider
    classes (avoiding a heavier import graph in the hot path).
    """
    name = type(idp_provider).__name__.lower()
    if "entra" in name:
        return "entra"
    if "keycloak" in name:
        return "keycloak"
    return "unsupported"


def _entra_exchange_body(
    client_id: str,
    client_secret: str,
    subject_token: str,
    target_audience: str,
    scopes: list[str],
) -> dict[str, str]:
    """Build the Entra ``jwt-bearer`` OBO request body.

    Entra requires ``scope`` to carry the target resource; ``.default`` requests
    every delegated permission the gateway app holds on that resource. If explicit
    scopes are supplied we pass them verbatim; otherwise we synthesize
    ``<target_audience>/.default``.
    """
    if scopes:
        scope = " ".join(scopes)
    else:
        scope = f"{target_audience.rstrip('/')}/.default"
    return {
        "grant_type": _ENTRA_JWT_BEARER_GRANT,
        "client_id": client_id,
        "client_secret": client_secret,
        "assertion": subject_token,
        "scope": scope,
        "requested_token_use": "on_behalf_of",
    }


def _keycloak_exchange_body(
    client_id: str,
    client_secret: str,
    subject_token: str,
    target_audience: str,
    scopes: list[str],
) -> dict[str, str]:
    """Build the Keycloak RFC 8693 token-exchange (OBO) request body.

    The gateway authenticates with its own ``client_id``/``client_secret``,
    sends the caller's ingress access token as ``subject_token`` and names the
    target by its bare Keycloak client id in ``audience`` (not an https URL,
    and not Entra's ``scope=api://.../.default`` form). ``requested_token_type``
    is pinned because legacy exchange defaults to a refresh token. ``scope`` is
    sent only when the server entry configures explicit scopes. Which default
    client scopes apply when it is omitted depends on the Keycloak
    implementation that serves the request; see ``_KEYCLOAK_VERSION_NOTES``.
    """
    body: dict[str, str] = {
        "grant_type": _RFC8693_TOKEN_EXCHANGE_GRANT,
        "client_id": client_id,
        "client_secret": client_secret,
        "subject_token": subject_token,
        "subject_token_type": _RFC8693_ACCESS_TOKEN_TYPE,
        "requested_token_type": _RFC8693_ACCESS_TOKEN_TYPE,
        "audience": target_audience,
    }
    if scopes:
        body["scope"] = " ".join(scopes)
    return body


def _sanitize_idp_text(value: object) -> str:
    """Return IdP-supplied text that is safe to log or carry in an error.

    The token endpoint's ``error``/``error_description`` fields are untrusted
    input. Every non-printable character (CR, LF, tab and the other C0/C1
    controls, U+2028/U+2029 line separators, bidi overrides) becomes a space
    so the text cannot forge extra log lines or reorder what a reader sees,
    and the result is truncated so a verbose IdP cannot flood the log.

    Args:
        value: The raw field from the IdP's JSON body.

    Returns:
        The sanitized text; empty when ``value`` is not a string.
    """
    if not isinstance(value, str):
        return ""
    cleaned = "".join(ch if ch.isprintable() else " " for ch in value)
    return cleaned.strip()[:_IDP_TEXT_MAX_CHARS]


def _map_token_error(
    status_code: int,
    payload: dict[str, object],
    kind: str,
) -> OboExchangeError:
    """Map an IdP token-endpoint error response to a typed exception.

    ``kind`` is the IdP family (``entra``/``keycloak``). Codes whose meaning is
    provider-independent are mapped here; codes whose cause and fix are
    Keycloak-specific go through ``_map_keycloak_token_error`` so their
    remediation never reaches an operator running Entra.
    """
    err = _sanitize_idp_text(payload.get("error"))
    if err == "interaction_required":
        return OboConsentRequired("IdP requires consent")
    if err in ("invalid_grant", "invalid_token"):
        # Both mean the subject token is expired or unusable (Keycloak up to
        # 25.x answers invalid_token). Re-auth is the safer reading: a fresh
        # token is cheap to try, and a config problem simply fails again. The
        # IdP's error_description is operator diagnostics and is not returned.
        return OboReauthRequired("IdP rejected the user assertion")
    if kind == "keycloak":
        keycloak_error = _map_keycloak_token_error(err, status_code)
        if keycloak_error is not None:
            return keycloak_error
    if err in ("invalid_client", "invalid_scope", "unauthorized_client"):
        return OboConfigError(f"IdP rejected exchange configuration ({err})")
    # invalid_request stays unclassified: Keycloak answers it for an expired
    # subject token and for two configuration faults, and only its
    # error_description (logged by the caller) tells them apart. Classifying an
    # expired token as config would stop the caller retrying with a fresh one.
    return OboExchangeError(
        f"IdP token exchange failed (status={status_code}, error={err or 'unknown'})"
    )


def _map_keycloak_token_error(
    err: str,
    status_code: int,
) -> OboExchangeError | None:
    """Map the token-exchange error codes whose cause and fix are Keycloak's own.

    Entra answers ``access_denied`` for denied consent and conditional access,
    and it sends the jwt-bearer grant, so neither code below means the same
    thing there. On Entra they keep the generic ``exchange_failed`` mapping.

    Args:
        err: The sanitized ``error`` code from the token endpoint.
        status_code: The HTTP status of the token endpoint response.

    Returns:
        The typed error, or None when the code has no Keycloak-specific meaning.
    """
    if err == "unsupported_grant_type":
        # The token-exchange grant is not enabled on this Keycloak at all.
        return OboConfigError(
            "IdP does not support the token-exchange grant "
            f"(unsupported_grant_type, status={status_code})"
        )
    if err == "access_denied":
        # Legacy and standard exchange each have their own permission model, so
        # the log names both remedies; the caller gets the short message.
        logger.error(
            f"Keycloak denied the exchange (access_denied, status={status_code}): "
            "grant the target client's token-exchange permission (legacy exchange), "
            "or place the gateway client inside the subject token's audience "
            f"(standard exchange, Keycloak 26.2+). See {_KEYCLOAK_VERSION_NOTES}"
        )
        return OboConfigError("Keycloak did not permit the OBO exchange for this server")
    return None


def _keycloak_verifier(idp_provider: object) -> _ExchangedTokenVerifier:
    """Return the provider's exchanged-token verifier, checked before anything is sent.

    Without a verifier, or with a JWKS endpoint the security policy blocks, the
    exchanged token could not be verified, so the gateway's client secret and
    the user's JWT must not leave the process. The JWKS URL gets the same
    structural check as the token URL: the provider's own client resolves the
    name when it fetches, so a DNS lookup here would block the event loop
    without pinning anything.

    Args:
        idp_provider: The gateway's Keycloak provider.

    Returns:
        The provider's ``exchanged_token_gateway_audiences`` method.

    Raises:
        OboConfigError: The provider has no exchanged-token verifier.
        OboExchangeError: The realm JWKS URL is blocked by the security policy.
    """
    verifier = getattr(idp_provider, "exchanged_token_gateway_audiences", None)
    if not callable(verifier):
        raise OboConfigError("IdP provider cannot verify exchanged tokens")
    jwks_url = getattr(idp_provider, "jwks_url", "") or ""
    try:
        validate_url(
            jwks_url,
            profile=CREDENTIALED_OAUTH_PROFILE,
            resolve=False,
        )
    except UrlValidationError as exc:
        logger.error("obo_exchange: realm JWKS endpoint blocked by security policy")
        raise OboExchangeError("IdP JWKS endpoint blocked by security policy") from exc
    return verifier


async def _refuse_gateway_valid_token(
    verifier: _ExchangedTokenVerifier,
    token: str,
    target_audience: str,
    subject_token: str,
) -> None:
    """Fail closed unless an exchanged Keycloak token is safe to forward upstream.

    The exchanged token leaves the gateway, so it must not also be a credential
    for the gateway itself: an upstream that received it could replay it
    against the gateway as the user. Legacy exchange applies the audience
    client's default client scopes, and the realm's ``mcp-gateway`` audience
    mapper on the ``basic`` scope then lands in ``aud``. The verifier checks
    the JWKS signature, issuer, expiry, target audience, and that ``sub``
    matches the subject token's, before it reads ``aud``. It runs in a worker
    thread because the provider fetches its JWKS synchronously.

    Args:
        verifier: The provider's ``exchanged_token_gateway_audiences``.
        token: The access token returned by the exchange.
        target_audience: The audience the exchange was requested for.
        subject_token: The ingress token that was exchanged.

    Raises:
        OboConfigError: The token would also be accepted by this gateway.
        OboExchangeError: The token failed verification.
    """
    try:
        leaked = await asyncio.to_thread(verifier, token, target_audience, subject_token)
    except Exception as exc:
        # Fail closed on any verification error (signature, issuer, expiry,
        # target audience, subject, unreachable or malformed JWKS): an
        # unverified token is never forwarded. The verifier's messages are
        # written by the gateway and name at most the token's issuer or key id,
        # never the token itself; they are sanitized like IdP text.
        logger.warning(
            "obo_exchange: exchanged token failed verification type=%s reason=%s",
            type(exc).__name__,
            _sanitize_idp_text(str(exc)) or "-",
        )
        raise OboExchangeError("IdP returned a token that failed verification") from exc
    if leaked:
        # The audiences and the remedy go to the operator's log only; the
        # caller gets a generic message.
        logger.error(
            "obo_exchange: exchanged token carries gateway audience(s) %s; refusing to "
            "forward it. Keep gateway audiences out of the target client's default "
            "client scopes, or use standard token exchange (Keycloak 26.2+; see %s)",
            leaked,
            _KEYCLOAK_VERSION_NOTES,
        )
        raise OboConfigError(
            "IdP returned a token that this gateway itself would accept; refusing to forward it"
        )


def _gateway_credentials(
    idp_provider: object,
    target_audience: str,
) -> tuple[str, str, str]:
    """Return the gateway's client id, client secret and validated token URL.

    This is the last hop before the gateway's client secret and the user's raw
    JWT leave the process, so every value is checked here even when an earlier
    layer (registration) already enforced it. ``CREDENTIALED_OAUTH_PROFILE``
    requires https, also for hosts named in ``EGRESS_OAUTH_TRUSTED_IDP_HOSTS``;
    the internal-CA setup for a self-hosted Keycloak is in the Keycloak
    prerequisites of docs/design/egress-auth-design.md.

    Args:
        idp_provider: The gateway's own IdP provider.
        target_audience: The audience the exchange is requested for.

    Returns:
        ``(client_id, client_secret, token_url)``.

    Raises:
        OboConfigError: A credential, the token URL or the target audience is missing.
        OboExchangeError: The token URL is blocked by the security policy.
    """
    client_id = getattr(idp_provider, "client_id", "") or ""
    client_secret = getattr(idp_provider, "client_secret", "") or ""
    token_url = getattr(idp_provider, "token_url", "") or ""
    if not token_url or not client_id or not client_secret:
        raise OboConfigError("gateway IdP credentials/token_url not configured for OBO exchange")
    if not target_audience.strip():
        # Entra would send scope="/.default" and Keycloak a blank audience
        # field; neither should ever be attempted.
        raise OboConfigError("obo target_audience missing")
    try:
        validate_url(
            token_url,
            profile=CREDENTIALED_OAUTH_PROFILE,
            resolve=False,
        )
    except UrlValidationError as exc:
        logger.error("obo_exchange: token endpoint blocked by security policy")
        raise OboExchangeError("IdP token endpoint blocked by security policy") from exc
    return client_id, client_secret, token_url


def _exchange_body(
    idp_provider: object,
    kind: str,
    credentials: tuple[str, str, str],
    subject_token: str,
    target_audience: str,
    scopes: list[str],
) -> dict[str, str]:
    """Build the token-endpoint request body for the provider's IdP family.

    Args:
        idp_provider: The gateway's own IdP provider (named in the error only).
        kind: The IdP family from ``_idp_kind``.
        credentials: ``(client_id, client_secret, token_url)``.
        subject_token: The raw ingress JWT.
        target_audience: The internal MCP server's audience.
        scopes: Explicit scopes; may be empty.

    Returns:
        The form body for the token endpoint.

    Raises:
        OboUnsupportedIdpError: The IdP family has no OBO implementation.
    """
    client_id, client_secret, _ = credentials
    if kind == "entra":
        return _entra_exchange_body(
            client_id, client_secret, subject_token, target_audience, scopes
        )
    if kind == "keycloak":
        return _keycloak_exchange_body(
            client_id, client_secret, subject_token, target_audience, scopes
        )
    raise OboUnsupportedIdpError(
        f"OBO exchange not supported for IdP provider {type(idp_provider).__name__!r}"
    )


async def _post_token_request(
    token_url: str,
    body: dict[str, str],
) -> httpx.Response:
    """POST the exchange to the IdP token endpoint through the SSRF-guarded client.

    The body carries the gateway's own client secret and the user's ingress
    JWT. ``token_url`` comes from the gateway's IdP provider config, but the
    request still goes through the pinned, rebinding-safe client used by the
    3LO path (registry.egress_auth.oauth_engine), so a future change that
    derives ``token_url`` from IdP discovery cannot turn into an SSRF that
    sends the secret to an internal target. The client is resolved from its
    module at call time so tests and policy instrumentation see one reference.

    Args:
        token_url: The validated IdP token endpoint.
        body: The form body from ``_exchange_body``.

    Returns:
        The token endpoint's response.

    Raises:
        OboExchangeError: The guard blocked the target or the transport failed.
    """
    # Flat-first import: the container flattens auth_server/* into /app, so the
    # packaged form resolves only when running from the repo root (tests).
    try:
        from observability.meters import record_egress_conn_reset
    except ImportError:
        from auth_server.observability.meters import record_egress_conn_reset

    from registry.utils.url_guard import post_with_reconnect, shared_guarded_async_client

    try:
        # Pooled, process-lifetime client. A keep-alive closed while idle is
        # re-POSTed once (a token exchange is safe to repeat); a residual
        # failure fails closed below.
        client = shared_guarded_async_client(profile=CREDENTIALED_OAUTH_PROFILE)
        return await post_with_reconnect(
            client,
            token_url,
            data=body,
            timeout=_TOKEN_EXCHANGE_TIMEOUT_SECONDS,
            on_reset=lambda: record_egress_conn_reset("obo"),
        )
    except UrlValidationError as exc:
        # Guard rejected the target WITHOUT sending the credential/assertion.
        logger.error("obo_exchange: token endpoint blocked by security policy")
        raise OboExchangeError("IdP token endpoint blocked by security policy") from exc
    except httpx.HTTPError as exc:
        logger.error(f"obo_exchange: token endpoint transport failure type={type(exc).__name__}")
        raise OboExchangeError("IdP token endpoint unreachable") from exc


def _token_error(
    resp: httpx.Response,
    kind: str,
) -> OboExchangeError:
    """Log a non-200 token endpoint response and return its typed error.

    Only the standard error code and the status are logged: an OAuth token
    endpoint's error_description can echo the client_id and credential context
    (docs/SECURITY_GUIDELINES.md). The one exception is Keycloak's
    invalid_request, which covers several unrelated situations; its sanitized
    error_description is the only way to tell them apart.

    Args:
        resp: The token endpoint response (status other than 200).
        kind: The IdP family from ``_idp_kind``.

    Returns:
        The typed error for the caller to raise.
    """
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        # Valid JSON that is not an object ([], null, "oops", 3) carries no
        # error fields; treat it like a non-JSON body.
        payload = {}
    error_code = _sanitize_idp_text(payload.get("error")) or "unknown"
    description = ""
    if kind == "keycloak" and error_code == "invalid_request":
        description = _sanitize_idp_text(payload.get("error_description"))
    logger.warning(
        "obo_exchange: IdP token exchange failed status=%s error=%s description=%s",
        resp.status_code,
        error_code,
        description or "-",
    )
    return _map_token_error(resp.status_code, payload, kind)


def _access_token_from(resp: httpx.Response) -> str:
    """Return the access token from a 200 token endpoint response.

    A broken body must stay inside the typed error family, or the caller
    returns a 500 instead of a JSON-RPC failure.

    Args:
        resp: The token endpoint response (status 200).

    Returns:
        The non-empty ``access_token`` string.

    Raises:
        OboExchangeError: The body is not a JSON object or has no access token.
    """
    try:
        payload = resp.json()
    except ValueError as exc:
        raise OboExchangeError("IdP returned 200 with a non-JSON body") from exc
    if not isinstance(payload, dict):
        raise OboExchangeError("IdP returned 200 with a JSON body that is not an object")
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise OboExchangeError("IdP returned 200 but no access_token")
    return access_token


async def obo_exchange(
    idp_provider: object,
    subject_token: str,
    target_audience: str,
    scopes: list[str] | None = None,
) -> str:
    """Perform the OBO exchange: re-audience the ingress JWT to ``target_audience``.

    Args:
        idp_provider: the gateway's OWN IdP provider (from get_auth_provider()),
            exposing ``client_id``/``client_secret``/``token_url``.
        subject_token: the raw ingress JWT (the user's gateway token).
        target_audience: the internal MCP server's audience (IdP-shaped).
        scopes: audience-scoped scopes; empty/None -> ``.default`` for Entra.

    Returns:
        The exchanged access token (``aud`` = target, ``sub`` = the user).

    Raises:
        OboReauthRequired, OboConsentRequired, OboConfigError,
        OboUnsupportedIdpError, OboExchangeError.

    This token bakes in the user's ``sub`` and MUST NOT be cached across users;
    callers invoke this per request.
    """
    kind = _idp_kind(idp_provider)
    verifier = _keycloak_verifier(idp_provider) if kind == "keycloak" else None
    credentials = _gateway_credentials(idp_provider, target_audience)
    scopes = scopes or []
    body = _exchange_body(idp_provider, kind, credentials, subject_token, target_audience, scopes)
    logger.info(
        "obo_exchange: idp=%s target_audience=%s scopes=%s",
        kind,
        target_audience,
        scopes or "[.default]",
    )
    resp = await _post_token_request(credentials[2], body)
    if resp.status_code != 200:
        raise _token_error(resp, kind)
    access_token = _access_token_from(resp)
    if verifier is not None:
        await _refuse_gateway_valid_token(verifier, access_token, target_audience, subject_token)
    return access_token
