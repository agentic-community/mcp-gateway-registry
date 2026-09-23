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

    Keycloak's token exchange authenticates the gateway's own client
    via ``client_id``/``client_secret`` form fields, carries the caller's
    ingress access token as ``subject_token`` (typed as an access_token), and
    names the target as ``audience`` — the bare target client id, not an
    https URL and not Entra's ``assertion``/``scope=api://.../.default``
    convention.

    Keycloak has two token-exchange implementations, and the same request can
    reach either. Legacy exchange (V1, the ``token-exchange`` feature) is the
    only one up to 26.1 and remains available as a preview on 26.2+. Standard
    exchange (V2, 26.2+) is enabled by default but serves a request only when
    the requesting client's "Standard token exchange" switch is on; otherwise
    the request falls back to legacy exchange, if that feature is enabled.

    ``requested_token_type`` is pinned rather than left to the server default,
    because the two implementations default differently: legacy exchange
    defaults to ``refresh_token`` and would mint a refresh token this code
    reads past and discards on every request, while standard exchange
    (26.2+) defaults to ``access_token``. RFC 8693 §2.1 makes the parameter
    OPTIONAL with a server-chosen default, which is exactly why relying on it
    is an interop hazard.

    ``scope`` is sent only when explicit scopes are requested. When it is
    omitted, whose default client scopes apply depends on the implementation:
    legacy exchange mints the token for the ``audience`` client and applies
    that client's default scopes and protocol mappers, while standard exchange
    applies the requesting client's (the gateway's) default scopes and filters
    ``aud`` down to the requested audience; it narrows ``aud`` and cannot add
    an audience those scopes do not already provide.
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

    ``kind`` is the IdP family (``entra``/``keycloak``). It exists so a
    provider-specific remediation hint never reaches an operator running the
    other IdP: the codes overlap but their causes and fixes do not. Codes whose
    meaning is provider-independent stay in the shared branches below.
    """
    err = _sanitize_idp_text(payload.get("error"))
    if err == "interaction_required":
        return OboConsentRequired("IdP requires consent")
    if err in ("invalid_grant", "invalid_token"):
        # invalid_grant spans both user-fixable (expired/no-permission) and
        # config (gateway not granted access) cases; re-auth is the safer of the
        # two, since retrying with a fresh token is cheap and a config problem
        # simply fails again. The IdP's error_description is not returned: it
        # is operator diagnostics, not something the calling agent can act on
        # (it is logged only for Keycloak's invalid_request).
        # Keycloak never answers invalid_grant for token-exchange. Up to 25.x
        # it reports an expired or unusable subject_token as invalid_token,
        # which is the same user-fixable situation; from 26.0 legacy exchange
        # answers invalid_request instead (as does standard exchange, 26.2+),
        # which stays unclassified below.
        return OboReauthRequired("IdP rejected the user assertion")
    if err == "unsupported_grant_type":
        # The token-exchange grant is not enabled on the server at all. On
        # Keycloak before 26.2 that means KC_FEATURES lacks token-exchange; from
        # 26.2 standard exchange is on by default, so it was switched off too.
        return OboConfigError(
            f"IdP does not support the token-exchange grant "
            f"(unsupported_grant_type, status={status_code})"
        )
    if err == "access_denied" and kind == "keycloak":
        # Keycloak answers access_denied when the exchange is not permitted,
        # and what "permitted" means depends on the implementation that served
        # the request, so name both rather than asserting one: legacy exchange
        # wants the token-exchange permission on the TARGET client (on 26.2+
        # that needs fine-grained admin permissions v1), while standard
        # exchange wants the gateway's own client inside the subject token's
        # audience. Both are operator-actionable configuration.
        #
        # Deliberately NOT shared with Entra: Entra returns access_denied for
        # denied consent and for conditional-access blocks, which are a
        # different fix and, for CA, not operator configuration at all.
        # Reclassifying it there would also move an already-released code path
        # from the exchange_failed audit bucket into config_error.
        return OboConfigError(
            f"Keycloak denied the exchange (access_denied, status={status_code}): "
            "grant the target client's token-exchange permission (legacy exchange), "
            "or place the gateway client inside the subject token's audience "
            "(standard exchange, Keycloak 26.2+)"
        )
    if err in ("invalid_client", "invalid_scope", "unauthorized_client"):
        return OboConfigError(f"IdP rejected exchange configuration ({err})")
    # invalid_request is deliberately NOT classified. Keycloak answers it for at
    # least three unrelated situations: an expired or unusable subject_token
    # (legacy exchange from 26.0, and standard exchange), the requesting
    # client's Standard token exchange switch being off while legacy exchange
    # is disabled, and an audience standard exchange cannot place in the token.
    # Two are operator config and one is user re-auth, and the code alone
    # cannot tell them apart — only error_description can, which is why it is
    # logged at the call site. Guessing here would be worse than the generic
    # error: classifying an expired token as config would stop the caller
    # retrying with a fresh one.
    return OboExchangeError(
        f"IdP token exchange failed (status={status_code}, error={err or 'unknown'})"
    )


async def _refuse_gateway_valid_token(
    idp_provider: object,
    token: str,
    target_audience: str,
) -> None:
    """Fail closed unless an exchanged Keycloak token is safe to forward upstream.

    The exchanged token leaves the gateway, so it must not also be a credential
    for the gateway itself: an upstream that received it could replay it
    against the gateway as the user. Legacy Keycloak token exchange (V1)
    applies the audience client's default client scopes, and the realm's
    ``mcp-gateway`` audience mapper on the ``basic`` scope then lands in
    ``aud``. The provider verifies the token (JWKS signature, issuer, expiry,
    target audience, ``sub``) before ``aud`` is inspected. It runs in a worker
    thread because the provider fetches its JWKS synchronously; the provider is
    built per request, so the JWKS is fetched on every exchange, as it already
    is on /validate.

    Args:
        idp_provider: The gateway's Keycloak provider.
        token: The access token returned by the exchange.
        target_audience: The audience the exchange was requested for.

    Raises:
        OboConfigError: The token would also be accepted by this gateway.
        OboExchangeError: The token failed verification.
    """
    # obo_exchange already refused a Keycloak provider without this method
    # before sending anything; a missing verifier would still fail closed below.
    check = getattr(idp_provider, "exchanged_token_gateway_audiences", None)
    try:
        leaked = await asyncio.to_thread(check, token, target_audience)
    except Exception as exc:
        # Fail closed on any verification error (signature, issuer, expiry,
        # missing target audience or sub, unreachable or malformed JWKS): an
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
            "client scopes, or use standard token exchange (Keycloak 26.2+; see the "
            "Keycloak prerequisites in docs/design/egress-auth-design.md)",
            leaked,
        )
        raise OboConfigError(
            "IdP returned a token that this gateway itself would accept; refusing to forward it"
        )


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
    if kind == "keycloak" and not callable(
        getattr(idp_provider, "exchanged_token_gateway_audiences", None)
    ):
        # Checked before anything is sent: without it the exchanged token could
        # not be verified, so the client secret and the user's JWT must not leave.
        raise OboConfigError("IdP provider cannot verify exchanged tokens")
    client_id = getattr(idp_provider, "client_id", "") or ""
    client_secret = getattr(idp_provider, "client_secret", "") or ""
    token_url = getattr(idp_provider, "token_url", "") or ""
    if not token_url or not client_id or not client_secret:
        raise OboConfigError("gateway IdP credentials/token_url not configured for OBO exchange")
    if not target_audience.strip():
        # Registration enforces a non-empty audience, but this is the last hop
        # before the gateway's client_secret and the user's raw JWT leave the
        # process, and a check that can be reached with the value missing is
        # equivalent to no check. Entra would send scope="/.default" and
        # Keycloak a blank audience field; neither should ever be attempted.
        raise OboConfigError("obo target_audience missing")

    # CREDENTIALED_OAUTH_PROFILE enforces TLS (require_https=True): self-hosted
    # IdPs (Keycloak is the default self-managed IdP) are supported over https
    # only. EGRESS_OAUTH_TRUSTED_IDP_HOSTS (#1707) relaxes the public-address
    # requirement for the named IdP hosts, not the TLS requirement, so the
    # shipped in-cluster http://keycloak:8080 default cannot serve OBO as-is.
    # An internal CA works via the process trust store (SSL_CERT_FILE); there
    # is no per-hop CA-bundle setting. In-cluster non-TLS OBO is deliberately
    # out of scope pending an explicit, operator-gated design (see
    # docs/design/egress-auth-design.md).
    try:
        validate_url(
            token_url,
            profile=CREDENTIALED_OAUTH_PROFILE,
            resolve=False,
        )
    except UrlValidationError as exc:
        logger.error("obo_exchange: token endpoint blocked by security policy")
        raise OboExchangeError("IdP token endpoint blocked by security policy") from exc

    scopes = scopes or []
    if kind == "entra":
        body = _entra_exchange_body(
            client_id, client_secret, subject_token, target_audience, scopes
        )
    elif kind == "keycloak":
        body = _keycloak_exchange_body(
            client_id, client_secret, subject_token, target_audience, scopes
        )
    else:
        raise OboUnsupportedIdpError(
            f"OBO exchange not supported for IdP provider {type(idp_provider).__name__!r}"
        )

    logger.info(
        "obo_exchange: idp=%s target_audience=%s scopes=%s",
        kind,
        target_audience,
        scopes or "[.default]",
    )
    # The token endpoint receives the gateway's OWN client_secret and the user's
    # ingress JWT (the OBO assertion). token_url comes from the gateway's IdP
    # provider config (not per-request/registration input), so it is trusted --
    # but we still route through the SSRF/rebinding-safe client for defense-in-
    # depth and consistency with the 3LO path (registry.egress_auth.oauth_engine),
    # so a future change that lets token_url be derived from IdP discovery can
    # never silently become an SSRF that exfiltrates the client_secret/assertion
    # to an internal target. The pinned guard rejects a non-http(s) scheme or a
    # private/metadata IP (including a post-config DNS rebind) at connect time.
    # Resolve from the canonical module at request time so policy instrumentation
    # and tests cannot be bypassed by a stale imported client reference.
    from auth_server.observability.meters import record_egress_conn_reset
    from registry.utils.url_guard import post_with_reconnect, shared_guarded_async_client

    try:
        # Pooled, process-lifetime SSRF-guarded client (keep-alive reuse across
        # token exchanges). Timeout is per-request; a keep-alive that was closed
        # while idle is transparently re-POSTed once (a token exchange is safe to
        # re-POST; a residual failure fails closed below).
        client = shared_guarded_async_client(profile=CREDENTIALED_OAUTH_PROFILE)
        resp = await post_with_reconnect(
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

    if resp.status_code != 200:
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            # Valid JSON that is not an object ([], null, "oops", 3) carries no
            # error fields; treat it like a non-JSON body.
            payload = {}
        # Only the standard error code and the status are logged: an OAuth
        # token endpoint's error_description can echo the client_id and
        # credential context (docs/SECURITY_GUIDELINES.md). The one exception
        # is Keycloak's invalid_request, which it returns for several
        # unrelated situations (see _map_token_error); its error_description is
        # the only way to tell them apart, so it is logged there, sanitized.
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
        raise _map_token_error(resp.status_code, payload, kind)

    try:
        success_payload = resp.json()
    except ValueError as exc:
        # The error path already tolerates a non-JSON body; the success path
        # must too, or a 200 with a broken body escapes as an unhandled
        # exception and the caller returns 500 instead of a typed JSON-RPC
        # failure.
        raise OboExchangeError("IdP returned 200 with a non-JSON body") from exc
    if not isinstance(success_payload, dict):
        raise OboExchangeError("IdP returned 200 with a JSON body that is not an object")

    access_token = success_payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise OboExchangeError("IdP returned 200 but no access_token")
    if kind == "keycloak":
        await _refuse_gateway_valid_token(idp_provider, access_token, target_audience)
    return access_token
