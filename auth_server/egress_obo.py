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
"""

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

    Keycloak's standard token exchange authenticates the gateway's own client
    via ``client_id``/``client_secret`` form fields, carries the caller's
    ingress access token as ``subject_token`` (typed as an access_token), and
    names the target as ``audience`` — the bare target client id, not an
    https URL and not Entra's ``assertion``/``scope=api://.../.default``
    convention.

    ``requested_token_type`` is pinned rather than left to the server default,
    because that default differs by Keycloak generation: legacy exchange
    (<= 26.1) defaults to ``refresh_token`` and would mint a refresh token this
    code reads past and discards on every request, while standard exchange
    (26.2+) defaults to ``access_token``. RFC 8693 §2.1 makes the parameter
    OPTIONAL with a server-chosen default, which is exactly why relying on it
    is an interop hazard.

    ``scope`` is sent only when explicit scopes are requested; omitting it
    makes Keycloak apply the **requesting** client's default client scopes —
    the gateway's own client, not the audience client.
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


def _map_token_error(status_code: int, payload: dict, kind: str = "") -> OboExchangeError:
    """Map an IdP token-endpoint error response to a typed exception.

    ``kind`` is the IdP family (``entra``/``keycloak``). It exists so a
    provider-specific remediation hint never reaches an operator running the
    other IdP: the codes overlap but their causes and fixes do not. Codes whose
    meaning is provider-independent stay in the shared branches below.
    """
    err = (payload.get("error") or "").strip()
    if err == "interaction_required":
        return OboConsentRequired("IdP requires consent")
    if err in ("invalid_grant", "invalid_token"):
        # invalid_grant spans both user-fixable (expired/no-permission) and
        # config (gateway not granted access) cases; re-auth is the safer of the
        # two, since retrying with a fresh token is cheap and a config problem
        # simply fails again. The IdP's error_description is logged, not
        # returned: it is operator diagnostics, not something the calling agent
        # can act on.
        # Keycloak never answers invalid_grant for token-exchange: legacy
        # exchange reports an expired or unusable subject_token as
        # invalid_token, which is the same user-fixable situation.
        return OboReauthRequired("IdP rejected the user assertion")
    if err == "unsupported_grant_type":
        # The token-exchange grant is not enabled on the server at all — on
        # Keycloak <= 26.1 that means KC_FEATURES lacks token-exchange.
        return OboConfigError(
            f"IdP does not support the token-exchange grant "
            f"(unsupported_grant_type, status={status_code})"
        )
    if err == "access_denied" and kind == "keycloak":
        # Keycloak answers access_denied when the exchange is not permitted,
        # and what "permitted" means depends on the server generation, so name
        # both rather than asserting one: legacy exchange (<= 26.1) wants the
        # token-exchange permission on the TARGET client, while standard
        # exchange (26.2+) wants the gateway's own client inside the subject
        # token's audience. Both are operator-actionable configuration.
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
    # invalid_request is deliberately NOT classified. Keycloak standard exchange
    # (26.2+) answers it for at least three unrelated situations: an expired
    # subject_token, the client's standard-token-exchange toggle being off, and
    # a requested audience that cannot be placed in the token. Two are operator
    # config and one is user re-auth, and the code alone cannot tell them apart
    # — only error_description can, which is why it is logged at the call site.
    # Guessing here would be worse than the generic error: classifying an
    # expired token as config would stop the caller retrying with a fresh one.
    return OboExchangeError(
        f"IdP token exchange failed (status={status_code}, error={err or 'unknown'})"
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
    from registry.utils.url_guard import guarded_async_client

    try:
        async with guarded_async_client(
            profile=CREDENTIALED_OAUTH_PROFILE,
            timeout=_TOKEN_EXCHANGE_TIMEOUT_SECONDS,
        ) as client:
            resp = await client.post(token_url, data=body)
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
        # error_description is the only way to tell apart the situations that
        # share an error code (notably invalid_request on standard exchange).
        # It carries IdP configuration text, never a token or a secret, and is
        # truncated so a verbose IdP cannot flood the log.
        description = str(payload.get("error_description") or "")[:200]
        logger.warning(
            "obo_exchange: IdP token exchange failed status=%s error=%s description=%s",
            resp.status_code,
            payload.get("error") or "unknown",
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

    access_token = success_payload.get("access_token")
    if not access_token:
        raise OboExchangeError("IdP returned 200 but no access_token")
    return access_token
