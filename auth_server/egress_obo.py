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

logger = logging.getLogger(__name__)

# OAuth grant types for the two supported IdPs.
_ENTRA_JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_RFC8693_TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
_RFC8693_ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"

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
    """Build the Keycloak RFC 8693 token-exchange request body.

    Phase 4 (follow-on). Keycloak uses ``subject_token``/``audience`` (the bare
    target client id), NOT Entra's ``assertion``/``scope=api://.../.default``.
    """
    raise OboUnsupportedIdpError(
        "Keycloak OBO token-exchange (RFC 8693) is not yet implemented; "
        "Entra (jwt-bearer) ships first. Tracked as Phase 4."
    )


def _map_token_error(status_code: int, payload: dict) -> OboExchangeError:
    """Map an IdP token-endpoint error response to a typed exception."""
    err = (payload.get("error") or "").strip()
    detail = payload.get("error_description") or payload.get("error") or "exchange failed"
    if err == "interaction_required":
        return OboConsentRequired(detail)
    if err == "invalid_grant":
        # invalid_grant spans both user-fixable (expired/no-permission) and
        # config (gateway not granted access) cases; surface as re-auth with the
        # IdP detail so the agent/user sees the actual reason.
        return OboReauthRequired(detail)
    if err in ("invalid_client", "invalid_scope", "unauthorized_client"):
        return OboConfigError(detail)
    return OboExchangeError(f"idp returned {status_code}: {detail}")


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
    from registry.exceptions import UrlValidationError
    from registry.utils.url_guard import PROXY_PROFILE, guarded_async_client

    try:
        async with guarded_async_client(
            profile=PROXY_PROFILE, timeout=_TOKEN_EXCHANGE_TIMEOUT_SECONDS
        ) as client:
            resp = await client.post(token_url, data=body)
    except UrlValidationError as exc:
        # Guard rejected the target WITHOUT sending the credential/assertion.
        logger.error("obo_exchange: token endpoint blocked by SSRF guard: %s", exc)
        raise OboExchangeError(f"IdP token endpoint blocked by SSRF guard: {exc}") from exc
    except httpx.HTTPError as exc:
        logger.error("obo_exchange: transport error calling IdP token endpoint: %s", exc)
        raise OboExchangeError(f"IdP token endpoint unreachable: {exc}") from exc

    if resp.status_code != 200:
        try:
            payload = resp.json()
        except ValueError:
            payload = {"error_description": resp.text[:200]}
        logger.warning(
            "obo_exchange: IdP returned %s: %s",
            resp.status_code,
            payload.get("error") or payload.get("error_description"),
        )
        raise _map_token_error(resp.status_code, payload)

    access_token = resp.json().get("access_token")
    if not access_token:
        raise OboExchangeError("IdP returned 200 but no access_token")
    return access_token
