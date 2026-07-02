"""Shared gateway-proxy opt-in for registry entities + the SSRF egress guard.

Any entity type (MCP server, A2A agent, skill, virtual server, custom entity)
can opt into being served through the gateway by setting ``is_proxied=True``.
The registry then generates an nginx location block that routes authenticated
traffic to the entity's effective backend URL.

Security posture (this module is the application-layer SSRF control):
- ``_assert_egress_allowed`` rejects loopback / private / reserved / multicast
  targets unless ``gateway_proxy_allow_private_targets`` is set, and ALWAYS
  rejects link-local (which includes the cloud metadata endpoint) and the
  unspecified address regardless of that flag.
- The guard is a best-effort STATIC check on literal-IP targets. Hostnames pass
  here (DNS is resolved downstream by nginx/httpx); the authoritative defense
  against DNS-rebind is the network egress policy (deny metadata/link-local at
  the socket layer) documented in the LLD, not this function.
- The same guard is DESIGNED to run at three points (model validator, repository
  persist, nginx render) so a row written before the policy existed, or by a
  federation sync that bypasses the Pydantic validator, still cannot emit a live
  route to a denied host. This module lands the model validator; the persist-path
  and render-path hooks are added in the federation-enforcement and nginx-render
  changes respectively. The feature ships disabled (gateway_generic_proxy_enabled
  defaults false) until all three are in place.

Note vs. registry/services/ard_net_guard.py: that guard resolves DNS and checks
every resolved IP, defeating rebind at check time. This guard deliberately does
NOT resolve (proxy targets are long-lived and re-validated by a refresh + the
network egress policy), so a hostname target is only network-layer-protected.
It is therefore a WEAKER static check than ard_net_guard by design; do not enable
the feature until the network egress policy (SSRF layer 1) is deployed.
"""

import logging
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from registry.utils.ip_guard import coerce_ip_literal, ip_denial_reason

logger = logging.getLogger(__name__)

# Canonical built-in entity-type tokens that resolve_proxy_target knows how to
# derive a fallback backend URL for. Custom entities pass their own descriptor
# name as the type token and must always carry an explicit proxy_target_url.
# Kept as the single source of truth so a typo in a caller cannot silently
# resolve to "not proxyable" without notice (see resolve_proxy_target).
CANONICAL_ENTITY_TYPES: frozenset[str] = frozenset(
    {"mcp_server", "a2a_agent", "skill", "virtual_server"}
)

# Proxy fields that must NEVER cross the federation boundary (owner decision:
# no proxying federated entities, either direction). Stripped from a payload
# both on INGEST (a synced entity can never become a local gateway route, and a
# peer cannot plant an SSRF target) and on EXPORT (we do not advertise our proxy
# config, incl. the internal proxy_target_url, to peers). is_proxied and
# proxy_target_url are the opt-in; the three proxy_* bookkeeping fields are
# internal refresh state that likewise should not travel. See strip_proxy_fields.
PROXY_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "is_proxied",
        "proxy_target_url",
        "proxy_resolved_ips",
        "proxy_target_host",
        "proxy_disabled_reason",
    }
)


def strip_proxy_fields(doc: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``doc`` with all proxy fields removed.

    Used at the federation boundary in both directions. Removing the keys (vs.
    forcing is_proxied=False) is deliberate: on ingest it leaves any existing
    stored value untouched on an update re-sync (a peer sync must not clobber a
    local admin's opt-in), while a create defaults is_proxied to False; on export
    it simply omits our proxy config from the outbound payload.
    """
    if not any(k in doc for k in PROXY_FIELD_NAMES):
        return doc
    return {k: v for k, v in doc.items() if k not in PROXY_FIELD_NAMES}


class EgressPolicyError(ValueError):
    """Raised when a proxy_target_url resolves to a denied network range.

    Subclasses ValueError so the Pydantic field_validator surfaces it as a 422
    at the API edge, while callers that persist raw dicts (federation) can catch
    it explicitly on the persist path.
    """


def _assert_egress_allowed(url: str) -> None:
    """Reject loopback / link-local / metadata / private / unspecified targets.

    Delegates the IP category logic to ``registry.utils.ip_guard`` (the single
    source of truth for encoding/NAT64/mapped bypasses). When
    ``settings.gateway_proxy_allow_private_targets`` is true the
    private/loopback/reserved/multicast checks are skipped (for trusted
    on-cluster service URLs), but link-local (the cloud metadata endpoint) and
    the unspecified address are ALWAYS denied.

    Literal-IP hosts (including obfuscated spellings) are category-checked here.
    Genuine hostnames are allowed through (DNS is resolved downstream); see the
    module docstring for why the real rebind defense is the network egress
    policy, not this static check.

    Args:
        url: The candidate proxy target URL.

    Raises:
        ValueError: If the scheme is not http(s).
        EgressPolicyError: If the target host is in a denied network range.
    """
    # Imported here (not at module import) so tests/callers that monkeypatch
    # settings see the live value, and to avoid an import cycle with config.
    from registry.core.config import settings

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"proxy_target_url must be an http(s) URL, got scheme {parsed.scheme!r}")

    host = (parsed.hostname or "").strip("[]")
    ip = coerce_ip_literal(host)
    if ip is None:
        # Genuine hostname -> resolved downstream; static guard passes. The
        # authoritative rebind defense is the network egress policy, not this.
        return

    reason = ip_denial_reason(ip, allow_private=settings.gateway_proxy_allow_private_targets)
    if reason is not None:
        raise EgressPolicyError(
            f"proxy_target_url host {host} is denied ({reason}); "
            "if this is a trusted on-cluster target set GATEWAY_PROXY_ALLOW_PRIVATE_TARGETS=true"
        )


class ProxyableMixin(BaseModel):
    """Shared opt-in fields for serving an entity through the gateway.

    Set ``is_proxied=True`` to have the registry generate an nginx location
    block routing authenticated traffic to ``proxy_target_url``. For MCP servers
    and A2A agents, ``proxy_target_url`` falls back to their native backend URL
    (``proxy_pass_url`` / ``url``) when unset; skills and custom entities must
    set it explicitly (see ``resolve_proxy_target``).
    """

    is_proxied: bool = Field(
        default=False,
        description="When true, the registry generates a gateway route for this entity.",
    )
    proxy_target_url: str | None = Field(
        default=None,
        description=(
            "Backend HTTP(S) URL the gateway forwards to. Required when is_proxied "
            "is true and the entity has no native backend URL (skills, custom entities)."
        ),
    )
    # Operational bookkeeping written by the resolve-and-validate refresh; not user-set.
    proxy_resolved_ips: list[str] = Field(
        default_factory=list,
        description="IPs the hostname target last resolved to (egress re-validation bookkeeping).",
    )
    proxy_target_host: str | None = Field(
        default=None,
        description="Original hostname of proxy_target_url (preserved for Host/SNI; informational).",
    )
    proxy_disabled_reason: str | None = Field(
        default=None,
        description=(
            "Set when the refresh auto-disables proxying (e.g. target now resolves "
            "to a denied IP). When non-null the entity is treated as NOT proxied."
        ),
    )

    # NOTE: no raising field_validator on proxy_target_url here. Storage models
    # inherit this mixin and are RECONSTRUCTED from the DB on every read; a
    # raising egress check would make a bypass-written denied literal (federation
    # raw-doc write, migration, manual edit) throw on read and silently vanish the
    # entity from every listing. The egress raise is instead an API-EDGE fast-fail
    # on the request/patch models (which never reconstruct from the DB) via
    # ``egress_guard_validator``, and the authoritative net is the persist/render
    # guard. See assert_proxy_target_resolvable, which
    # is likewise read-safe.


def egress_guard_validator(v: str | None) -> str | None:
    """Reusable field-validator body: raise on a denied proxy_target_url.

    Attach to a request/patch model's ``proxy_target_url`` field (an API edge that
    never reconstructs from stored data) so a bad target fails fast with 422.
    Do NOT attach to a storage model — see the ProxyableMixin note above.
    """
    if v is None:
        return v
    _assert_egress_allowed(v)
    return v


def _is_federated(doc: dict[str, Any]) -> bool:
    """True if the entity was synced from a peer (sync_metadata.is_federated).

    Federated entities are never local gateway routes (owner decision), so this
    gates resolve_proxy_target regardless of a locally-flipped is_proxied.
    """
    meta = doc.get("sync_metadata")
    return bool(isinstance(meta, dict) and meta.get("is_federated"))


def resolve_proxy_target(
    entity_type: str,
    doc: dict[str, Any],
) -> str | None:
    """Return the effective backend URL for a proxied entity, or None.

    Returns None (entity is not gateway-served) when: it is not flagged
    ``is_proxied``; the refresh auto-disabled it (``proxy_disabled_reason`` set);
    or its type has no resolvable target (a local-deployment MCP server has no
    HTTP backend; a skill/custom entity without an explicit ``proxy_target_url``).

    SECURITY: this function does NOT re-run the egress guard — it returns the
    stored target verbatim. A document written straight to the DB (federation
    sync, a pre-feature migration) can carry a denied target that the model
    validator never saw. Any caller that turns this URL into a live route (nginx
    render, the proxy hop) MUST pass it through ``_assert_egress_allowed`` first.
    The render-path hook is the enforcement point; do not wire this into a route
    generator without it.

    Args:
        entity_type: Canonical entity-type token (mcp_server, a2a_agent, skill, ...).
        doc: The stored entity document (or projection) as a dict.

    Returns:
        The effective backend URL to forward to, or None if not proxyable.
    """
    if not doc.get("is_proxied"):
        return None
    if _is_federated(doc):
        # ABSOLUTE isolation (owner decision): a federated entity is never a
        # local gateway route, even if a local admin flips is_proxied=True on the
        # synced record after ingest. Enforced here at the resolve chokepoint (in
        # addition to the strip-on-ingest boundary) so there is no path from
        # peer-supplied data to a live route. The proxy_target_url was already
        # stripped on ingest, so without this a proxied synced record would fall
        # back to the PEER-SUPPLIED proxy_pass_url/url below.
        return None
    if "is_enabled" in doc and not doc["is_enabled"]:
        # A disabled entity is not reachable, so it gets no gateway route. Gated
        # only when the caller supplied is_enabled (list_proxied projects it);
        # absent means "not my decision" (a partial projection), so we don't
        # assume disabled.
        return None
    if doc.get("proxy_disabled_reason"):
        return None  # refresh auto-disabled this route
    explicit = doc.get("proxy_target_url")
    if explicit:
        return explicit
    if entity_type == "mcp_server":
        if doc.get("deployment") == "local":
            return None  # stdio servers have no HTTP backend
        return doc.get("proxy_pass_url")
    if entity_type == "a2a_agent":
        return doc.get("url")
    return None  # skills / custom entities must set proxy_target_url explicitly


def proxy_target_missing(
    entity_type: str,
    doc: dict[str, Any],
) -> bool:
    """Return True if ``is_proxied`` is set but no backend target can be resolved.

    Pure predicate (no raise/log). Exempt states (``is_proxied`` is a dormant
    no-op, NOT missing): not proxied at all; ``proxy_disabled_reason`` set (the
    documented auto-disabled state); a local-deployment MCP server (no HTTP
    backend). All other proxied entities must resolve to a target.
    """
    if not doc.get("is_proxied"):
        return False
    if doc.get("proxy_disabled_reason"):
        return False  # auto-disabled route: dormant opt-in
    if entity_type == "mcp_server" and doc.get("deployment") == "local":
        return False  # stdio server: is_proxied is a documented no-op
    return resolve_proxy_target(entity_type, doc) is None


def assert_proxy_target_resolvable(
    entity_type: str,
    doc: dict[str, Any],
    *,
    read_safe: bool = False,
) -> None:
    """Enforce "if proxied, a target must resolve", turning the silent
    "I set is_proxied=True and nothing happened" failure into an error.

    Called from a model's ``model_validator(mode="after")`` so the check sees the
    model's native fallback fields (proxy_pass_url / url).

    ``read_safe`` picks the behavior by context:
    - STORAGE models pass ``read_safe=True``: a missing target is LOGGED, not
      raised. Storage models are reconstructed from the stored document on every
      READ, and a bypass write (federation sync copying is_proxied without a
      target, a migration, a manual edit) could produce the missing-target state.
      Raising would make the entity throw on load and silently vanish from every
      listing. The route simply won't render (resolve_proxy_target returns None).
    - REQUEST/PATCH models pass ``read_safe=False`` (default): a missing target is
      a 422 at the API edge, where there is no vanish risk (the model is built
      from a client payload, never from stored data).

    Args:
        entity_type: Canonical entity-type token.
        doc: The model's proxy-relevant scalars as a dict.
        read_safe: When True, log instead of raise (storage/read context).

    Raises:
        ValueError: If proxied but no target resolves and ``read_safe`` is False.
    """
    if not proxy_target_missing(entity_type, doc):
        return
    msg = (
        f"is_proxied=true on {entity_type} requires a resolvable backend URL: set "
        "proxy_target_url (or the entity's native backend URL) to a valid http(s) target"
    )
    if read_safe:
        logger.warning(
            "Loaded proxied %s with no resolvable target (route will not render): %s",
            entity_type,
            msg,
        )
        return
    raise ValueError(msg)
