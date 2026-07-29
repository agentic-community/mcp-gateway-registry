"""Shared gateway-proxy opt-in for registry entities + the SSRF egress guard.

Any entity type (MCP server, A2A agent, skill, virtual server, custom entity)
can opt into being served through the gateway by setting ``is_proxied=True``.
The registry then generates an nginx location block that routes authenticated
traffic to the entity's effective backend URL.

Security posture (this module wires the entity/proxy layer to the SSRF control):
- IP classification + the egress allowlist are owned by the canonical
  ``registry.utils.url_guard``, including its inline literal parser and IP-category
  classifier. The functions here are thin PROXY_PROFILE adapters that translate
  the guard's ``UrlValidationError`` into the local ``EgressPolicyError`` (a
  ``ValueError``) so a Pydantic field validator surfaces a clean 4xx. Registration
  and the fetch-time pinned transport therefore share ONE policy — the
  ``gateway_proxy_allow_private_targets`` bool plus ``ssrf_allowed_hosts`` /
  ``ssrf_allowed_cidrs`` can relax only loopback/private/CGNAT targets;
  cloud/workload credential endpoints (including Alibaba), link-local, reserved,
  multicast, and unspecified destinations remain hard-denied, including scoped
  and embedded/tunneled IPv6 forms.
- The static check is best-effort on literal-IP targets; hostnames pass it (no DNS
  in a validator). The authoritative rebind defense is the fetch-time pinned
  transport (``url_guard.guarded_async_client``, which resolves+validates+connects
  inside the transport call for every request and redirect hop) plus the network
  egress policy — not this static edge check.
- The guard runs at three points (model validator, service resolve-and-pin, nginx
  render) so a row written before the policy existed, or by a federation sync that
  bypasses the Pydantic validator, still cannot emit a live route to a denied host.

Two application-layer checks, by context:
- ``_assert_egress_allowed`` / ``egress_guard_validator`` — the STATIC (literal-IP)
  check at the Pydantic API edge (422), via ``url_guard.validate_url(resolve=False)``.
- ``resolve_and_validate_proxy_target`` / ``validate_and_pin_proxy_target`` — the
  DNS-aware check (layer 2), run at the service register/update layer and by
  the scheduled pin-refresh, via ``url_guard.validate_url(resolve=True)``. Resolves
  the hostname, validates every resolved IP, and returns them for pin bookkeeping.
"""

import logging
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from registry.utils.url_guard import coerce_ip_literal

logger = logging.getLogger(__name__)


class CustomHeaderEncrypted(BaseModel):
    """Stored form of a custom upstream header: name + Fernet-encrypted value.

    Defined here (not core.schemas) so every proxyable entity can carry custom
    upstream headers via ProxyableMixin without a schemas<->proxy_mixin import
    cycle. core.schemas re-exports it for backward compatibility.
    """

    name: str
    value_encrypted: str


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
        "proxy_streaming",
        "proxy_resolved_ips",
        "proxy_target_host",
        "proxy_disabled_reason",
        "proxy_client_url",
        # Upstream auth secrets must NEVER travel to/from a peer. Stripped both
        # ways (a peer must not plant creds, and we must not leak our own).
        # The plaintext "custom_headers" key is normally never persisted, but is
        # stripped too as defense-in-depth in case a raw-doc path ever carries it.
        "custom_headers",
        "custom_headers_encrypted",
        "custom_header_names",
        "custom_header_overridable_names",
        "custom_headers_updated_at",
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


def _target_identity(url: str | None) -> str:
    """Return normalized full-URL credential identity, or a fail-closed marker."""
    if not url:
        return ""
    from registry.exceptions import UrlValidationError
    from registry.utils.url_guard import normalize_url_identity

    try:
        return normalize_url_identity(url)
    except UrlValidationError:
        # Invalid legacy values should never share credentials with a replacement.
        # Include the original value so an unchanged malformed row remains stable,
        # while any edit necessarily changes identity and clears credentials.
        return f"invalid:{url}"


def clear_upstream_headers_on_repoint(
    updates: dict[str, Any],
    existing_target: str | None,
    new_target: str | None,
) -> None:
    """Clear stored upstream custom headers when an update repoints the target.

    Credential-misdirection guard (shared by every entity update path). The
    encrypted ``custom_headers`` were entered by an operator for one exact
    normalized upstream URL. A scheme, host, port, path, or query change can
    repoint the credential to a different tenant/resource even on the same host,
    so every full-target identity change clears the stored headers.

    No-op only when the normalized full URLs are identical (for example, host
    case or an explicit default port differs) or when there is nothing to clear.
    """
    if _target_identity(existing_target) == _target_identity(new_target):
        return
    updates["custom_headers_encrypted"] = None
    updates["custom_header_names"] = []
    updates["custom_header_overridable_names"] = []
    updates["custom_headers_updated_at"] = None


class EgressPolicyError(ValueError):
    """Raised when a proxy_target_url resolves to a denied network range.

    Subclasses ValueError so the Pydantic field_validator surfaces it as a 422
    at the API edge, while callers that persist raw dicts (federation) can catch
    it explicitly on the persist path.
    """


def _assert_egress_allowed(url: str) -> None:
    """Reject loopback / link-local / metadata / private / unspecified targets.

    Thin adapter over the canonical ``url_guard.validate_url`` (PROXY_PROFILE,
    STATIC / no-DNS): the URL guard owns the classification (obfuscated literals,
    embedded-v4, the non-overridable metadata deny) AND the egress allowlist (the
    ``gateway_proxy_allow_private_targets`` bool + ``ssrf_allowed_hosts/cidrs``), so
    registration and the fetch-time pinned transport share ONE policy. This adapter
    exists only to (a) bind PROXY_PROFILE and (b) translate the guard's
    ``UrlValidationError`` (a bare ``RegistryError``) into ``EgressPolicyError`` (a
    ``ValueError``) so a Pydantic field validator surfaces it as a clean 4xx rather
    than a 500.

    Literal-IP hosts (including obfuscated spellings) are category-checked here.
    Genuine hostnames pass the static check (DNS is resolved downstream); the
    authoritative rebind defense is the pinned transport at fetch time + the network
    egress policy, not this static check.

    Args:
        url: The candidate proxy target URL.

    Raises:
        EgressPolicyError: If the scheme is not http(s) or the target host is in a
            denied network range (both are ``ValueError`` subclasses).
    """
    from registry.exceptions import UrlValidationError
    from registry.utils.url_guard import PROXY_PROFILE, validate_url

    try:
        validate_url(url, profile=PROXY_PROFILE, resolve=False)
    except UrlValidationError as e:
        raise EgressPolicyError(
            f"proxy_target_url rejected: {e.reason}; if this is a trusted on-cluster "
            "target set GATEWAY_PROXY_ALLOW_PRIVATE_TARGETS=true or add it to "
            "SSRF_ALLOWED_HOSTS / SSRF_ALLOWED_CIDRS"
        ) from e


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
    proxy_streaming: bool = Field(
        default=False,
        description=(
            "When true, the generic hop streams the upstream response chunk-by-chunk "
            "(SSE / chunked) instead of buffering it, and the generated nginx route "
            "disables proxy_buffering with a long read timeout. Set this for "
            "long-lived or token-streaming backends (e.g. an LLM proxied as a custom "
            "type). When false (default) the response is buffered up to "
            "GENERIC_PROXY_MAX_BODY_BYTES. The value is bound into the signed "
            "generic-proxy token, so the hop never trusts a forgeable inbound header."
        ),
    )
    custom_headers_encrypted: list[CustomHeaderEncrypted] | None = Field(
        default=None,
        description=(
            "Static upstream auth headers presented to the backend the gateway "
            "proxies to (e.g. an API key for an LLM proxied as a custom type). "
            "Stored as {name, value_encrypted} (Fernet); the plaintext values are "
            "NEVER serialized to API consumers, rendered into nginx, or logged. At "
            "request time the generic hop fetches the decrypted headers from the "
            "registry's internal vend endpoint (same trust boundary as the egress "
            "vault) and injects them on egress. Accepted on create/update via a "
            "plaintext 'custom_headers' list that the registration path encrypts."
        ),
    )
    custom_header_names: list[str] = Field(
        default_factory=list,
        description="Names of the custom upstream headers (values are encrypted). Safe to surface.",
    )
    custom_header_overridable_names: list[str] = Field(
        default_factory=list,
        description=(
            "Subset of custom_header_names the CALLER may override or supply on the "
            "request (the entries registered with overridable=true). This IS the "
            "caller passthrough allowlist: on egress the generic hop forwards a "
            "caller-supplied header ONLY if its name is in this set; a name here with "
            "a stored encrypted value is an operator DEFAULT the caller may override, "
            "and a name here WITHOUT a stored value is a caller-only slot. Names only, "
            "no secret. Safe to surface on reads."
        ),
    )
    custom_headers_updated_at: str | None = Field(
        default=None,
        description="ISO timestamp of the last custom-headers update.",
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
    # DERIVED, read-only: the client-facing gateway path ({prefix}/{type}/{name}).
    # Never user-set — it is recomputed from the entity's type + path on every
    # construction (see populate_proxy_client_url, called from each storage
    # model's after-validator), so any client-supplied or stored value is
    # overwritten. Distinct from proxy_target_url (the backend/origin the gateway
    # forwards to). Null when the entity is not proxied.
    proxy_client_url: str | None = Field(
        default=None,
        description=(
            "Read-only, auto-derived client-facing gateway path "
            "(/{gateway_proxy_prefix}/{entity_type}/{name}). Clients connect here; "
            "the registry forwards to proxy_target_url. Recomputed on every read — "
            "any value sent by a client is ignored."
        ),
    )

    def populate_proxy_client_url(
        self,
        entity_type: str,
    ) -> None:
        """Set ``proxy_client_url`` from the entity's type + path, or clear it.

        Called from each storage model's ``model_validator(mode="after")`` so the
        derived client path is present in ``model_dump`` for API reads and stays
        self-healing (recomputed on every construction, never trusted from the DB
        or a client payload). Populated whenever ``is_proxied`` is set and the
        model carries a non-empty ``path``; cleared to None otherwise.

        Args:
            entity_type: The entity-type token used as the URL's type segment
                (must match the authz key, e.g. "skill", "a2a_agent", or a custom
                type name).
        """
        path = getattr(self, "path", None)
        if self.is_proxied and path:
            from registry.core.config import settings

            self.proxy_client_url = build_proxy_client_path(
                entity_type,
                path,
                settings.gateway_proxy_prefix,
            )
        else:
            self.proxy_client_url = None

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

    NOTE: this is the STATIC (literal-IP) check only. A genuine hostname passes
    here and is resolved-and-validated separately by
    ``resolve_and_validate_proxy_target`` at the service/registration layer (DNS
    is I/O and cannot run in a Pydantic validator).
    """
    if v is None:
        return v
    _assert_egress_allowed(v)
    return v


async def resolve_and_validate_proxy_target(
    url: str,
    *,
    profile: Any = None,
) -> tuple[str | None, list[str]]:
    """Resolve a proxy_target_url's hostname and validate every resolved IP.

    This is **layer 2** — the registration-time (and refresh-time)
    DNS-aware SSRF check. It resolves the hostname and validates EVERY resolved IP
    via the canonical ``url_guard.validate_url`` (PROXY_PROFILE, resolve=True), so
    it shares ONE classifier + egress allowlist with the fetch-time pinned
    transport (no register-vs-fetch policy drift). It is best-effort against
    DNS-rebind (the resolved set can change before the real request) — which is why
    the authoritative control is the pinned transport at fetch time + the network
    egress policy (layer 1), not this.

    Args:
        url: The candidate proxy target URL.

    Returns:
        ``(hostname, resolved_ips)`` — the original hostname (or None for a
        literal-IP target) and the list of resolved IP strings (the literal
        itself for an IP target; empty when the host is operator-allowlisted, which
        ``validate_url`` returns without resolving). Callers persist these for pin
        bookkeeping.

    Raises:
        EgressPolicyError: If the scheme is not http(s), the host cannot be
            resolved, or any resolved IP is in a denied network range (a
            ``ValueError`` subclass, so a caller mapping ValueError -> 4xx works).
    """
    import asyncio

    from registry.exceptions import UrlValidationError
    from registry.utils.url_guard import PROXY_PROFILE, validate_url

    selected_profile = profile or PROXY_PROFILE
    host = (urlparse(url).hostname or "").strip("[]")
    is_literal = coerce_ip_literal(host) is not None
    try:
        # resolve=True performs getaddrinfo + validates every resolved IP, and
        # returns the validated IP list (the literal itself for an IP target, or
        # [] for an operator-allowlisted host it does not pin). validate_url is
        # SYNCHRONOUS (socket.getaddrinfo), so run it in a worker thread — a slow
        # or adversarial DNS answer must not block the event loop on this async
        # register/refresh path.
        ips = await asyncio.to_thread(
            validate_url,
            url,
            profile=selected_profile,
            resolve=True,
        )
    except UrlValidationError as e:
        raise EgressPolicyError(f"proxy_target_url rejected: {e.reason}") from e

    # A literal-IP target has no hostname to preserve for the pin (Host/SNI); a
    # genuine hostname is returned so callers can persist it for pin bookkeeping.
    return (None if is_literal else host), ips


async def validate_and_pin_proxy_target(
    entity_type: str,
    doc: dict[str, Any],
) -> dict[str, Any]:
    """Registration/refresh helper: resolve+validate the effective proxy target
    and return the pin-bookkeeping fields to persist.

    Computes the effective target via ``resolve_proxy_target`` (so it inherits the
    federated / disabled / auto-disabled / target-required gating and the native
    fallback), and — only when a real target exists — resolves its hostname and
    validates every IP against the egress policy (layer 2). Returns the
    fields the caller should merge into the persisted document; an empty dict when
    the entity is not proxied / has no resolvable target (nothing to pin).

    Call this at the service create/update layer BEFORE persisting, so a target
    that resolves to a metadata/private IP is rejected with a clear error at
    register/update instead of silently dropped at render.

    Args:
        entity_type: Canonical entity-type token.
        doc: The proxy-relevant scalars (same shape as assert_proxy_target_resolvable).

    Returns:
        ``{"proxy_resolved_ips": [...], "proxy_target_host": "..."}`` when a target
        was validated, else ``{}``.

    Raises:
        ValueError / EgressPolicyError: If the target is denied or unresolvable.
    """
    target = resolve_proxy_target(entity_type, doc)
    if not target:
        return {}
    from registry.utils.url_guard import proxy_profile_for_entity_target

    profile = proxy_profile_for_entity_target(entity_type, doc.get("path"), target)
    host, ips = await resolve_and_validate_proxy_target(target, profile=profile)
    return {"proxy_resolved_ips": ips, "proxy_target_host": host or ""}


def build_proxy_client_path(
    entity_type: str,
    path: str,
    prefix: str,
) -> str:
    """Return the client-facing gateway path for a proxied entity.

    This is the URL a CLIENT connects to — distinct from ``proxy_target_url``
    (the backend/origin the gateway forwards to). It is ALWAYS auto-derived, so
    operators never hand-enter it: they only supply the origin.

    Shape: ``/{prefix}/{entity_type}/{name}`` where ``{name}`` is the registered
    ``path`` with its leading namespace segment stripped, so a skill at
    ``/skills/pdf-processing`` (entity_type ``skill``) yields
    ``/gateway/skill/pdf-processing`` — not the doubled ``/skill/skills/...``.
    Uniqueness follows from ``path`` being unique per type (the Mongo ``_id``),
    so the stripped name cannot collide within a type's namespace.

    Args:
        entity_type: The entity-type token (e.g. "skill", "a2a_agent", or a
            custom type name). Used verbatim as the URL's type segment so it
            matches the authz key (``X-Generic-Proxy-Kind``).
        path: The registered entity path (e.g. "/skills/pdf-processing").
        prefix: The configured gateway path prefix (settings.gateway_proxy_prefix).

    Returns:
        The client-facing path, leading-slash-prefixed and with no trailing slash.
    """
    clean = path.strip("/")
    # Drop the leading namespace segment (skills/, agents/, {custom-type}/) so the
    # type is not doubled; keep any remaining structure (nested paths stay unique).
    _head, _sep, rest = clean.partition("/")
    name = rest if rest else clean
    return f"/{prefix.strip('/')}/{entity_type}/{name}"


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


def effective_proxy_target(entity_type: str, doc: dict[str, Any]) -> str | None:
    """The target-IDENTITY of an entity, WITHOUT the routability gates.

    Same target-derivation as ``resolve_proxy_target`` (explicit
    ``proxy_target_url``, else the type's native fallback) but deliberately
    OMITS the is_enabled / is_proxied / proxy_disabled_reason / federated
    short-circuits. Use this ONLY to answer "does this update repoint the
    exact backend URL?" for the credential-misdirection guard: a disabled or
    not-yet-enabled server can still be repointed, and its dormant upstream
    headers must be cleared on that host change even though it currently
    resolves to no live route. Never use this to decide whether to EMIT a route
    (that must stay ``resolve_proxy_target``, which is fail-closed).
    """
    explicit = doc.get("proxy_target_url")
    if explicit:
        return explicit
    if entity_type == "mcp_server":
        if doc.get("deployment") == "local":
            return None
        return doc.get("proxy_pass_url")
    if entity_type == "a2a_agent":
        return doc.get("url")
    return None


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
