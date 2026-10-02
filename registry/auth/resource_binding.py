"""This module centralizes the logic for:
    1. Validating that a user can be issued a token bound to a given
       (resource_type, resource_id) before the token is minted.
    2. Normalizing resource ids so that tokens minted with "/foo" and "foo"
       are treated identically.
    3. Classifying a request URL into a (resource_type, resource_id) pair so
       that route guards can check it against the token's claims.
    4. Deciding whether a path is reachable at all by a resource-bound token
       (vs. blocked endpoints like /api/tokens/generate or /api/admin/*).

``ResourceType`` is the canonical enum shared between the mint endpoint in
``auth_server`` and the edge guard in this module. Keep the enum values
stable — they are persisted inside JWT claims.

Design note — access revocation / TOCTOU:
    Access is verified once at mint time (via
    ``validate_user_can_bind_resource``). At runtime the edge guard only
    checks that the token's claims match the requested URL; it does not
    re-authorize against the user's current scopes. This mirrors how
    regular user JWTs behave — revoking a scope does not invalidate
    already-issued tokens until they expire. The mitigation is the default
    short TTL (8 hours). A future revocation list (jti deny-list) could
    close this window if stronger guarantees are required.
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
import re
from enum import Enum
from typing import Any, Final
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# Collapse runs of consecutive '/' into a single slash. Defense-in-depth
# against a deployment where the upstream proxy is not configured to merge
# slashes (nginx ``merge_slashes on`` is default but not universal; Traefik,
# Envoy, and custom-compiled nginx can disable it). Without this, a request
# like "//api//tokens//generate" would bypass prefix-based deny-list checks
# because ``startswith("/api/tokens")`` returns False. Applied at the top of
# every helper that consumes a URL path so the rest of the logic can assume
# canonical single-slash form.
_CONSECUTIVE_SLASHES = re.compile(r"/{2,}")


def _normalize_path(path: str) -> str:
    """Canonicalize a URL path for prefix/allow-list matching.

    Strips query string, percent-decodes, resolves ``.``/``..`` segments to
    a canonical absolute path, guarantees a leading slash, and collapses
    consecutive slashes.

    Percent-decode-then-resolve is required for correctness of every
    authorization decision that consumes this path (the resource-token
    deny-list and the (type, id) classifier). The raw request URI is
    attacker-controlled and arrives here percent-encoded (nginx forwards
    ``$request_uri`` via ``X-Original-URL``/``X-Original-URI``). If we matched
    on the raw bytes, an encoded-traversal URL such as
    ``/api/agents/foo/%2e%2e/tokens/generate`` would classify differently from
    its canonical form ``/api/tokens/generate`` and slip past the byte-exact
    deny-list. Decoding once and normalizing segments makes an encoded path
    classify identically to its canonical form, and clamps a traversal that
    tries to escape above root (``/api/../../etc`` -> ``/etc``) rather than
    silently letting it through.

    Fail closed: decoding is done exactly ONCE (a single ``unquote`` — we do
    not re-decode the result, so a doubly-encoded ``%252e`` stays literal and
    still fails classification rather than resolving to ``..``). Segment
    resolution never rises above root. The existing leading-slash and
    consecutive-slash canonicalization is preserved.
    """
    if "?" in path:
        path = path.split("?", 1)[0]
    # Decode percent-escapes exactly once so encoded traversal / deny-listed
    # segments are seen in their canonical form by the matchers below.
    path = unquote(path)
    if not path.startswith("/"):
        path = "/" + path
    # Collapse consecutive slashes before segment resolution so runs like
    # "//api//foo" and "/api/./foo" canonicalize consistently.
    path = _CONSECUTIVE_SLASHES.sub("/", path)
    # Resolve "." and ".." segments. posixpath.normpath operates on the
    # already-absolute path, so it can never escape above root ("/.." -> "/").
    # It strips a trailing slash and may collapse a leading "//"; we only pass
    # a single-leading-slash path in, and re-collapse afterwards, so host
    # semantics are never affected (this is a path, not a URL authority).
    normalized = posixpath.normpath(path)
    # normpath drops a trailing slash and returns "." for an empty path; force
    # a leading slash and re-collapse to keep the documented invariants.
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return _CONSECUTIVE_SLASHES.sub("/", normalized)


def normalize_request_path(path: str) -> str:
    """Public form of ``_normalize_path``: the canonical path nginx routes on.

    EXPORTED for ``auth_server.server``, which must compare a raw
    ``X-Original-URL`` against the location nginx selected on its normalized
    ``$uri`` (decoded once, slashes merged, dot-segments resolved).
    """
    return _normalize_path(path)


def _prepare_path(
    path: str,
    root_path: str = "",
    strip_trailing: bool = True,
) -> str:
    """Shared preamble for the three request-URL helpers.

    Runs ``_normalize_path``, then strips the configured registry
    ``root_path`` prefix (so ``/registry/api/foo`` behaves the same as
    ``/api/foo``), then optionally trims a trailing slash so
    ``/api/auth/me`` and ``/api/auth/me/`` compare equal.

    ``classify_request_url`` passes ``strip_trailing=False`` because it
    discards trailing empty segments via ``split("/")`` anyway, and
    stripping would hide that trailing-slash variants exist from the
    caller.
    """
    path = _normalize_path(path)
    root = (root_path or "").strip().strip("/")
    if root:
        candidate = f"/{root}"
        if path == candidate or path.startswith(candidate + "/"):
            path = path[len(candidate) :] or "/"
    if strip_trailing and len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return path


# Maximum time we will wait for the repository lookups that the pre-mint
# authorization check performs (agent visibility, skill visibility). These
# queries hit MongoDB; without a bound a slow/frozen DB would hang the
# /api/tokens/generate request indefinitely because the request is not
# covered by an outer HTTP timeout. 5 seconds is long enough for a warm
# query and short enough to fail the mint rather than starve the client.
_BIND_CHECK_LOOKUP_TIMEOUT_SECONDS: float = 5.0


class ResourceType(str, Enum):
    """Types of resources a JWT can be bound to.

    Inherits from ``str`` so it serializes natively to JSON and can be
    compared directly to string values (e.g., JWT claim strings, FastAPI
    path params) without explicit ``.value`` access.

    The string values are the wire format persisted in JWT claims and the
    token-generation API; do not rename them.
    """

    SERVER = "server"
    VIRTUAL_SERVER = "virtual_server"
    AGENT = "agent"
    SKILL = "skill"


class TokenKind(str, Enum):
    """Classification of a JWT minted by this gateway.

    ``token_kind`` is emitted on every self-signed JWT and checked by the
    edge guard in ``/validate``.

    ``str`` subclass so JWT/JSON serialization produces a plain string
    and ``claim == TokenKind.USER`` works without explicit ``.value``.

    The string values are the wire format persisted in JWT claims; do
    not rename them.
    """

    USER = "user"
    RESOURCE = "resource"


# JWT claim names.
TOKEN_KIND_CLAIM: Final[str] = "token_kind"
RESOURCE_TYPE_CLAIM: Final[str] = "resource_type"
RESOURCE_ID_CLAIM: Final[str] = "resource_id"

# Sentinel values that different scope/permission stores use to mean
# "no restriction". Checked by ``validate_user_can_bind_resource`` across
# all three accessible-list types; a future config format that
# introduces a new wildcard only needs to be added here.
_WILDCARD_ACCESS: Final[frozenset[str]] = frozenset({"all", "*"})


# Convenience tuple used for error messages and Pydantic validation. Derived
# from the enum so the two can't drift.
RESOURCE_TYPES: tuple[str, ...] = tuple(rt.value for rt in ResourceType)

# URL path prefixes used to classify an incoming request URL into a resource
# type. Order matters: more specific prefixes must come first so that, e.g.,
# /api/agents/foo is classified as an agent rather than a generic server.
# The "server" entry is the catch-all for non-API, non-/virtual/ paths.
#
# EXPORTED — this tuple is imported by ``auth_server.server`` so the
# registry-side pre-mint helpers and the auth-server-side edge guard share
# exactly one definition.
API_PREFIX_TO_TYPE: tuple[tuple[str, ResourceType], ...] = (
    ("/api/agents/", ResourceType.AGENT),
    ("/api/skills/", ResourceType.SKILL),
    # /api/servers/{path} is the REST metadata endpoint for a given
    # server; a server-bound token should be able to inspect its own
    # server (GET) as a normal extension of reaching its MCP path.
    # The resource_id produced here is the plain server slug (no
    # leading slash), matching the mint-time normalization of the
    # ``/path`` field sent by the frontend.
    #
    # Caveat for federated servers: the classifier takes only the
    # first path segment after ``/api/servers/`` as the resource id.
    # A federated server registered at ``/peer-registry/cf-docs``
    # mints a token with id ``peer-registry/cf-docs`` (full path),
    # but ``/api/servers/peer-registry/cf-docs`` would classify as
    # id ``peer-registry``. The mismatch fails closed (403) rather
    # than granting wrong-resource access. Federated-server bound
    # tokens therefore reach the MCP transport (``/peer-registry/
    # cf-docs/mcp``) but not the REST metadata endpoint — a
    # deliberate restriction that mirrors the virtual-server case
    # noted below.
    ("/api/servers/", ResourceType.SERVER),
    # NOTE: no ``/api/virtual-servers/`` mapping. Mint-time normalizes
    # the virtual server id to ``virtual/<slug>`` (with the prefix,
    # matching the MCP gateway path /virtual/<slug>/mcp). A classifier
    # entry for /api/virtual-servers/<slug> would produce ``<slug>``
    # without the prefix and silently fail to match the token claim.
    # Virtual-server tokens are therefore scoped to the MCP path only;
    # REST metadata on /api/virtual-servers/<slug> is unreachable with
    # a bound token — a deliberate restriction.
)

# Transport segments stripped from the tail of MCP server URLs before they
# are treated as a resource id. /cloudflare-docs/mcp -> "cloudflare-docs".
# EXPORTED — see comment on ``API_PREFIX_TO_TYPE``.
MCP_TRANSPORT_SEGMENTS: frozenset[str] = frozenset({"mcp", "sse", "messages"})

# Signed in place of a version id when a virtual mapping pins a version the
# backend does not have. No mcp-proxy token is minted for it and every verifier
# refuses it. EXPORTED: the nginx generator, /validate and both verifiers use it.
INVALID_VERSION_SENTINEL: Final[str] = "__invalid_version__"


def mcp_proxy_route_binding(claims: dict) -> tuple[str, bool] | None:
    """The signed ``(version_id, virtual_backend)`` of an mcp-proxy token.

    One definition for the auth-server verifier, the registry-side verifier and
    the egress vend. A token minted before route binding existed carries neither
    claim (an older auth-server replica during a rolling deploy); it is a direct,
    active-version token, which is all such a replica could mint. Accepting it is
    safe: the vend still requires its signed upstream to equal the active
    version's exact registered URL. A token naming the invalid version, or with
    only one of the two claims or a mistyped one, is malformed (None).
    """
    if "version_id" not in claims and "virtual_backend" not in claims:
        return "", False
    version_id = claims.get("version_id")
    virtual_backend = claims.get("virtual_backend")
    if (
        not isinstance(version_id, str)
        or version_id == INVALID_VERSION_SENTINEL
        or not isinstance(virtual_backend, bool)
    ):
        return None
    return version_id, virtual_backend


# Registry API paths that resource-bound tokens must never be allowed to
# reach, regardless of what resource they are bound to. A resource-bound
# token that could hit /api/tokens/generate could mint new tokens (including
# user tokens); one that could hit /api/admin/* could change registry state.
# These are exact paths or prefixes; match is substring-agnostic (prefix-only).
#
# EXPORTED — this tuple is imported by ``auth_server.server`` so the two
# enforcement layers share the same deny-list. Adding a new blocked prefix
# here closes the gap on both sides simultaneously.
RESOURCE_TOKEN_BLOCKED_PREFIXES: tuple[str, ...] = (
    "/api/tokens",
    "/api/admin",
    "/api/search",
    "/api/federation",
    "/api/registry",
    "/api/peers",
    "/api/config",
    # /api/auth is blocked at the resource-token layer as defense-in-depth.
    # In practice nginx serves most of /api/auth/* as a public endpoint
    # that does NOT go through /validate (OAuth callbacks, login, logout);
    # only /api/auth/me is protected by auth_request. This entry therefore
    # mostly guards that single endpoint, which is then re-enabled by the
    # RESOURCE_TOKEN_ALLOWED_PATHS allow-list below.
    "/api/auth",
)

# Carve-outs from the blocked list: these paths are under a blocked prefix
# but are harmless (read-only introspection) and useful for tokens to reach.
# /api/auth/me is the "who am I" endpoint a client calls to verify its token
# works — blocking it would break token-introspection flows for
# resource-bound tokens without any security benefit.
#
# EXPORTED alongside the blocked-prefix list so both enforcement layers see
# the same allow-list.
RESOURCE_TOKEN_ALLOWED_PATHS: frozenset[str] = frozenset({"/api/auth/me"})

# Internal combined prefix map used by ``classify_request_url``. Extends
# the exported ``API_PREFIX_TO_TYPE`` (which covers /api/* only) with the
# /virtual/ prefix for virtual servers. The plain ``server`` type has no
# prefix — it's the catch-all branch of the classifier for any non-API,
# non-/virtual/ path.
_PREFIX_TO_TYPE = API_PREFIX_TO_TYPE + (("/virtual/", ResourceType.VIRTUAL_SERVER),)

# Private aliases used by the helper functions in this module. The public
# names (``RESOURCE_TOKEN_BLOCKED_PREFIXES``, ``RESOURCE_TOKEN_ALLOWED_PATHS``)
# are the source of truth; these aliases exist so a future refactor of
# the exported names does not silently break the helpers.
_RESOURCE_TOKEN_BLOCKED_PREFIXES = RESOURCE_TOKEN_BLOCKED_PREFIXES
_RESOURCE_TOKEN_ALLOWED_PATHS = RESOURCE_TOKEN_ALLOWED_PATHS


def normalize_resource_id(raw: str) -> str:
    """Normalize a resource id for consistent comparison.

    Resource ids are passed around both as URL paths ("/my-server") and as
    slugs ("my-server"). We canonicalize to the no-leading-slash,
    no-trailing-slash form so that the mint-time claim and the edge-time
    comparison agree regardless of which form the caller used.

    Virtual server ids keep their "virtual/" prefix (after slash stripping)
    so that virtual_server:virtual/foo and server:foo remain distinct.
    """
    return raw.strip().strip("/")


def classify_request_url(
    path: str,
    root_path: str = "",
) -> tuple[ResourceType, str] | None:
    """Classify a request URL path into (resource_type, resource_id).

    Returns None if the path cannot be classified (e.g., static assets,
    health endpoints, unauthenticated routes). A None return at the guard
    layer means the guard has nothing to compare against and should fall
    back to "allow user tokens, reject resource tokens" — see
    ``check_resource_token_allowed``.

    The registry API mount is ``/api`` (see registry/main.py). Paths may
    arrive with or without a trailing slash.

    Assumes the upstream proxy has normalized slashes — nginx's default
    ``merge_slashes on`` collapses ``//api//foo`` to ``/api/foo`` before
    setting ``X-Original-URL``. If deployed behind a proxy that does NOT
    merge slashes (e.g. Traefik), unmerged paths fall through to the
    catch-all server branch and fail classification, which is still
    fail-closed for resource-bound tokens (returns 403).

    Examples:
        /api/agents/code-reviewer -> (AGENT, "code-reviewer")
        /api/skills/python-linter -> (SKILL, "python-linter")
        /virtual/my-agg/mcp       -> (VIRTUAL_SERVER, "virtual/my-agg")
        /cloudflare-docs/mcp      -> (SERVER, "cloudflare-docs")
        /peer-registry/cf-docs    -> (SERVER, "peer-registry/cf-docs")
        /api/tokens/generate      -> None (blocked for resource tokens)
        /api/health               -> None (public, no classification)
    """
    if not path:
        return None

    # ``strip_trailing=False`` — the branch logic below uses ``split("/")``
    # and discards trailing empty segments naturally. Stripping here would
    # also hide ``/api/`` from matching ``path == "/api"``.
    path = _prepare_path(path, root_path, strip_trailing=False)

    # Handle /api/* paths first.
    if path.startswith("/api/") or path == "/api":
        for prefix, resource_type in _PREFIX_TO_TYPE:
            if prefix.startswith("/api/") and path.startswith(prefix):
                resource_id = path[len(prefix) :]
                # Strip sub-paths like /toggle, /rate, /health — we want the
                # top-level resource slug only. Agents and skills don't have
                # nested MCP transport segments, so splitting on "/" and
                # taking the first component is correct.
                resource_id = resource_id.split("/", 1)[0]
                resource_id = resource_id.rstrip("/")
                if not resource_id:
                    return None
                return (resource_type, resource_id)
        # Any other /api/* path (tokens, admin, search, health, etc.) is
        # not a classifiable resource.
        return None

    # /virtual/* is a virtual server. resource_id includes the "virtual/"
    # prefix so it can never collide with a plain server of the same slug.
    if path.startswith("/virtual/"):
        rest = path[len("/virtual/") :].strip("/")
        if not rest:
            return None
        parts = rest.split("/")
        # Strip trailing transport segment if present.
        if parts and parts[-1] in MCP_TRANSPORT_SEGMENTS:
            parts = parts[:-1]
        if not parts:
            return None
        return (ResourceType.VIRTUAL_SERVER, "virtual/" + "/".join(parts))

    # Otherwise, treat as a regular MCP server. The id is the full path
    # minus any trailing transport segment. This also handles federated
    # paths like /peer-registry-lob-1/cloudflare-docs.
    stripped = path.strip("/")
    if not stripped:
        return None
    parts = stripped.split("/")
    if parts and parts[-1] in MCP_TRANSPORT_SEGMENTS:
        parts = parts[:-1]
    if not parts:
        return None
    return (ResourceType.SERVER, "/".join(parts))


def is_resource_token_introspection_path(path: str, root_path: str = "") -> bool:
    """True if ``path`` is on the resource-token allow-list.

    Distinct from :func:`check_resource_token_allowed`, which answers
    "is this path NOT blocked". This function answers "is this path an
    explicit carve-out that should bypass (resource_type, resource_id)
    matching entirely". Use it at the edge guard to let resource-bound
    tokens reach introspection endpoints like ``/api/auth/me`` that do
    not classify to any resource but are safe for every token.
    """
    if not path:
        return False
    return _prepare_path(path, root_path) in _RESOURCE_TOKEN_ALLOWED_PATHS


def check_resource_token_allowed(path: str, root_path: str = "") -> bool:
    """Return True if a resource-bound token is allowed to reach ``path``.

    Used to block resource-bound tokens from reaching endpoints that could
    escalate their privileges (/api/tokens/generate) or bypass the binding
    (/api/admin/*, /api/search/*). Has no effect on user-token requests.

    ``root_path`` lets callers strip a registry sub-path prefix (e.g.
    "/registry") before matching, so the deny-list works whether or not
    the registry is hosted under a sub-path.
    """
    if not path:
        return False
    path = _prepare_path(path, root_path)
    # Exact-match allow-list takes precedence over prefix-based deny-list
    # so that, e.g., /api/auth/me is reachable even though /api/auth is
    # otherwise blocked.
    if path in _RESOURCE_TOKEN_ALLOWED_PATHS:
        return True
    for blocked in _RESOURCE_TOKEN_BLOCKED_PREFIXES:
        # Match "/api/admin" and "/api/admin/foo" but not "/api/administration".
        if path == blocked or path.startswith(blocked + "/"):
            return False
    return True


async def validate_user_can_bind_resource(
    resource_type: ResourceType | str,
    resource_id: str,
    user_context: dict[str, Any],
) -> bool:
    """Validate that ``user_context`` is permitted to mint a token bound to
    (resource_type, resource_id).

    This is a pre-mint authorization check: we refuse to issue a
    resource-bound token for a resource the user cannot reach today, so the
    bound token can never grant more than the user already had.

    Semantics per type:
        SERVER         -> user_context['accessible_servers'] must contain
                          the server name, or the user is admin / has "all".
        VIRTUAL_SERVER -> must appear in user's list_virtual_server UI
                          permission (or admin / "all").
        AGENT          -> accessible_agents contains the agent path, or
                          user is admin / has "all".
        SKILL          -> skill visibility check (public, owner, group).

    Accepts either a ``ResourceType`` enum member or its string value so
    callers that received the type from an HTTP request can pass it through
    without an explicit conversion.

    Imports are deferred to avoid pulling service modules during auth_server
    startup (this module is also imported by the registry at runtime).
    """
    try:
        rtype = ResourceType(resource_type)
    except ValueError:
        logger.warning(f"Unknown resource type '{resource_type}' during binding check")
        return False

    normalized_id = normalize_resource_id(resource_id)

    if user_context.get("is_admin"):
        return True

    if rtype is ResourceType.SERVER:
        accessible = user_context.get("accessible_servers") or []
        if _WILDCARD_ACCESS & set(accessible):
            return True
        return normalized_id in accessible

    if rtype is ResourceType.VIRTUAL_SERVER:
        ui_permissions = user_context.get("ui_permissions") or {}
        list_virtual_perms = ui_permissions.get("list_virtual_server") or []
        if _WILDCARD_ACCESS & set(list_virtual_perms):
            return True
        # UI permission values are stored as "/virtual/foo"; normalize the
        # right-hand side the same way as the claim.
        normalized_perms = {normalize_resource_id(p) for p in list_virtual_perms}
        return normalized_id in normalized_perms

    if rtype is ResourceType.AGENT:
        accessible_agents = user_context.get("accessible_agents") or []
        if _WILDCARD_ACCESS & set(accessible_agents):
            return True
        normalized_agents = {normalize_resource_id(a) for a in accessible_agents}
        if normalized_id not in normalized_agents:
            return False
        # Defer to the visibility check so the binding can never exceed what
        # the user can actually see. Bound the lookup with a timeout so a
        # frozen DB cannot hang the mint request.
        try:
            from ..services.agent_service import agent_service

            agent_card = await asyncio.wait_for(
                agent_service.get_agent_info(f"/{normalized_id}"),
                timeout=_BIND_CHECK_LOOKUP_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "Timed out looking up agent %s during resource-binding check; denying bind request",
                normalized_id,
            )
            return False
        except Exception:
            logger.exception(
                "Failed to look up agent %s during resource-binding check", normalized_id
            )
            return False
        if not agent_card:
            return False
        if agent_card.visibility == "public":
            return True
        if agent_card.visibility == "private":
            return agent_card.registered_by == user_context.get("username")
        if agent_card.visibility == "group-restricted":
            user_groups = set(user_context.get("groups") or [])
            return bool(set(agent_card.allowed_groups) & user_groups)
        return False

    if rtype is ResourceType.SKILL:
        # Bound the lookup with a timeout so a frozen DB cannot hang the
        # mint request.
        try:
            from ..services.skill_service import get_skill_service

            skill = await asyncio.wait_for(
                get_skill_service().get_skill(f"/{normalized_id}"),
                timeout=_BIND_CHECK_LOOKUP_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "Timed out looking up skill %s during resource-binding check; denying bind request",
                normalized_id,
            )
            return False
        except Exception:
            logger.exception(
                "Failed to look up skill %s during resource-binding check", normalized_id
            )
            return False
        if not skill:
            return False

        from ..schemas.skill_models import VisibilityEnum

        visibility = getattr(skill, "visibility", None)
        if visibility == VisibilityEnum.PUBLIC:
            return True
        if visibility == VisibilityEnum.PRIVATE:
            return skill.owner == user_context.get("username")
        if visibility == VisibilityEnum.GROUP:
            user_groups = set(user_context.get("groups") or [])
            return bool(user_groups & set(getattr(skill, "allowed_groups", []) or []))
        return False

    # All ResourceType members handled above. Unreachable under normal
    # operation but preserved as a safety net in case new enum members are
    # added without updating this function.
    logger.error(f"Unhandled ResourceType {rtype!r} in validate_user_can_bind_resource")
    return False
