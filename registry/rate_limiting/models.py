"""Pydantic models for rate-limit definitions and decisions.

A definition is described by two orthogonal dimensions plus a window:

- ``axis``: which side of the call the limit applies to (``caller`` or ``target``).
- ``entity_type``: what kind of thing on that axis. Allowlisted per axis so the
  model is not tied to any single target kind (MCP server, A2A agent, and later
  tool/skill all plug in without a schema change).
- ``window_seconds``: the fixed window. Ranges from seconds to a full day, so a
  per-day volume cap is the same mechanism as a per-second burst cap.

See ``.scratchpad/issue-295/lld.md`` (Data Models) for the full rationale.
"""

import logging

from pydantic import BaseModel, Field, model_validator

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Allowlisted entity types per axis. Extend these as new entity kinds gain a
# gateway chokepoint. Unknown values are rejected in the validator (fail closed).
#   Caller kind (name from the validated token's groups claim):
#     "group" -> a group the caller belongs to. A specific user or agent is made
#                subject to a group limit by being a MEMBER of that group (via the
#                IAM membership APIs + token group enrichment), not by naming the
#                user/client directly here.
#   Target kinds:
#     Coarse (name = path-derived):  "mcp_server", "a2a_agent"
#     Fine   (name = "<parent>:<leaf>", from the JSON-RPC payload; later phase,
#             NOT enforced in v1): "mcp_tool", "a2a_skill"
CALLER_ENTITY_TYPES: frozenset[str] = frozenset({"group"})
TARGET_ENTITY_TYPES: frozenset[str] = frozenset(
    {"mcp_server", "a2a_agent", "mcp_tool", "a2a_skill"}
)

# A "server group" is a convenience label on the target axis: a single definition
# names a SET of servers (via the on-definition ``members`` list), and EACH member
# gets its OWN independent per-window bucket (per-member uniform N, NOT a shared
# pool). It is never a request-classified target (a live request always classifies
# to mcp_server / a2a_agent), so it is deliberately NOT in TARGET_ENTITY_TYPES; it
# only exists as a definition whose limit is surfaced for each member server at
# lookup time. See docs/design/rate-limiting.md.
SERVER_GROUP_ENTITY_TYPE: str = "server_group"

# Fine-grained target kinds whose enforcement classifier is not wired in v1.
# The admin API accepts these only when explicitly acknowledged, so an operator
# never gets a silently-inert limit. See lld.md API Design.
UNENFORCED_ENTITY_TYPES: frozenset[str] = frozenset({"mcp_tool", "a2a_skill"})

# Axis abbreviations used in counter keys (kept short and stable).
#   caller        -> per-caller quota across all targets
#   target        -> per-target quota across all callers
#   caller_target -> per-caller-per-target composite quota (each caller gets its
#                    own independent quota against each specific target)
#   quarantine    -> sentinel group with NO rate; membership drops all traffic
AXIS_ABBREVIATION: dict[str, str] = {
    "caller": "clr",
    "target": "tgt",
    "caller_target": "ctg",
    "quarantine": "qtn",
}

# Axes whose definition is a "group" carrying per-caller-type limits (a caller is
# subject to it by membership). Both caller and caller_target share this shape.
GROUP_AXES: frozenset[str] = frozenset({"caller", "caller_target"})

# Reserved rate-limit group names. Membership in one of these DROPS all
# data-plane traffic from the caller / to the target (a kill switch, not a rate).
# Seeded empty at startup; operators cannot create a normal definition with these
# names, and the groups cannot be deleted (only disabled = global off-switch).
QUARANTINE_CALLER_GROUP: str = "quarantine-callers"
QUARANTINE_TARGET_GROUP: str = "quarantine-targets"
RESERVED_GROUP_NAMES: frozenset[str] = frozenset({QUARANTINE_CALLER_GROUP, QUARANTINE_TARGET_GROUP})

# Scopes for the quarantine sentinel axis.
QUARANTINE_SCOPES: frozenset[str] = frozenset({"caller", "target"})

# Bounds for window_seconds: 1 second up to one full day.
MIN_WINDOW_SECONDS: int = 1
MAX_WINDOW_SECONDS: int = 86400


# Caller types for floor selection and per-type limit resolution.
CALLER_TYPE_USER: str = "user"
CALLER_TYPE_AGENT: str = "agent"


def normalize_server_name(name: str) -> str:
    """Canonicalize a server target name to the form the request classifier keys on.

    The auth-server classifies an MCP request's target as ``path.strip("/").split("/")[0]``
    -- the BARE first path segment, no slashes (e.g. a request to ``/aws-kb/mcp`` keys
    on ``aws-kb``). A quarantine or server_group member stored as ``/aws-kb`` (or
    ``/aws-kb/``) would therefore never match at enforcement and the control would be
    silently inert (fail-open). Normalizing on the way in makes ``aws-kb``, ``/aws-kb``,
    and ``/aws-kb/`` all resolve to the same enforceable subject. An empty string is
    returned as-is so callers can reject it.
    """
    if not isinstance(name, str):
        return name
    stripped = name.strip().strip("/")
    if not stripped:
        return ""
    return stripped.split("/")[0]


def normalize_agent_path(path: str) -> str:
    """Canonicalize an agent target path to the form the request classifier keys on.

    The auth-server classifies an A2A request's target as the agent path WITH a
    leading slash and (possibly) multiple segments (e.g. ``/flight-booking-agent``),
    the opposite convention from a server name. Mirror that: ensure a single leading
    slash and strip a trailing slash, so ``flight-booking-agent`` and
    ``/flight-booking-agent/`` both resolve to ``/flight-booking-agent``. An empty
    string is returned as-is so callers can reject it.
    """
    if not isinstance(path, str):
        return path
    stripped = path.strip()
    if not stripped:
        return ""
    if not stripped.startswith("/"):
        stripped = "/" + stripped
    if len(stripped) > 1:
        stripped = stripped.rstrip("/")
    return stripped


def normalize_target_subject(
    subject_type: str,
    subject: str,
) -> str:
    """Normalize a target subject to the classifier's form, by entity type.

    ``server`` -> bare first segment; ``agent`` -> leading-slash path. Any other
    subject type (caller: user/client) is returned unchanged -- a caller subject is
    an identity, not a path, and must not be slash-mangled.
    """
    if subject_type == "server":
        return normalize_server_name(subject)
    if subject_type == "agent":
        return normalize_agent_path(subject)
    return subject


class RateLimitDefinition(BaseModel):
    """A rate-limit definition on one axis, for one entity type, at one window.

    - ``axis="caller"``: aggregate limit on a group the caller belongs to. A group
      carries a SEPARATE limit for human users (``user_max_requests``) and for
      agents/M2M clients (``agent_max_requests``); at least one is required, both
      are allowed. A specific user/agent is subject to the group by a membership.
    - ``axis="target"``: aggregate limit on one target entity across all callers
      via ``max_requests`` (v1 enforced: an MCP server or an A2A agent; tool/skill
      modeled but not wired).
    """

    axis: str = Field(
        ...,
        description="Which side of the call: 'caller' or 'target'",
    )
    entity_type: str = Field(
        ...,
        description="caller: 'group'; target: 'mcp_server' | 'a2a_agent' | 'mcp_tool' | 'a2a_skill'",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Group name, server path, or agent name this applies to",
    )
    # Target axis uses this single aggregate limit.
    max_requests: int | None = Field(
        default=None,
        ge=1,
        description="Target-axis: max requests per window across all callers",
    )
    # server_group target entity only: the set of server paths the definition
    # covers. Each member gets its OWN independent max_requests/window bucket
    # (per-member uniform, not a shared pool). None/absent for every other
    # definition -- this is the backward-compatibility anchor for pre-existing docs.
    members: list[str] | None = Field(
        default=None,
        description="server_group only: server paths, each getting its own N/window bucket",
    )
    # Caller (group) axis uses these per-caller-type limits. At least one required.
    user_max_requests: int | None = Field(
        default=None,
        ge=1,
        description="Caller/group: max requests per window for a human user in the group",
    )
    agent_max_requests: int | None = Field(
        default=None,
        ge=1,
        description="Caller/group: max requests per window for an agent/client in the group",
    )
    window_seconds: int = Field(
        default=60,
        ge=MIN_WINDOW_SECONDS,
        le=MAX_WINDOW_SECONDS,
        description="Window length in seconds (up to one day)",
    )
    # Quarantine sentinel axis only: which side the kill-switch group applies to.
    scope: str | None = Field(
        default=None,
        description="quarantine axis only: 'caller' | 'target'",
    )
    fail_closed: bool = Field(
        default=False,
        description="If true, deny on backend error (security-critical only)",
    )
    enabled: bool = Field(
        default=True,
        description="Toggle without deleting",
    )

    @model_validator(mode="after")
    def _check_axis_and_entity_type(self) -> "RateLimitDefinition":
        """Validate axis/entity_type allowlists and the per-axis limit fields (fail closed)."""
        if self.axis not in AXIS_ABBREVIATION:
            raise ValueError(f"axis must be one of {sorted(AXIS_ABBREVIATION)}, got '{self.axis}'")

        if self.axis == "quarantine":
            return self._check_quarantine()

        # Non-quarantine axes may not use a reserved kill-switch group name (an
        # operator must not be able to shadow a quarantine group with a rate group).
        if self.name in RESERVED_GROUP_NAMES:
            raise ValueError(
                f"'{self.name}' is a reserved quarantine group name and cannot be used "
                f"for a '{self.axis}' definition"
            )

        allowed = (
            CALLER_ENTITY_TYPES
            if self.axis in GROUP_AXES
            else (TARGET_ENTITY_TYPES | {SERVER_GROUP_ENTITY_TYPE})
        )
        if self.entity_type not in allowed:
            raise ValueError(
                f"entity_type '{self.entity_type}' invalid for axis '{self.axis}' "
                f"(allowed: {sorted(allowed)})"
            )

        if self.axis in GROUP_AXES:
            # A group (caller or caller_target) must set at least one per-caller-type limit.
            if self.user_max_requests is None and self.agent_max_requests is None:
                raise ValueError(
                    f"a {self.axis} (group) definition must set user_max_requests and/or "
                    "agent_max_requests"
                )
        else:
            # Target axis uses the single max_requests.
            if self.max_requests is None:
                raise ValueError("a target definition must set max_requests")
            self._check_members()
        return self

    def _check_members(self) -> None:
        """Validate + normalize the ``members`` list (server_group only; absent otherwise).

        Members are normalized to the bare server name the request classifier keys on
        (``/aws-kb`` -> ``aws-kb``), so a member stored with a leading/trailing slash
        can't silently miss at enforcement. Normalization happens before the
        duplicate check so ``aws-kb`` and ``/aws-kb`` are caught as the same member.
        """
        if self.entity_type == SERVER_GROUP_ENTITY_TYPE:
            # A server_group must name at least one member; each gets its own bucket.
            if not self.members:
                raise ValueError("a server_group definition must set a non-empty members list")
            if any(not isinstance(m, str) or not m.strip() for m in self.members):
                raise ValueError("server_group members must be non-empty server-path strings")
            normalized = [normalize_server_name(m) for m in self.members]
            if any(not m for m in normalized):
                raise ValueError("server_group members must be non-empty server-path strings")
            if len(set(normalized)) != len(normalized):
                raise ValueError("server_group members must not contain duplicates")
            self.members = normalized
        elif self.members is not None:
            # members is meaningless for a single-target definition; reject it so a
            # caller can't set a field that would be silently ignored.
            raise ValueError("members is only valid for a server_group definition")

    def _check_quarantine(self) -> "RateLimitDefinition":
        """Validate a quarantine sentinel: reserved name, group entity, scope, no rate."""
        if self.entity_type != "group":
            raise ValueError("a quarantine definition must use entity_type 'group'")
        if self.name not in RESERVED_GROUP_NAMES:
            raise ValueError(
                f"a quarantine definition name must be one of {sorted(RESERVED_GROUP_NAMES)}"
            )
        if self.scope not in QUARANTINE_SCOPES:
            raise ValueError(f"quarantine scope must be one of {sorted(QUARANTINE_SCOPES)}")
        if (
            self.max_requests is not None
            or self.user_max_requests is not None
            or self.agent_max_requests is not None
        ):
            raise ValueError("a quarantine definition must not set any rate limit")
        return self

    def limit_for_caller_type(
        self,
        caller_type: str,
    ) -> int | None:
        """Return the per-window limit for a caller type ('user'/'agent'), or None if unset."""
        if caller_type == CALLER_TYPE_AGENT:
            return self.agent_max_requests
        return self.user_max_requests

    def build_id(self) -> str:
        """Build the definition document ``_id``: '<axis>:<entity_type>:<name>:<window_seconds>'."""
        return f"{self.axis}:{self.entity_type}:{self.name}:{self.window_seconds}"


# Subject kinds for a rate-limit membership.
#   Callers: a human user (by username) or an agent / M2M client (by client_id).
#   Targets: an mcp_server (subject "server:<name>") or a2a_agent ("agent:<path>")
#            -- used ONLY to place a target into the quarantine-targets group.
# Rejected if outside this set (fail closed).
CALLER_SUBJECT_TYPES: frozenset[str] = frozenset({"user", "client"})
TARGET_SUBJECT_TYPES: frozenset[str] = frozenset({"server", "agent"})
MEMBERSHIP_SUBJECT_TYPES: frozenset[str] = CALLER_SUBJECT_TYPES | TARGET_SUBJECT_TYPES


class RateLimitMembership(BaseModel):
    """Maps a caller (a user or an agent/client) to rate-limit group name(s).

    This is deliberately SEPARATE from the token's authz groups: no IdP emits
    rate-limit groups, and mixing them into the authz groups list could change a
    caller's scopes. The limiter reads *only* this collection to resolve which
    rate-limit ``group`` definitions apply to a caller; it never affects
    authorization.
    """

    subject_type: str = Field(
        ...,
        description="'user' (name = username) or 'client' (name = client_id / azp)",
    )
    subject: str = Field(
        ...,
        min_length=1,
        description="The username or client_id this membership applies to",
    )
    groups: list[str] = Field(
        default_factory=list,
        description="Rate-limit group names this subject belongs to",
    )

    @model_validator(mode="after")
    def _check_subject_type(self) -> "RateLimitMembership":
        """Reject a subject_type outside the allowlist and a mis-scoped quarantine (fail closed)."""
        if self.subject_type not in MEMBERSHIP_SUBJECT_TYPES:
            raise ValueError(
                f"subject_type must be one of {sorted(MEMBERSHIP_SUBJECT_TYPES)}, "
                f"got '{self.subject_type}'"
            )
        # A target subject (server/agent) exists ONLY to be quarantined: it may
        # belong to nothing other than the target quarantine group.
        if self.subject_type in TARGET_SUBJECT_TYPES:
            if self.groups and self.groups != [QUARANTINE_TARGET_GROUP]:
                raise ValueError(
                    "a server/agent subject may only join the quarantine-targets group"
                )
        # A caller may not join the target quarantine group, and vice versa
        # (fail closed: a mis-scoped quarantine must not silently do nothing).
        if QUARANTINE_TARGET_GROUP in self.groups and self.subject_type in CALLER_SUBJECT_TYPES:
            raise ValueError("quarantine-targets is for server/agent subjects only")
        if QUARANTINE_CALLER_GROUP in self.groups and self.subject_type in TARGET_SUBJECT_TYPES:
            raise ValueError("quarantine-callers is for user/client subjects only")
        return self

    def build_id(self) -> str:
        """Build the membership document ``_id``: '<subject_type>:<subject>'."""
        return f"{self.subject_type}:{self.subject}"


class RateLimitDecision(BaseModel):
    """The result of enforcing a single gate (or the aggregate allow).

    ``reset_epoch`` is used to pick the friendliest ``Retry-After`` when several
    gates deny at once: the caller should back off until the gate with the
    longest reset clears, not a short burst window.
    """

    allowed: bool
    axis: str | None = None
    entity_type: str | None = None
    limit: int | None = None
    remaining: int = 0
    reset_epoch: int = 0
    retry_after: int = 0
    # A hard quarantine block (not a throttle). Drives the response branch at the
    # enforcement hook: quarantine -> plain 403 (no X-RateLimit-Throttled marker,
    # so nginx does NOT rewrite it to 429); a throttle -> 403 + marker -> 429.
    quarantined: bool = False

    @classmethod
    def allow(
        cls,
        remaining: int = 0,
    ) -> "RateLimitDecision":
        """Build an allow decision. ``remaining`` is best-effort headroom for headers."""
        return cls(allowed=True, remaining=max(0, remaining))

    @classmethod
    def quarantine_deny(
        cls,
        axis: str,
        entity_type: str,
    ) -> "RateLimitDecision":
        """Build a hard-block decision (no rate metadata -> a plain 403, never a 429)."""
        return cls(allowed=False, axis=axis, entity_type=entity_type, quarantined=True)

    def headers(self) -> dict[str, str]:
        """Build the ``X-RateLimit-*`` / ``Retry-After`` headers for a throttle response.

        ``X-RateLimit-Throttled`` is a marker the nginx ``auth_request`` layer uses
        to tell a throttle-403 apart from a genuine authorization 403: the throttle
        response leaves ``/validate`` as a 403 (the only non-2xx status other than
        401 that ``auth_request`` forwards), and nginx rewrites it back into a real
        429 with these headers for the client. See the ``@forbidden_error`` named
        location in the nginx templates.
        """
        return {
            "X-RateLimit-Throttled": "1",
            "X-RateLimit-Limit": str(self.limit if self.limit is not None else 0),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(self.reset_epoch),
            "Retry-After": str(self.retry_after),
            "Connection": "close",
        }
