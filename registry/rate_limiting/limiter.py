"""The rate limiter: orchestrates caller (Limit A) and target (Limit B) gates.

A "gate" is one (axis, entity_type, subject, window) tuple with its definition.
Every applicable gate is enforced; ALL must pass. Different ``window_seconds``
are independent gates (a burst cap and a daily cap both apply); most-restrictive
resolution applies only among a caller's groups sharing the same window.

Gates are enforced **sequentially, tightest-window-first**, stopping at the first
denial. This guarantees that a request rejected by a tight burst gate never
increments a wider volume/day gate: the burst gate is evaluated first, denies,
and the later (longer-window) gates are never touched. A concurrent design would
consume every gate's quota on every attempt, letting a rejected burst exhaust the
daily cap -- so we deliberately trade one round trip per configured gate on the
allowed path (only paid when limits are configured; bounded by the backend
timeout) for correct cross-window behavior.
"""

import asyncio
import logging
import time

from ..observability.meters import (
    rate_limit_checks_total,
    rate_limit_errors_total,
    rate_limit_quarantine_denied_total,
    rate_limit_throttled_total,
)
from .backend import RateLimiterBackend
from .definitions_repository import DefinitionsRepository
from .memberships_repository import MembershipsRepository
from .models import (
    AXIS_ABBREVIATION,
    CALLER_TYPE_AGENT,
    CALLER_TYPE_USER,
    QUARANTINE_CALLER_GROUP,
    QUARANTINE_TARGET_GROUP,
    RESERVED_GROUP_NAMES,
    RateLimitDecision,
    RateLimitDefinition,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Default hard timeout per backend counter op. On timeout the op is treated as a
# backend error and follows the fail-open/closed policy, so a slow counter store
# fails FAST and never hangs the hot /validate path.
DEFAULT_BACKEND_TIMEOUT_SECONDS: float = 0.25


def _by_window(
    definitions: list[RateLimitDefinition],
) -> dict[int, list[RateLimitDefinition]]:
    """Group definitions by ``window_seconds`` so each window is resolved separately."""
    grouped: dict[int, list[RateLimitDefinition]] = {}
    for definition in definitions:
        grouped.setdefault(definition.window_seconds, []).append(definition)
    return grouped


def _window_reset_epoch(
    window_seconds: int,
    now_epoch: float,
) -> int:
    """Return the epoch second when the current fixed window for ``window_seconds`` resets."""
    window_index = int(now_epoch) // window_seconds
    return (window_index + 1) * window_seconds


class _Gate:
    """One fully-resolved gate: axis, entity type, subject, window, limit, fail-closed.

    Carries an explicit resolved ``max_requests`` (for caller gates this is the
    user- or agent-specific number picked by caller type), so the limiter never
    reaches back into the definition's internal per-type fields.
    """

    def __init__(
        self,
        axis_abbr: str,
        entity_type: str,
        subject: str,
        window_seconds: int,
        max_requests: int,
        fail_closed: bool,
    ) -> None:
        self.axis_abbr = axis_abbr
        self.entity_type = entity_type
        self.subject = subject
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.fail_closed = fail_closed

    def counter_key(self) -> str:
        """Build the counter key (window index is appended by the backend)."""
        return f"{self.axis_abbr}:{self.entity_type}:{self.subject}:{self.window_seconds}"


class RateLimiter:
    """Enforces caller and target aggregate limits on every checked call."""

    def __init__(
        self,
        backend: RateLimiterBackend,
        definitions: DefinitionsRepository,
        memberships: "MembershipsRepository",
        fail_open: bool = True,
        backend_timeout_seconds: float = DEFAULT_BACKEND_TIMEOUT_SECONDS,
        quarantine_fail_closed: bool = False,
    ) -> None:
        self._backend = backend
        self._definitions = definitions
        self._memberships = memberships
        self._fail_open = fail_open
        self._backend_timeout_seconds = backend_timeout_seconds
        # When true, a backend error reading quarantine membership DENIES (fails
        # closed) instead of the default fail-open. A stricter block at the cost of
        # denying data-plane traffic during a memberships-store outage.
        self._quarantine_fail_closed = quarantine_fail_closed

    async def _build_caller_gates(
        self,
        identity: str,
        groups: list[str],
        caller_type: str,
    ) -> list[_Gate]:
        """Build one caller gate per window, most-restrictive across the caller's groups.

        ``groups`` are the caller's RATE-LIMIT groups, resolved from the memberships
        collection (keyed by username/client_id) -- NOT the token's authz groups.
        For each group definition, the limit used is the one for ``caller_type``
        ('user' or 'agent'); a group that does not set that caller type's limit
        simply does not gate this caller. Enforced per caller: the counter subject
        is the caller's identity.
        """
        if not groups:
            return []
        group_defs = await self._definitions.list_caller_limits("group", groups)
        # (window -> list of resolved max_requests + the def they came from)
        by_window: dict[int, list[tuple[int, RateLimitDefinition]]] = {}
        for definition in group_defs:
            limit = definition.limit_for_caller_type(caller_type)
            if limit is None:
                continue
            by_window.setdefault(definition.window_seconds, []).append((limit, definition))

        gates: list[_Gate] = []
        for window, candidates in by_window.items():
            # Most-restrictive within the window for this caller type.
            limit, definition = min(candidates, key=lambda pair: pair[0])
            gates.append(
                _Gate(
                    AXIS_ABBREVIATION["caller"],
                    "group",
                    identity,
                    window,
                    limit,
                    definition.fail_closed,
                )
            )
        return gates

    async def _build_target_gates(
        self,
        target_entity_type: str | None,
        target_name: str | None,
    ) -> list[_Gate]:
        """Build one target gate per window for the classified target entity (if any)."""
        if not target_entity_type or not target_name:
            return []
        target_defs = await self._definitions.list_target_limits(target_entity_type, target_name)
        gates: list[_Gate] = []
        for definition in target_defs:
            if definition.max_requests is None:
                continue
            gates.append(
                _Gate(
                    AXIS_ABBREVIATION["target"],
                    definition.entity_type,
                    target_name,
                    definition.window_seconds,
                    definition.max_requests,
                    definition.fail_closed,
                )
            )
        return gates

    async def _build_caller_target_gates(
        self,
        identity: str,
        groups: list[str],
        caller_type: str,
        target_entity_type: str | None,
        target_name: str | None,
    ) -> list[_Gate]:
        """Build per-caller-per-target gates: one per window, keyed on (caller, target).

        Mirrors ``_build_caller_gates`` but (a) returns ``[]`` when no target is
        classified, and (b) sets the counter subject to the COMPOSITE of the caller
        identity and the target, so each caller gets an independent quota per target.
        The ``|`` delimiter separates the two halves unambiguously (the caller
        identity is defensively stripped of ``|``); the target half keeps the
        existing ``<entity_type>:<name>`` shape.
        """
        if not groups or not target_entity_type or not target_name:
            return []
        group_defs = await self._definitions.list_caller_target_limits("group", groups)
        by_window: dict[int, list[tuple[int, RateLimitDefinition]]] = {}
        for definition in group_defs:
            limit = definition.limit_for_caller_type(caller_type)
            if limit is None:
                continue
            by_window.setdefault(definition.window_seconds, []).append((limit, definition))

        safe_identity = identity.replace("|", "_")
        subject = f"{safe_identity}|{target_entity_type}:{target_name}"
        gates: list[_Gate] = []
        for window, candidates in by_window.items():
            # Most-restrictive within the window for this caller type.
            limit, definition = min(candidates, key=lambda pair: pair[0])
            gates.append(
                _Gate(
                    AXIS_ABBREVIATION["caller_target"],
                    "group",
                    subject,
                    window,
                    limit,
                    definition.fail_closed,
                )
            )
        return gates

    async def _is_target_quarantined(
        self,
        target_entity_type: str | None,
        target_name: str | None,
    ) -> bool:
        """Target quarantine check, honoring the global off-switch + fail policy.

        A read error follows the quarantine fail policy: fail-open (allow => not
        quarantined) by default, or deny (=> quarantined) when
        ``quarantine_fail_closed``. The reserved group's own ``enabled`` flag is the
        operator's global off-toggle.
        """
        if not target_entity_type or not target_name:
            return False
        try:
            if not await self._definitions.is_quarantine_group_enabled(QUARANTINE_TARGET_GROUP):
                return False
            return await asyncio.wait_for(
                self._memberships.is_target_quarantined(target_entity_type, target_name),
                timeout=self._backend_timeout_seconds,
            )
        except Exception as exc:
            logger.warning(f"quarantine target read error ({target_name}): {exc}")
            rate_limit_errors_total.add(1, {"axis": "qtn"})
            return self._quarantine_fail_closed

    async def _is_caller_quarantined(
        self,
        groups: list[str],
    ) -> bool:
        """Caller quarantine check on the already-fetched group list (honors global off-switch).

        No new query: ``groups`` is the list ``get_groups_for`` already returned on
        the hot path. Only the global-enabled lookup may touch the (cached) defs.
        """
        if QUARANTINE_CALLER_GROUP not in groups:
            return False
        try:
            return await self._definitions.is_quarantine_group_enabled(QUARANTINE_CALLER_GROUP)
        except Exception as exc:
            logger.warning(f"quarantine caller enabled-read error: {exc}")
            rate_limit_errors_total.add(1, {"axis": "qtn"})
            # Membership already says quarantined; fail policy decides on a defs error.
            return not self._fail_open or self._quarantine_fail_closed

    def _emit_quarantine(
        self,
        scope: str,
        entity_type: str,
        caller_username: str | None,
        caller_client_id: str | None,
    ) -> None:
        """Count + log a quarantine deny (attributable, bounded metric labels)."""
        rate_limit_quarantine_denied_total.add(1, {"scope": scope, "entity_type": entity_type})
        logger.warning(
            f"rate-limit quarantine deny: scope={scope} entity_type={entity_type} "
            f"caller_username={caller_username or ''} "
            f"caller_client_id={caller_client_id or ''}"
        )

    async def check(
        self,
        *,
        username: str | None,
        client_id: str | None,
        is_admin: bool = False,
        target_entity_type: str | None = None,
        target_name: str | None = None,
    ) -> RateLimitDecision:
        """Enforce every applicable gate; return the first denial or an aggregate allow.

        The caller's RATE-LIMIT groups are resolved internally from the memberships
        collection keyed by ``username`` / ``client_id`` -- the token's authz groups
        claim is intentionally NOT consulted (no IdP emits rate-limit groups, and
        mixing them into authz groups could change scopes). The per-group limit used
        for the caller is the user- or agent-specific number based on caller type
        (agent when a ``client_id`` is present, else user).

        **Admin callers bypass caller gates** (an operator must not be able to lock
        themselves out); target gates still apply so a weak backend stays protected.
        Gates are evaluated **sequentially, tightest-window-first**, short-circuiting
        on the first denial (deny-does-not-consume across windows).

        NOTE: this is only called for DATA-PLANE requests (an MCP/A2A target); the
        caller (``server.py``) skips enforcement for control-plane ``/api/*`` calls,
        so caller limits never throttle the dashboard/login.

        ``username``, ``client_id``, ``is_admin`` MUST come from the validated token.
        """
        # Caller type drives which per-group limit and floor applies.
        caller_type = CALLER_TYPE_AGENT if client_id else CALLER_TYPE_USER
        # Per-caller counter subject: prefer client_id (agents), else username.
        identity = client_id or username or ""

        # --- Quarantine short-circuit (before any counter work) ---
        # Target quarantine applies to EVERYONE (even admins): a target taken out of
        # rotation must be unreachable. It is checked first, before the admin bypass.
        if await self._is_target_quarantined(target_entity_type, target_name):
            self._emit_quarantine("target", target_entity_type or "", username, client_id)
            return RateLimitDecision.quarantine_deny("tgt", target_entity_type or "")

        groups: list[str] = []
        if not is_admin:
            groups = await self._memberships.get_groups_for(username, client_id)
            # Caller quarantine, like caller gates, is bypassed for admins (no self-lockout).
            if await self._is_caller_quarantined(groups):
                self._emit_quarantine("caller", "group", username, client_id)
                return RateLimitDecision.quarantine_deny("clr", "group")

        # A reserved quarantine group name is a sentinel, not a rate definition, so it
        # never builds a gate. Drop reserved names before gate building for clarity.
        rate_groups = [g for g in groups if g not in RESERVED_GROUP_NAMES]
        caller_gates: list[_Gate] = []
        caller_target_gates: list[_Gate] = []
        if not is_admin:
            caller_gates = await self._build_caller_gates(identity, rate_groups, caller_type)
            caller_target_gates = await self._build_caller_target_gates(
                identity, rate_groups, caller_type, target_entity_type, target_name
            )
        target_gates = await self._build_target_gates(target_entity_type, target_name)
        gates = caller_gates + caller_target_gates + target_gates
        if not gates:
            return RateLimitDecision.allow()

        # Tightest window first: a short burst cap is evaluated (and can deny) before a wider
        # daily cap, so the daily counter is never touched by a request the burst cap rejects.
        gates.sort(key=lambda gate: gate.window_seconds)

        remaining_values: list[int] = []
        for gate in gates:
            decision = await self._enforce(gate, caller_type, username, client_id)
            if not decision.allowed:
                return decision
            remaining_values.append(decision.remaining)
        # All gates passed. Report the tightest remaining headroom for the response headers.
        return RateLimitDecision.allow(remaining=min(remaining_values))

    async def _enforce(
        self,
        gate: _Gate,
        caller_type: str,
        caller_username: str | None,
        caller_client_id: str | None,
    ) -> RateLimitDecision:
        """Enforce a single gate: conditional increment, emit metrics, build the decision.

        ``caller_type`` / ``caller_username`` / ``caller_client_id`` come from the
        validated token and are logged on a throttle so operators can answer
        "which user or client got throttled" from the app logs (they are NOT
        added as metric labels, which would be unbounded cardinality).
        """
        labels: dict[str, str | int] = {
            "axis": gate.axis_abbr,
            "entity_type": gate.entity_type,
            "window_seconds": gate.window_seconds,
        }
        try:
            # Hard timeout so a slow counter store fails fast into the fail-open/closed
            # policy below, never hanging the /validate subrequest (a TimeoutError is an Exception).
            result = await asyncio.wait_for(
                self._backend.incr_if_allowed(
                    gate.counter_key(),
                    gate.window_seconds,
                    gate.max_requests,
                ),
                timeout=self._backend_timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                f"rate-limit backend error ({gate.counter_key()}): {exc} "
                f"caller_type={caller_type} "
                f"caller_username={caller_username or ''} "
                f"caller_client_id={caller_client_id or ''}"
            )
            rate_limit_errors_total.add(1, {"axis": gate.axis_abbr})
            if gate.fail_closed or not self._fail_open:
                return self._deny(gate)
            return RateLimitDecision.allow()  # fail-open

        outcome = "allow" if result.allowed else "deny"
        rate_limit_checks_total.add(1, {**labels, "outcome": outcome})
        if not result.allowed:
            rate_limit_throttled_total.add(1, labels)
            # Structured key=value fields so the throttle is attributable to a
            # specific user / client_id in the app logs. caller_type tells whether
            # this was a human (user) or an agent/M2M (client). caller_username /
            # caller_client_id are the validated-token identity (one is empty).
            # Metric labels stay low-cardinality (no identity) -- see labels above.
            logger.warning(
                f"rate-limit throttled: axis={gate.axis_abbr} "
                f"entity_type={gate.entity_type} name={gate.subject} "
                f"limit={gate.max_requests}/{gate.window_seconds}s "
                f"caller_type={caller_type} "
                f"caller_username={caller_username or ''} "
                f"caller_client_id={caller_client_id or ''}"
            )
            return self._deny(gate)
        return RateLimitDecision.allow(remaining=gate.max_requests - result.current)

    def _deny(
        self,
        gate: _Gate,
    ) -> RateLimitDecision:
        """Build a deny decision with reset/retry-after computed from the window boundary."""
        now_epoch = time.time()
        reset_epoch = _window_reset_epoch(gate.window_seconds, now_epoch)
        retry_after = max(1, reset_epoch - int(now_epoch))
        return RateLimitDecision(
            allowed=False,
            axis=gate.axis_abbr,
            entity_type=gate.entity_type,
            limit=gate.max_requests,
            remaining=0,
            reset_epoch=reset_epoch,
            retry_after=retry_after,
        )
