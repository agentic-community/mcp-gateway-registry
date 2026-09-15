"""Per-auth-path request counts, accumulated in memory and flushed to the
shared stats document.

The auth-server already counts every ``/validate`` call by auth path on its OTel
instruments, but that meter is per-process: on ECS two auth tasks sit behind an
ALB, and the registry -- which builds the telemetry heartbeat -- shares memory
with neither. So the counts travel through the one channel both deployables
already share: an ``$inc`` into the ``mcp_stats`` document, which sums across
tasks by construction. ``increment_search_counter`` uses the same route.

Nothing here may affect a request. ``record`` is one dict increment on the hot
path, and the flush runs on its own task and swallows every failure.
"""

import asyncio
import logging
from collections import defaultdict

# Dual-path import: in-container the auth_server package is flattened into /app,
# under pytest it is rooted at the repo. Mirrors metrics_middleware.py.
try:
    from observability.meters import auth_path_flush_total
except ImportError:
    from auth_server.observability.meters import auth_path_flush_total

logger = logging.getLogger(__name__)

# Not a configuration parameter. The interval trades flush volume against how
# much a graceless kill loses, and against a 24-hour window neither side of that
# trade is deployment-specific.
_FLUSH_INTERVAL_SECONDS = 60

# Every value the /validate handler can report as its `method`: the twelve named
# paths plus the middleware's own default for a request that completed no path.
# A closed set is what keeps this bounded -- `method` reaches here from a
# response header, and an unrecognized value would otherwise mint a key in the
# accumulator, in the stats document, and in the telemetry payload.
KNOWN_AUTH_PATHS: frozenset[str] = frozenset(
    {
        "session_cookie",
        "self_signed",
        "jwt",
        "boto3",
        "federation-static",
        "network-trusted",
        "cognito",
        "keycloak",
        "entra",
        "okta",
        "auth0",
        "pingfederate",
        "unknown",
    }
)

# Counts since the last flush. Only the event loop touches this, so the drain in
# flush_once needs no lock: the swap has no await between read and rebind.
_counts: dict[str, int] = defaultdict(int)


def record(method: str) -> None:
    """Count one /validate request against its auth path.

    Called inline from the middleware's ``finally``, so it must be cheap and it
    must not raise. A value outside :data:`KNOWN_AUTH_PATHS` is dropped.
    """
    if method not in KNOWN_AUTH_PATHS:
        return
    _counts[method] += 1


async def flush_once() -> None:
    """Write the accumulated counts to the shared stats document.

    Drains before writing, not after: a failed write loses one interval instead
    of folding it into the next one, and for a 24-hour share losing a minute
    beats reporting a minute twice.
    """
    global _counts

    try:
        drained, _counts = _counts, defaultdict(int)
        if not drained:
            return

        # Imported here rather than at module scope, as search_routes.py does
        # for the search counter, so importing this module does not drag the
        # registry's repository stack into the auth-server's import graph.
        from registry.repositories.stats_repository import increment_auth_path_counters

        await increment_auth_path_counters(dict(drained))
        auth_path_flush_total.add(1, {"outcome": "ok"})
    except Exception as e:
        # The one place the write's failure is absorbed: increment_auth_path_counters
        # raises on purpose so this except can tell a flush apart from a no-op, and
        # the loop must survive either way. WARNING, not DEBUG, because a flush that
        # fails quietly looks exactly like a deployment with no traffic. The exception
        # type only -- a driver error's message carries the connection host and its
        # whole topology description.
        logger.warning("[auth-path] flush failed: %s", type(e).__name__)
        auth_path_flush_total.add(1, {"outcome": "error"})


async def flush_loop() -> None:
    """Flush on a fixed interval until cancelled.

    The first flush lands one interval in. A graceful shutdown gets the tail of
    the interval from the lifespan, which flushes once more after cancelling.
    """
    while True:
        await asyncio.sleep(_FLUSH_INTERVAL_SECONDS)
        await flush_once()
