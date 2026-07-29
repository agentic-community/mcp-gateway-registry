"""Drift guard for the audit health-check allowlist.

``_HEALTH_CHECK_EXACT_PATHS`` in ``registry.audit.middleware`` is a
hand-maintained mirror of the real monitoring routes. When
``audit_log_health_checks`` is False, any path in that set is skipped from the
audit trail. Drift is only ever over-audit-safe (a stale entry means a REMOVED
health route keeps getting skipped — mild), never under-audit (the exact/prefix
match already fails closed for anything not provably a monitoring endpoint — see
``test_audit_health_path_bypass.py``). But a silently rotting allowlist is still
worth pinning so an edit forces a conscious update.

Why a PIN rather than a live route-enumeration assert: the health endpoints are
registered across several routers (``registry/health/routes.py`` mounted at
``/api/health``, plus federation / ANS / server routers), and the registry
FastAPI ``app.routes`` does not surface every one of them as a literal path in a
unit-test harness (sub-router prefixes, conditionally-included routers). A
naive "every entry must be an app route" test would produce false failures and
couple this suite to full-app import/config. The pin below records the INTENDED
set with the source route for each entry (kept in lockstep with the
cross-reference comment above the constant). If someone edits the constant
without updating this pin, the test fails loudly and points at the rationale.
"""

from registry.audit.middleware import (
    _HEALTH_CHECK_EXACT_PATHS,
    _HEALTH_CHECK_PREFIX,
    _is_health_check_path,
)

# The intended allowlist, each entry mapped to the route that justifies it.
# Update BOTH this dict and the cross-reference comment in middleware.py when a
# monitoring route is added, removed, or renamed.
_INTENDED_HEALTH_PATHS: dict[str, str] = {
    "/health": 'registry/main.py @app.get("/health")',
    "/api/health": "registry/health/routes.py mounted at prefix /api/health",
    "/api/federation/health": (
        "registry/api/federation_export_routes.py GET /api/federation/health"
    ),
    "/api/admin/ans/health": "registry/api/ans_routes.py GET /api/admin/ans/health",
    "/api/internal/healthcheck": ("registry/api/server_routes.py POST /api/internal/healthcheck"),
    "/api/servers/health": "registry/api/server_routes.py GET /api/servers/health",
}


class TestHealthAllowlistPin:
    """The allowlist must not silently drift from its documented set."""

    def test_exact_paths_match_the_documented_intent(self):
        """Editing _HEALTH_CHECK_EXACT_PATHS without updating the pin fails here.

        This is the loud regression guard: any add/remove/rename to the constant
        must be mirrored in _INTENDED_HEALTH_PATHS (and its per-entry rationale),
        or this assertion breaks.
        """
        assert _HEALTH_CHECK_EXACT_PATHS == frozenset(_INTENDED_HEALTH_PATHS)

    def test_every_intended_path_is_classified_as_health(self):
        """Each pinned path must actually be recognised by the classifier."""
        for path in _INTENDED_HEALTH_PATHS:
            assert _is_health_check_path(path) is True

    def test_sub_tree_prefix_is_covered(self):
        """The /api/health/ sub-tree (ws/health_status, ws/stats) is a health path."""
        assert _HEALTH_CHECK_PREFIX == "/api/health/"
        assert _is_health_check_path("/api/health/ws/health_status") is True
        assert _is_health_check_path("/api/health/ws/stats") is True

    def test_top_level_health_route_is_really_registered(self):
        """Cheap live check: the one literal top-level route is a real app route.

        ``/health`` is registered directly on the registry app, so we CAN verify
        it against the live route table without the false-positive risk that
        makes the other (sub-router) entries a pin rather than a live assert.
        """
        from registry.main import app

        app_paths = {getattr(r, "path", None) for r in app.routes}
        assert "/health" in app_paths
