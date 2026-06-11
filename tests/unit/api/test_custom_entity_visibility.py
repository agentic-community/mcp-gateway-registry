"""Unit tests for registry/api/custom_entity_visibility.py.

These helpers are security-relevant: ``_build_visibility_filter`` (the
Mongo list/count predicate) and ``_user_can_view`` (its single-record Python
twin) MUST stay in lockstep. ``_require_owner_or_admin`` guards the mutating
single-record paths. Tests assert the public / private-owner /
group-restricted / deny-by-default branches for all three.
"""

import logging

import pytest
from fastapi import HTTPException

from registry.api.custom_entity_visibility import (
    _build_visibility_filter,
    _require_owner_or_admin,
    _user_can_view,
)
from registry.schemas.custom_entity_models import CustomEntityRecord

logger = logging.getLogger(__name__)


def _record(
    visibility: str = "private",
    owner: str | None = "alice",
    allowed_groups: list[str] | None = None,
) -> CustomEntityRecord:
    """Build a minimal record for visibility checks."""
    return CustomEntityRecord(
        entity_type="thing",
        name="r",
        visibility=visibility,
        owner=owner,
        allowed_groups=allowed_groups or [],
    )


class TestBuildVisibilityFilter:
    """Admins get no restriction; everyone else gets the three-branch $or."""

    def test_admin_gets_none(self):
        assert _build_visibility_filter({"is_admin": True}) is None

    def test_non_admin_filter_shape(self):
        ctx = {"is_admin": False, "username": "alice", "groups": ["g1"]}
        f = _build_visibility_filter(ctx)
        assert f is not None
        clauses = f["$or"]
        assert {"visibility": "public"} in clauses
        assert {"visibility": "private", "owner": "alice"} in clauses
        assert {
            "visibility": "group-restricted",
            "allowed_groups": {"$in": ["g1"]},
        } in clauses

    def test_no_entity_type_key_to_avoid_clobber(self):
        f = _build_visibility_filter({"username": "alice", "groups": []})
        assert "entity_type" not in f


class TestUserCanView:
    """Single-record analogue mirrors the filter branch-for-branch."""

    def test_admin_sees_everything(self):
        ctx = {"is_admin": True}
        assert _user_can_view(_record(visibility="private", owner="bob"), ctx)

    def test_public_visible_to_all(self):
        assert _user_can_view(_record(visibility="public"), {"username": "x"})

    def test_private_only_owner(self):
        rec = _record(visibility="private", owner="alice")
        assert _user_can_view(rec, {"username": "alice"})
        assert not _user_can_view(rec, {"username": "bob"})

    def test_group_restricted_requires_group_overlap(self):
        rec = _record(visibility="group-restricted", allowed_groups=["g1", "g2"])
        assert _user_can_view(rec, {"username": "x", "groups": ["g2"]})
        assert not _user_can_view(rec, {"username": "x", "groups": ["g9"]})

    def test_unknown_visibility_denied(self):
        rec = _record(visibility="mystery")
        assert not _user_can_view(rec, {"username": "alice"})


class TestRequireOwnerOrAdmin:
    """403 unless the caller owns the record or is an admin."""

    def test_admin_allowed(self):
        _require_owner_or_admin(_record(owner="bob"), {"is_admin": True})

    def test_owner_allowed(self):
        _require_owner_or_admin(_record(owner="alice"), {"username": "alice"})

    def test_non_owner_forbidden(self):
        with pytest.raises(HTTPException) as exc:
            _require_owner_or_admin(_record(owner="alice"), {"username": "bob"})
        assert exc.value.status_code == 403

    def test_record_without_owner_forbidden_for_non_admin(self):
        with pytest.raises(HTTPException) as exc:
            _require_owner_or_admin(_record(owner=None), {"username": "bob"})
        assert exc.value.status_code == 403
