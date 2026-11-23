"""Tests for the M2M enrichment gate and privileged-grant audit.

M2M group enrichment reads groups from a mutable DB collection. It is a
legitimate authorization source for machine clients (which carry no group
claim), but it must be strictly gated to M2M tokens and must never silently
escalate a normal user token whose groups were legitimately empty. When
enrichment does add a registry-admin group, that grant must be audited so a
write to the collection is attributable.
"""

from unittest.mock import AsyncMock, patch

import mongodb_groups_enrichment as enrichment
import pytest


class TestShouldEnrichGroupsGate:
    """The M2M enrichment gate must fail closed for anything that is not a
    genuine M2M token with empty groups."""

    def test_m2m_token_with_empty_groups_enriches(self):
        result = {
            "valid": True,
            "groups": [],
            "client_id": "real-m2m-client",
            "token_type": "m2m",
        }
        assert enrichment.should_enrich_groups(result) is True

    def test_user_generated_token_not_enriched(self):
        # A self-signed user token carries token_type "user_generated"; it is
        # not an M2M client and must not be enriched from idp_m2m_clients even
        # when it happens to carry a client_id and has empty groups.
        result = {
            "valid": True,
            "groups": [],
            "client_id": "some-client",
            "token_type": "user_generated",
        }
        assert enrichment.should_enrich_groups(result) is False

    def test_user_generated_sentinel_client_id_not_enriched(self):
        # Fail closed on the sentinel client_id even if token_type is missing.
        result = {
            "valid": True,
            "groups": [],
            "client_id": "user-generated",
        }
        assert enrichment.should_enrich_groups(result) is False

    def test_non_empty_groups_not_enriched(self):
        result = {
            "valid": True,
            "groups": ["developers"],
            "client_id": "real-m2m-client",
            "token_type": "m2m",
        }
        assert enrichment.should_enrich_groups(result) is False

    def test_missing_client_id_not_enriched(self):
        result = {"valid": True, "groups": [], "client_id": None}
        assert enrichment.should_enrich_groups(result) is False

    def test_invalid_token_not_enriched(self):
        result = {"valid": False, "groups": [], "client_id": "real-m2m-client"}
        assert enrichment.should_enrich_groups(result) is False


def _patch_db_single_doc(doc: dict):
    collection = AsyncMock()
    collection.find_one = AsyncMock(return_value=doc)

    class _FakeDB:
        def __getitem__(self, _name: str):
            return collection

    return patch.object(enrichment, "_get_mongodb", AsyncMock(return_value=_FakeDB()))


class TestPrivilegedEnrichmentAudit:
    """A privileged group granted via DB enrichment must be audited."""

    @pytest.mark.asyncio
    async def test_privileged_group_grant_is_audited(self, caplog):
        doc = {"client_id": "svc", "groups": ["registry-admins"], "enabled": True}
        with _patch_db_single_doc(doc), caplog.at_level("WARNING"):
            result = await enrichment.enrich_groups_from_mongodb("svc", [])
        assert result == ["registry-admins"]
        assert any(
            "privileged group" in rec.message and "registry-admins" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_non_privileged_group_grant_not_audited(self, caplog):
        doc = {"client_id": "svc", "groups": ["developers"], "enabled": True}
        with _patch_db_single_doc(doc), caplog.at_level("WARNING"):
            result = await enrichment.enrich_groups_from_mongodb("svc", [])
        assert result == ["developers"]
        assert not any("privileged group" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stored", ["/registry-admins", "registry-admins/", " /mcp-registry-admin "]
    )
    async def test_slashed_privileged_group_grant_is_audited(self, caplog, stored):
        """``/registry-admins`` resolves the admin scope, so it is audited too (review on #1801)."""
        doc = {"client_id": "svc", "groups": [stored], "enabled": True}
        with _patch_db_single_doc(doc), caplog.at_level("WARNING"):
            await enrichment.enrich_groups_from_mongodb("svc", [])
        audit = [rec.message for rec in caplog.records if "privileged group" in rec.message]
        assert len(audit) == 1
        assert stored not in audit[0]

    @pytest.mark.asyncio
    async def test_slashed_privileged_user_group_grant_is_audited(self, caplog):
        """The user enrichment path applies the same identity rule as M2M."""
        doc = {"username": "alice", "groups": ["/registry-admins"], "enabled": True}
        with _patch_db_single_doc(doc), caplog.at_level("WARNING"):
            result = await enrichment.enrich_user_groups_from_mongodb("alice", [], "ping")
        assert result == ["/registry-admins"]
        audit = [rec.message for rec in caplog.records if "privileged group" in rec.message]
        assert len(audit) == 1
        assert "/registry-admins" not in audit[0]

    @pytest.mark.asyncio
    async def test_unrelated_group_containing_marker_text_not_audited(self, caplog):
        """Canonical equality, not substring: ``org/registry-admins`` is another group."""
        doc = {"client_id": "svc", "groups": ["org/registry-admins"], "enabled": True}
        with _patch_db_single_doc(doc), caplog.at_level("WARNING"):
            await enrichment.enrich_groups_from_mongodb("svc", [])
        assert not any("privileged group" in rec.message for rec in caplog.records)
