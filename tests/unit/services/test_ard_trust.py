"""Unit tests for ARD domain-anchored trust (issue #1296)."""

from registry.schemas.ard_models import ArdCatalogEntry
from registry.schemas.federation_schema import AiCatalogSourceConfig
from registry.services import ard_trust as t


def _entry(identifier):
    return ArdCatalogEntry(
        identifier=identifier, display_name="X",
        type="application/mcp-server-card+json", url="https://acme.com/x",
    )


_SRC = AiCatalogSourceConfig(source_id="acme", domain="acme.com")


class TestHostIdentityDomain:
    def test_https_url(self):
        assert t.host_identity_domain("https://acme.com/path") == "acme.com"

    def test_bare_domain(self):
        assert t.host_identity_domain("acme.com") == "acme.com"

    def test_empty(self):
        assert t.host_identity_domain("") is None


class TestVerifyEntryTrust:
    def test_match_accepts(self):
        ok, reason = t.verify_entry_trust(
            _entry("urn:air:acme.com:server:x"), "acme.com", _SRC, "reject"
        )
        assert ok is True
        assert reason is None

    def test_mismatch_rejected_under_reject(self):
        ok, reason = t.verify_entry_trust(
            _entry("urn:air:victim.com:server:x"), "acme.com", _SRC, "reject"
        )
        assert ok is False
        assert "victim.com" in reason

    def test_mismatch_accepted_under_flag(self):
        ok, reason = t.verify_entry_trust(
            _entry("urn:air:victim.com:server:x"), "acme.com", _SRC, "flag"
        )
        assert ok is True
        assert reason is not None

    def test_off_accepts_everything(self):
        ok, reason = t.verify_entry_trust(
            _entry("urn:air:victim.com:server:x"), "acme.com", _SRC, "off"
        )
        assert ok is True
        assert reason is None

    def test_unparseable_urn_rejected(self):
        ok, reason = t.verify_entry_trust(_entry("garbage"), "acme.com", _SRC, "reject")
        assert ok is False
        assert reason == "unparseable-urn"

    def test_expected_identity_pin_overrides_host(self):
        src = AiCatalogSourceConfig(source_id="acme", domain="acme.com", expected_identity="https://pinned.com")
        # URN publisher matches the pin, not the served host -> accepted.
        ok, _ = t.verify_entry_trust(
            _entry("urn:air:pinned.com:server:x"), "acme.com", src, "reject"
        )
        assert ok is True

    def test_anchors_to_source_not_self_declared_identity(self):
        # A source configured as acme.com serves a catalog whose manifest declares
        # identity=victim.com with victim.com URNs (impersonation). Anchoring to the
        # OPERATOR source domain (acme.com) must reject it, even though the entry is
        # internally self-consistent with the served manifest identity.
        ok, reason = t.verify_entry_trust(
            _entry("urn:air:victim.com:server:x"), "victim.com", _SRC, "reject"
        )
        assert ok is False
        assert "victim.com" in reason
