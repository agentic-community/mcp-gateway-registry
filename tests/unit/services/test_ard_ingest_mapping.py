"""Unit tests for ARD ingestion reverse-mapping (issue #1296)."""

from registry.schemas.ard_models import ArdCatalogEntry
from registry.services import ard_ingest_mapping as m


def _entry(identifier, type_):
    return ArdCatalogEntry(
        identifier=identifier, display_name="X", type=type_, url="https://acme.com/x",
        description="d", tags=["t"], version="1.0.0",
    )


class TestParseUrnPublisher:
    def test_extracts_publisher(self):
        assert m.parse_urn_publisher("urn:air:acme.com:server:github") == "acme.com"

    def test_bad_urn_returns_none(self):
        assert m.parse_urn_publisher("not-a-urn") is None

    def test_empty_returns_none(self):
        assert m.parse_urn_publisher("") is None


class TestEntryToRecord:
    def test_server_maps_to_unprefixed_path_and_markers(self):
        kind, path, record = m.entry_to_record(
            _entry("urn:air:acme.com:server:github", "application/mcp-server-card+json"), "acme"
        )
        assert kind == "server"
        assert path == "/github"  # peer-sync adds the /{source_id} prefix
        assert record["server_name"] == "X"
        assert record["registry_name"] == "acme"
        assert record["is_read_only"] is True
        assert record["record_kind"] == "ard_ingested"
        # Origin markers for UI classification + dynamic source grouping.
        assert {"federated", "ard", "acme"}.issubset(set(record["tags"]))
        # The resolve link: original source url + identifier preserved.
        assert record["ard_source_url"] == "https://acme.com/x"
        assert record["ard_source_identifier"] == "urn:air:acme.com:server:github"
        assert record["ard_source_entry"]["identifier"] == "urn:air:acme.com:server:github"

    def test_agent_uses_name_key(self):
        kind, _path, record = m.entry_to_record(
            _entry("urn:air:acme.com:agent:trav", "application/a2a-agent-card+json"), "acme"
        )
        assert kind == "agent"
        assert record["name"] == "X"
        # A2A agent-card requires url + version (else registration drops the agent).
        assert record["url"] == "https://acme.com/x"
        assert record["version"] == "1.0.0"
        assert record["capabilities"] == {}
        assert record["skills"] == []

    def test_skill_kind(self):
        kind, _path, _record = m.entry_to_record(
            _entry("urn:air:acme.com:skill:pdf", "application/ai-skill"), "acme"
        )
        assert kind == "skill"

    def test_registry_type_is_unsupported(self):
        assert m.entry_to_record(
            _entry("urn:air:acme.com:registry:self", "application/ai-registry+json"), "acme"
        ) is None

    def test_catalog_type_is_unsupported(self):
        assert m.entry_to_record(
            _entry("urn:air:acme.com:catalog:x", "application/ai-catalog+json"), "acme"
        ) is None
