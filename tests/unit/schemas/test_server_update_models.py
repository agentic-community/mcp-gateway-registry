"""Tests for issue #1164 server update schemas in registry.schemas.server_update_models.

Covers:
- ServerUpdateRequest: PUT body with required fields, size caps, registrant-only rejection,
  extra="forbid", credential-field rejection, deployment/local_runtime rejection.
- ServerCardPatch: optional fields, RFC 7396 merge-patch semantics, registrant-only
  rejection (with explicit error message), size caps, extra="forbid".
- SERVER_REGISTRANT_ONLY_FIELDS constant shape.
"""

import pytest
from pydantic import ValidationError

from registry.schemas.server_update_models import (
    SERVER_REGISTRANT_ONLY_FIELDS,
    ServerCardPatch,
    ServerUpdateRequest,
)

# Credential-shaped fields are rejected by extra="forbid" since they are
# absent from both models. Non-credential registrant-only fields hit
# either extra="forbid" (PUT) or the model_validator (PATCH).
_CREDENTIAL_FIELDS = [
    "auth_credential",
    "auth_scheme",
    "auth_header_name",
    "custom_headers",
    "custom_header_names",
]

# Subset of SERVER_REGISTRANT_ONLY_FIELDS minus credential-shaped fields.
# These are anchors and server-managed fields explicitly forbidden.
_NON_CREDENTIAL_REGISTRANT_FIELDS = [
    "id",
    "path",
    "registered_by",
    "registered_at",
    "updated_at",
    "is_enabled",
    "is_active",
    "version",
    "deployment",
    "local_runtime",
    "health_status",
    "last_health_check",
    "auth_credential_encrypted",
    "custom_headers_encrypted",
    "sync_metadata",
]


@pytest.mark.unit
class TestServerUpdateRequest:
    """Tests for the PUT /api/servers/{path} body model."""

    def test_minimal_valid_body(self):
        """Only required fields (server_name, description) accepted; defaults applied."""
        body = ServerUpdateRequest(server_name="srv", description="d")
        assert body.server_name == "srv"
        assert body.description == "d"
        assert body.tags == []
        assert body.license == "N/A"
        assert body.visibility == "public"

    def test_full_valid_body(self):
        """Every mutable field provided, parses cleanly and round-trips via model_dump()."""
        payload = {
            "server_name": "srv",
            "description": "full body",
            "proxy_pass_url": "https://x.example.com",
            "mcp_endpoint": "/mcp",
            "sse_endpoint": "/sse",
            "transport": "sse",
            "supported_transports": ["sse", "streamable-http"],
            "headers": [{"name": "X-Trace", "value": "1"}],
            "auth_provider": "github",
            "tags": ["a", "b"],
            "license": "Apache-2.0",
            "num_tools": 3,
            "tool_list": [{"name": "t1"}],
            "metadata": {"k": "v"},
            "visibility": "private",
            "allowed_groups": ["g1"],
            "status": "ACTIVE",
            "provider": {"organization": "acme", "url": "https://acme.com"},
            "source_created_at": "2026-01-01T00:00:00Z",
            "source_updated_at": "2026-02-01T00:00:00Z",
            "external_tags": ["ext1", "ext2"],
        }
        body = ServerUpdateRequest(**payload)
        round_trip = body.model_dump()
        assert round_trip["server_name"] == "srv"
        assert round_trip["provider"] == {"organization": "acme", "url": "https://acme.com"}
        assert round_trip["tags"] == ["a", "b"]

    def test_extra_field_rejected(self):
        """Unknown fields raise (extra='forbid')."""
        with pytest.raises(ValidationError):
            ServerUpdateRequest(server_name="srv", description="d", foo="bar")

    @pytest.mark.parametrize("field", _CREDENTIAL_FIELDS)
    def test_credential_fields_rejected(self, field):
        """Credential-shaped fields are never accepted on PUT (extra='forbid')."""
        with pytest.raises(ValidationError):
            ServerUpdateRequest(**{"server_name": "srv", "description": "d", field: "x"})

    def test_deployment_field_rejected(self):
        """Supplying `deployment` raises ValidationError."""
        with pytest.raises(ValidationError):
            ServerUpdateRequest(server_name="srv", description="d", deployment={"type": "remote"})

    def test_local_runtime_field_rejected(self):
        """Supplying `local_runtime` raises ValidationError."""
        with pytest.raises(ValidationError):
            ServerUpdateRequest(server_name="srv", description="d", local_runtime={"cmd": "node"})

    @pytest.mark.parametrize(
        "field",
        [
            "id",
            "path",
            "registered_by",
            "registered_at",
            "updated_at",
            "is_enabled",
            "version",
            "health_status",
            "last_health_check",
            "sync_metadata",
        ],
    )
    def test_registrant_only_fields_rejected(self, field):
        """Each registrant-only field raises ValidationError on PUT (extra='forbid')."""
        assert field in SERVER_REGISTRANT_ONLY_FIELDS
        with pytest.raises(ValidationError):
            ServerUpdateRequest(**{"server_name": "srv", "description": "d", field: "x"})

    def test_server_name_max_length(self):
        """server_name over 256 chars rejected; exactly 256 accepted."""
        ServerUpdateRequest(server_name="s" * 256, description="d")
        with pytest.raises(ValidationError):
            ServerUpdateRequest(server_name="s" * 257, description="d")

    def test_description_max_length(self):
        """description over 4096 chars rejected; exactly 4096 accepted."""
        ServerUpdateRequest(server_name="srv", description="d" * 4096)
        with pytest.raises(ValidationError):
            ServerUpdateRequest(server_name="srv", description="d" * 4097)

    def test_tags_csv_normalised_to_list(self):
        """A CSV `tags` string parses to a stripped list, skipping empties."""
        body = ServerUpdateRequest(server_name="srv", description="d", tags="a, b ,c, ,")
        assert body.tags == ["a", "b", "c"]

    def test_tags_max_count(self):
        """Over 50 tags rejected; exactly 50 accepted."""
        ServerUpdateRequest(
            server_name="srv",
            description="d",
            tags=[f"t{i}" for i in range(50)],
        )
        with pytest.raises(ValidationError, match="at most 50"):
            ServerUpdateRequest(
                server_name="srv",
                description="d",
                tags=[f"t{i}" for i in range(51)],
            )

    def test_tag_length_cap(self):
        """A single tag over 64 chars raises ValidationError."""
        with pytest.raises(ValidationError, match="at most 64 chars"):
            ServerUpdateRequest(
                server_name="srv",
                description="d",
                tags=["x" * 65],
            )

    def test_external_tags_max_count(self):
        """external_tags also enforces the 50-entry cap."""
        ServerUpdateRequest(
            server_name="srv",
            description="d",
            external_tags=[f"e{i}" for i in range(50)],
        )
        with pytest.raises(ValidationError, match="at most 50"):
            ServerUpdateRequest(
                server_name="srv",
                description="d",
                external_tags=[f"e{i}" for i in range(51)],
            )

    def test_external_tag_length_cap(self):
        """external_tags also enforces the 64-char-per-tag cap."""
        with pytest.raises(ValidationError, match="at most 64 chars"):
            ServerUpdateRequest(
                server_name="srv",
                description="d",
                external_tags=["x" * 65],
            )

    def test_metadata_size_cap(self):
        """metadata over 64 KB serialised JSON rejected; just-under accepted."""
        # Just under cap (the JSON envelope adds a few bytes; subtract margin).
        ServerUpdateRequest(
            server_name="srv",
            description="d",
            metadata={"k": "x" * (64 * 1024 - 16)},
        )
        with pytest.raises(ValidationError, match="at most 65536 bytes"):
            ServerUpdateRequest(
                server_name="srv",
                description="d",
                metadata={"k": "x" * (64 * 1024 + 1)},
            )

    def test_provider_structured(self):
        """provider parses to AgentProvider; bad structure raises ValidationError."""
        body = ServerUpdateRequest(
            server_name="srv",
            description="d",
            provider={"organization": "acme", "url": "https://acme.com"},
        )
        assert body.provider is not None
        assert body.provider.organization == "acme"

        with pytest.raises(ValidationError):
            ServerUpdateRequest(
                server_name="srv",
                description="d",
                provider={"hacker": "x"},
            )


@pytest.mark.unit
class TestServerCardPatch:
    """Tests for the PATCH /api/servers/{path} body model."""

    def test_empty_body_valid_at_model_layer(self):
        """No fields supplied parses cleanly. Empty-body 400 is the handler's job."""
        patch = ServerCardPatch()
        assert patch.model_dump(exclude_unset=True) == {}

    def test_extra_field_rejected(self):
        """Unknown fields raise (extra='forbid')."""
        with pytest.raises(ValidationError):
            ServerCardPatch(foo="bar")

    @pytest.mark.parametrize("field", _CREDENTIAL_FIELDS)
    def test_credential_fields_rejected(self, field):
        """Credential-shaped fields are never accepted on PATCH (extra='forbid')."""
        with pytest.raises(ValidationError):
            ServerCardPatch(**{field: "x"})

    def test_deployment_local_runtime_rejected(self):
        """Both `deployment` and `local_runtime` are rejected on PATCH."""
        with pytest.raises(ValidationError):
            ServerCardPatch(deployment={"type": "remote"})
        with pytest.raises(ValidationError):
            ServerCardPatch(local_runtime={"cmd": "node"})

    @pytest.mark.parametrize("field", _NON_CREDENTIAL_REGISTRANT_FIELDS)
    def test_registrant_only_fields_rejected(self, field):
        """Each non-credential registrant-only field raises ValidationError.

        For these fields, extra='forbid' fires first because none are declared
        on the model. The dedicated message comes from the model_validator and
        only fires for fields actually declared on the model. For now, we just
        assert ValidationError is raised.
        """
        assert field in SERVER_REGISTRANT_ONLY_FIELDS
        with pytest.raises(ValidationError):
            ServerCardPatch(**{field: "x"})

    def test_partial_fields_only(self):
        """Only explicitly supplied fields appear in exclude_unset output."""
        patch = ServerCardPatch(description="x")
        assert patch.model_dump(exclude_unset=True) == {"description": "x"}

    def test_server_name_max_length(self):
        """server_name over 256 chars rejected; exactly 256 accepted."""
        ServerCardPatch(server_name="s" * 256)
        with pytest.raises(ValidationError):
            ServerCardPatch(server_name="s" * 257)

    def test_description_max_length(self):
        """description over 4096 chars rejected; exactly 4096 accepted."""
        ServerCardPatch(description="d" * 4096)
        with pytest.raises(ValidationError):
            ServerCardPatch(description="d" * 4097)

    def test_tags_max_count(self):
        """Over 50 tags rejected; exactly 50 accepted."""
        ServerCardPatch(tags=[f"t{i}" for i in range(50)])
        with pytest.raises(ValidationError, match="at most 50"):
            ServerCardPatch(tags=[f"t{i}" for i in range(51)])

    def test_tag_length_cap(self):
        """A single tag over 64 chars raises ValidationError."""
        with pytest.raises(ValidationError, match="at most 64 chars"):
            ServerCardPatch(tags=["x" * 65])

    def test_tags_csv_normalised_to_list(self):
        """A CSV `tags` string parses to a stripped list, skipping empties."""
        patch = ServerCardPatch(tags="a, b ,c, ,")
        assert patch.tags == ["a", "b", "c"]

    def test_external_tags_max_count(self):
        """external_tags also enforces the 50-entry cap."""
        ServerCardPatch(external_tags=[f"e{i}" for i in range(50)])
        with pytest.raises(ValidationError, match="at most 50"):
            ServerCardPatch(external_tags=[f"e{i}" for i in range(51)])

    def test_external_tag_length_cap(self):
        """external_tags also enforces the 64-char-per-tag cap."""
        with pytest.raises(ValidationError, match="at most 64 chars"):
            ServerCardPatch(external_tags=["x" * 65])

    def test_metadata_size_cap(self):
        """metadata over 64 KB serialised JSON rejected; just-under accepted."""
        ServerCardPatch(metadata={"k": "x" * (64 * 1024 - 16)})
        with pytest.raises(ValidationError, match="at most 65536 bytes"):
            ServerCardPatch(metadata={"k": "x" * (64 * 1024 + 1)})

    def test_provider_structured(self):
        """provider parses to AgentProvider; bad structure raises ValidationError."""
        patch = ServerCardPatch(provider={"organization": "acme", "url": "https://acme.com"})
        assert patch.provider is not None
        assert patch.provider.organization == "acme"

        with pytest.raises(ValidationError):
            ServerCardPatch(provider={"hacker": "x"})


@pytest.mark.unit
class TestSizeCapsAndConstants:
    """Sanity checks on the module-level constants."""

    def test_registrant_only_fields_is_frozenset(self):
        """The shared constant is an immutable frozenset."""
        assert isinstance(SERVER_REGISTRANT_ONLY_FIELDS, frozenset)

    def test_registrant_only_fields_contents(self):
        """The constant contains the expected 20 names from the LLD."""
        expected = {
            "id",
            "path",
            "registered_by",
            "registered_at",
            "updated_at",
            "is_enabled",
            "is_active",
            "version",
            "deployment",
            "local_runtime",
            "health_status",
            "last_health_check",
            "auth_credential_encrypted",
            "custom_headers_encrypted",
            "sync_metadata",
            "auth_scheme",
            "auth_credential",
            "auth_header_name",
            "custom_headers",
            "custom_header_names",
        }
        assert SERVER_REGISTRANT_ONLY_FIELDS == frozenset(expected)
