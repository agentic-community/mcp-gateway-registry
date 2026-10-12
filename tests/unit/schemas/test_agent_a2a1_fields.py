"""Unit tests for the A2A 1.0 AgentCard schema fields.

Covers the three A2A 1.0 fields added to AgentCard:
- supported_interfaces (list[AgentInterface]): one entry per transport
  binding, so an agent can advertise more than one binding for a protocol.
- security_requirements (list[SecurityRequirement]): the A2A 1.0
  SecurityRequirement shape, which wraps the scheme->scopes map under a
  "schemes" field.
- signatures (list[AgentCardSignature]): the structured JWS array form.
"""

import pytest
from pydantic import ValidationError

from registry.schemas.agent_models import (
    AgentCard,
    AgentCardPatch,
    AgentCardSignature,
    AgentInterface,
    AgentRegistrationRequest,
    SecurityRequirement,
    StringList,
)



_REAL_CARD_INTERFACES = [
    {
        "url": "https://agent.example.com/a2a",
        "protocolBinding": "JSONRPC",
        "protocolVersion": "1.0",
    },
    {
        "url": "https://agent.example.com/a2a",
        "protocolBinding": "HTTP+JSON",
        "protocolVersion": "1.0",
    },
]


class TestAgentInterfaceModel:
    def test_accepts_camel_case_input(self):
        iface = AgentInterface(**_REAL_CARD_INTERFACES[0])

        assert iface.url == _REAL_CARD_INTERFACES[0]["url"]
        assert iface.protocol_binding == "JSONRPC"
        assert iface.protocol_version == "1.0"

    def test_accepts_snake_case_input(self):
        iface = AgentInterface(
            url="https://agent.example.net/a2a",
            protocol_binding="HTTP+JSON",
            protocol_version="1.0",
        )

        assert iface.protocol_binding == "HTTP+JSON"

    def test_serializes_camel_case_by_alias(self):
        iface = AgentInterface(**_REAL_CARD_INTERFACES[0])

        dumped = iface.model_dump(by_alias=True)

        assert dumped["protocolBinding"] == "JSONRPC"
        assert dumped["protocolVersion"] == "1.0"

    @pytest.mark.parametrize("binding", ["JSONRPC", "HTTP+JSON", "GRPC"])
    def test_accepts_well_known_protocol_bindings(self, binding):
        iface = AgentInterface(
            url="https://agent.example.net/a2a",
            protocolBinding=binding,
            protocolVersion="1.0",
        )

        assert iface.protocol_binding == binding

    def test_accepts_custom_protocol_binding(self):
        """A2A 1.0 treats protocol_binding as an open string: custom
        transports are identified by a URI, not restricted to the three
        values a2a.proto names explicitly. A closed enum would reject a
        genuinely valid card.
        """
        iface = AgentInterface(
            url="https://agent.example.net/a2a",
            protocolBinding="https://example.com/transports/websocket",
            protocolVersion="1.0",
        )

        assert iface.protocol_binding == "https://example.com/transports/websocket"

    def test_rejects_empty_protocol_binding(self):
        with pytest.raises(ValidationError, match="protocol_binding cannot be empty"):
            AgentInterface(
                url="https://agent.example.net/a2a",
                protocolBinding="",
                protocolVersion="1.0",
            )

    def test_grpc_accepts_bare_authority_address(self):
        """gRPC targets are conventionally host:port with no scheme, so
        the generic HTTP(S)-only url check would otherwise reject every
        real-world gRPC interface.
        """
        iface = AgentInterface(
            url="grpc.example.com:443",
            protocolBinding="GRPC",
            protocolVersion="1.0",
        )

        assert iface.url == "grpc.example.com:443"

    @pytest.mark.parametrize(
        "url",
        [
            "[::1]:50051",
            "[2001:db8::1]:443",
        ],
    )
    def test_grpc_accepts_bracketed_ipv6_authority(self, url):
        """IPv6 addresses need brackets to disambiguate their own colons
        from the port separator; a plain host:port split would mis-parse
        these.
        """
        iface = AgentInterface(url=url, protocolBinding="GRPC", protocolVersion="1.0")

        assert iface.url == url

    @pytest.mark.parametrize(
        "url",
        [
            ".:99999",  # empty host label, port out of range
            "-:0",  # host label is only a hyphen, port 0
            "grpc.example.com:0",  # port 0 is not a valid TCP port
            "grpc.example.com:99999",  # port out of 1-65535 range
            "[not-ipv6]:443",  # bracketed host is not a valid IPv6 address
        ],
    )
    def test_grpc_rejects_malformed_authority(self, url):
        """A loose character-class regex would accept all of these;
        each must be rejected by proper hostname/IPv6/port validation.
        """
        with pytest.raises(ValidationError):
            AgentInterface(url=url, protocolBinding="GRPC", protocolVersion="1.0")

    def test_grpc_also_accepts_full_http_url(self):
        """A gateway fronting gRPC behind HTTP(S) is still valid."""
        iface = AgentInterface(
            url="https://grpc.example.com",
            protocolBinding="GRPC",
            protocolVersion="1.0",
        )

        assert iface.url == "https://grpc.example.com"

    def test_non_grpc_binding_rejects_bare_authority_address(self):
        """Only GRPC gets the host:port exemption; JSONRPC/HTTP+JSON/
        custom bindings still require a full HTTP(S) URL.
        """
        with pytest.raises(ValidationError):
            AgentInterface(
                url="jsonrpc.example.com:443",
                protocolBinding="JSONRPC",
                protocolVersion="1.0",
            )

    def test_accepts_optional_tenant(self):
        iface = AgentInterface(
            url="https://agent.example.net/a2a",
            protocolBinding="JSONRPC",
            protocolVersion="1.0",
            tenant="acme-corp",
        )

        assert iface.tenant == "acme-corp"

    def test_tenant_defaults_to_none(self):
        iface = AgentInterface(**_REAL_CARD_INTERFACES[0])

        assert iface.tenant is None

    def test_tenant_survives_round_trip_through_agent_card(self):
        """Pydantic ignores unknown fields by default; without an
        explicit `tenant` field on AgentInterface, a card registered with
        a tenant-routed interface would silently lose it.
        """
        card = AgentCard(
            name="test-agent",
            description="A test agent.",
            url="https://agent.example.net",
            version="1.0.0",
            supportedInterfaces=[{**_REAL_CARD_INTERFACES[0], "tenant": "acme-corp"}],
        )

        assert card.supported_interfaces[0].tenant == "acme-corp"

    def test_rejects_invalid_url(self):
        with pytest.raises(ValidationError):
            AgentInterface(
                url="not-a-url",
                protocolBinding="JSONRPC",
                protocolVersion="1.0",
            )

    def test_rejects_invalid_protocol_version(self):
        with pytest.raises(ValidationError):
            AgentInterface(
                url="https://agent.example.net/a2a",
                protocolBinding="JSONRPC",
                protocolVersion="not-a-version",
            )


class TestAgentCardSupportedInterfaces:
    def _base_card_kwargs(self) -> dict:
        return {
            "name": "test-agent",
            "description": "A test agent.",
            "url": "https://agent.example.net",
            "version": "1.0.0",
        }

    def test_defaults_to_empty_list(self):
        card = AgentCard(**self._base_card_kwargs())

        assert card.supported_interfaces == []

    def test_accepts_multiple_interfaces_for_same_protocol(self):
        """The exact scenario preferred_transport (a single string) cannot
        represent: one agent, two transport bindings for the same protocol.
        """
        card = AgentCard(
            **self._base_card_kwargs(),
            supportedInterfaces=_REAL_CARD_INTERFACES,
        )

        assert len(card.supported_interfaces) == 2
        bindings = {iface.protocol_binding for iface in card.supported_interfaces}
        assert bindings == {"JSONRPC", "HTTP+JSON"}

    def test_preferred_transport_is_untouched(self):
        """This field adds a new field; it must not remove or change the
        meaning of the existing preferred_transport field.
        """
        card = AgentCard(**self._base_card_kwargs())

        assert card.preferred_transport == "JSONRPC"

    def test_serializes_supported_interfaces_by_alias(self):
        card = AgentCard(
            **self._base_card_kwargs(),
            supportedInterfaces=[_REAL_CARD_INTERFACES[0]],
        )

        dumped = card.model_dump(by_alias=True)

        assert dumped["supportedInterfaces"][0]["protocolBinding"] == "JSONRPC"


class TestAgentRegistrationRequestSupportedInterfaces:
    def test_optional_at_registration(self):
        request = AgentRegistrationRequest(
            name="test-agent",
            url="https://agent.example.net",
            supportedProtocol="a2a",
        )

        assert request.supported_interfaces is None

    def test_accepts_camel_case_interface_list(self):
        request = AgentRegistrationRequest(
            name="test-agent",
            url="https://agent.example.net",
            supportedProtocol="a2a",
            supportedInterfaces=_REAL_CARD_INTERFACES,
        )

        assert request.supported_interfaces == _REAL_CARD_INTERFACES


class TestAgentCardPatchSupportedInterfaces:
    def test_accepts_supported_interfaces(self):
        patch = AgentCardPatch(supportedInterfaces=_REAL_CARD_INTERFACES)

        assert patch.supported_interfaces == _REAL_CARD_INTERFACES

    def test_omitted_by_default(self):
        patch = AgentCardPatch(name="renamed-agent")

        assert "supported_interfaces" not in patch.model_dump(exclude_unset=True)


class TestSecurityRequirementsA2A1Shape:
    """GENAIP-7937 gap 2: spec-exact securityRequirements field (patch 024).

    A2A 1.0's SecurityRequirement wraps a map<string, StringList> under a
    "schemes" field (a2a.proto: `message SecurityRequirement { map<string,
    StringList> schemes = 1; }`), confirmed against the authoritative
    proto at https://github.com/a2aproject/A2A/blob/main/specification/a2a.proto.
    It is NOT a bare map<string, StringList> inlined onto the entry itself.
    This field co-exists with the legacy 'security' field; neither derives
    the other.
    """

    def test_string_list_accepts_wire_alias(self):
        sl = StringList(list=["read", "write"])
        assert sl.values == ["read", "write"]
        assert sl.model_dump(by_alias=True) == {"list": ["read", "write"]}

    def test_security_requirement_wraps_schemes_map(self):
        req = SecurityRequirement(schemes={"oauth2": {"list": ["read"]}})
        assert req.schemes["oauth2"].values == ["read"]
        assert req.model_dump(by_alias=True) == {"schemes": {"oauth2": {"list": ["read"]}}}

    def test_security_requirement_rejects_unwrapped_shape(self):
        # Regression: without extra="forbid", the pre-spec unwrapped shape
        # (scheme names as top-level keys) was accepted as a valid but
        # field-less SecurityRequirement: "schemes" silently defaulted to
        # {} and the real scheme/scope data was discarded instead of being
        # rejected.
        with pytest.raises(ValidationError):
            SecurityRequirement(**{"oauth2": {"list": ["read"]}})

    def test_agent_card_rejects_unwrapped_security_requirements(self):
        with pytest.raises(ValidationError):
            AgentCard(
                name="a",
                description="d",
                url="https://example.com",
                version="1.0",
                securityRequirements=[{"oauth2": {"list": ["read"]}}],
            )

    def test_agent_card_round_trips_security_requirements(self):
        card = AgentCard(
            name="a",
            description="d",
            url="https://example.com",
            version="1.0",
            securitySchemes={
                "oauth2": {"type": "oauth2"},
                "apiKey": {"type": "apiKey"},
            },
            securityRequirements=[
                {"schemes": {"oauth2": {"list": ["read"]}}},
                {"schemes": {"apiKey": {"list": []}}},
            ],
        )
        assert card.security_requirements[0].schemes["oauth2"].values == ["read"]
        dumped = card.model_dump(by_alias=True)["securityRequirements"]
        assert dumped == [
            {"schemes": {"oauth2": {"list": ["read"]}}},
            {"schemes": {"apiKey": {"list": []}}},
        ]

    def test_legacy_security_field_is_untouched(self):
        card = AgentCard(
            name="a",
            description="d",
            url="https://example.com",
            version="1.0",
            securitySchemes={"oauth2": {"type": "oauth2"}},
            security=[{"oauth2": ["read"]}],
            securityRequirements=[{"schemes": {"oauth2": {"list": ["read"]}}}],
        )
        assert card.security == [{"oauth2": ["read"]}]
        assert card.security_requirements[0].schemes["oauth2"].values == ["read"]

    def test_security_requirements_reject_undefined_scheme(self):
        # Regression: the validator used to check only the legacy `security`
        # field's scheme references against security_schemes; a
        # securityRequirements entry could reference an undefined scheme
        # even though the equivalent legacy payload would be rejected.
        with pytest.raises(ValidationError, match="undefined scheme"):
            AgentCard(
                name="a",
                description="d",
                url="https://example.com",
                version="1.0",
                securitySchemes={"oauth2": {"type": "oauth2"}},
                securityRequirements=[{"schemes": {"apiKey": {"list": ["read"]}}}],
            )

    def test_security_requirements_accepts_defined_scheme(self):
        card = AgentCard(
            name="a",
            description="d",
            url="https://example.com",
            version="1.0",
            securitySchemes={"oauth2": {"type": "oauth2"}},
            securityRequirements=[{"schemes": {"oauth2": {"list": ["read"]}}}],
        )
        assert card.security_requirements[0].schemes["oauth2"].values == ["read"]

    def test_registration_request_accepts_raw_dicts(self):
        # AgentRegistrationRequest.security_requirements is deliberately
        # untyped (list[dict[str, Any]]) at this layer; the dict is passed
        # through unvalidated and only becomes a real SecurityRequirement
        # (schemes-wrapped, extra="forbid") when merged into AgentCard. Use
        # the correct A2A wire shape here regardless, so this test doesn't
        # read as an example of valid input; it would be rejected one
        # layer down.
        req = AgentRegistrationRequest(
            name="a",
            url="https://example.com",
            supportedProtocol="a2a",
            securityRequirements=[{"schemes": {"oauth2": {"list": ["read"]}}}],
        )
        assert req.security_requirements == [{"schemes": {"oauth2": {"list": ["read"]}}}]


class TestSignaturesA2A1Shape:
    """GENAIP-7937 gap 3: spec-exact signatures array (patch 024).

    A2A 1.0's AgentCard.signatures is an array of detached-JWS entries
    (protected header + signature, both base64url, optional cleartext
    header). Co-exists with the legacy singular 'signature' string field.
    """

    def test_agent_card_round_trips_signatures(self):
        card = AgentCard(
            name="a",
            description="d",
            url="https://example.com",
            version="1.0",
            signatures=[{"protected": "abc", "signature": "xyz"}],
        )
        assert card.signatures == [AgentCardSignature(protected="abc", signature="xyz")]
        assert card.model_dump(by_alias=True)["signatures"] == [
            {"protected": "abc", "signature": "xyz", "header": None}
        ]

    def test_legacy_signature_field_is_untouched(self):
        card = AgentCard(
            name="a",
            description="d",
            url="https://example.com",
            version="1.0",
            signature="legacy-jws-string",
            signatures=[{"protected": "abc", "signature": "xyz"}],
        )
        assert card.signature == "legacy-jws-string"
        assert len(card.signatures) == 1

    def test_registration_request_accepts_raw_dicts(self):
        req = AgentRegistrationRequest(
            name="a",
            url="https://example.com",
            supportedProtocol="a2a",
            signatures=[{"protected": "abc", "signature": "xyz", "header": {"kid": "k1"}}],
        )
        assert req.signatures == [{"protected": "abc", "signature": "xyz", "header": {"kid": "k1"}}]


