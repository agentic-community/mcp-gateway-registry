"""Unit tests for the canonical server.json transform.

The transform is a pure function over a stored server dict. These tests
exercise the round-trip path (when metadata.mcp_registry_spec is present),
the synthesis fallback path (legacy/bespoke documents), the description
truncation rule, and the reverse-DNS namespace derivation.
"""

import copy

import pytest

from registry.services import canonical_export
from registry.services.canonical_export import (
    DEFAULT_SCHEMA_URL,
    MAX_CANONICAL_DESCRIPTION,
    _reverse_dns_base,
    redact_backend_urls,
    to_canonical,
)

# =============================================================================
# Fixtures and helpers
# =============================================================================


def _stored_remote_with_spec() -> dict:
    """Stored doc registered from canonical server.json (spec preserved)."""
    spec = {
        "$schema": (
            "https://raw.githubusercontent.com/modelcontextprotocol/registry/"
            "main/docs/reference/server-json/draft/server.schema.json"
        ),
        "original_name": "io.example/calculator-mcp",
        "version": "1.2.0",
        "repository": {
            "url": "https://github.com/example/calculator-mcp",
            "source": "github",
        },
        "remotes": [
            {
                "type": "streamable-http",
                "url": "https://backend.example/mcp",
                "headers": [
                    {"name": "X-Tenant", "value": "acme"},
                ],
            }
        ],
        "_meta": {
            "io.modelcontextprotocol.registry/publisher-provided": {
                "publisher": "example",
            }
        },
    }
    return {
        "id": "srv-1",
        "server_name": "Calculator",
        "path": "/calculator",
        "description": "A calculator MCP server",
        "version": "1.2.0",
        "proxy_pass_url": "https://backend.example/mcp",
        "deployment": "remote",
        "supported_transports": ["streamable-http"],
        "tags": ["math"],
        "num_tools": 3,
        "tool_list": [{"name": "add"}],
        "auth_scheme": "none",
        "is_active": True,
        "is_enabled": True,
        "metadata": {"mcp_registry_spec": spec, "extra_field": "kept"},
    }


def _stored_remote_no_spec() -> dict:
    """Legacy/bespoke remote server: no mcp_registry_spec preserved."""
    return {
        "id": "srv-2",
        "server_name": "Legacy Remote",
        "path": "/legacy-remote",
        "description": "Bespoke remote server.",
        "version": "0.5.0",
        "proxy_pass_url": "https://legacy.example/mcp",
        "deployment": "remote",
        "supported_transports": ["sse"],
        "tags": ["legacy"],
        "num_tools": 0,
        "auth_scheme": "bearer",
        "metadata": {},
    }


def _stored_local_no_spec() -> dict:
    """Legacy/bespoke local stdio server: no mcp_registry_spec preserved."""
    return {
        "id": "srv-3",
        "server_name": "Local Calc",
        "path": "/local-calc",
        "description": "Local stdio MCP server.",
        "version": "2.0.0",
        "deployment": "local",
        "supported_transports": ["stdio"],
        "tags": ["math", "local"],
        "num_tools": 1,
        "tool_list": [{"name": "compute"}],
        "auth_scheme": "none",
        "local_runtime": {
            "type": "npx",
            "package": "@example/calc-mcp",
            "version": "1.4.2",
            "env": {"LOG_LEVEL": "info"},
            "required_env": ["API_KEY"],
        },
        "metadata": {},
    }


def _stored_local_command() -> dict:
    """Local server with command-type runtime; tests registryType -> 'mcpb'."""
    return {
        "id": "srv-4",
        "server_name": "Cmd Server",
        "path": "/cmd",
        "description": "Direct command launcher.",
        "version": "1.0.0",
        "deployment": "local",
        "supported_transports": ["stdio"],
        "local_runtime": {
            "type": "command",
            "package": "/usr/local/bin/my-mcp",
            "version": "1.0.0",
        },
        "metadata": {},
    }


def _stored_docker_local() -> dict:
    """Docker local runtime carrying image_digest and platforms."""
    return {
        "id": "srv-5",
        "server_name": "Docker Server",
        "path": "/docker",
        "description": "Containerized stdio server.",
        "version": "1.0.0",
        "deployment": "local",
        "supported_transports": ["stdio"],
        "local_runtime": {
            "type": "docker",
            "package": "example/mcp-image",
            "version": "1.0.0",
            "image_digest": "sha256:abc123",
            "platforms": ["linux/amd64", "linux/arm64"],
        },
        "metadata": {},
    }


@pytest.fixture(autouse=True)
def _reset_reverse_dns_cache():
    """Clear the lru_cache on _reverse_dns_base between tests so REGISTRY_URL changes apply."""
    _reverse_dns_base.cache_clear()
    yield
    _reverse_dns_base.cache_clear()


@pytest.fixture
def patch_registry_url(monkeypatch: pytest.MonkeyPatch):
    """Helper to set the registry_url used by namespace derivation."""

    def _set(url: str) -> None:
        monkeypatch.setattr(canonical_export.settings, "registry_url", url)
        _reverse_dns_base.cache_clear()

    return _set


# =============================================================================
# Reverse-DNS derivation
# =============================================================================


class TestReverseDnsBase:
    """Cover _reverse_dns_base derivation per LLD table."""

    def test_corp_domain_reverses(self):
        assert _reverse_dns_base("https://mcpgateway.mycorp.com") == "com.mycorp.mcpgateway"

    def test_localhost_with_port_strips_port(self):
        assert _reverse_dns_base("http://localhost:8000") == "localhost"

    def test_multi_label_with_region(self):
        assert (
            _reverse_dns_base("https://registry.us-east-1.mycorp.click")
            == "click.mycorp.us-east-1.registry"
        )

    def test_illegal_chars_sanitized(self):
        # underscore must be mapped to '-' since the canonical name pattern excludes _
        assert _reverse_dns_base("http://my_host.example.com") == "com.example.my-host"

    def test_missing_host_falls_back_to_localhost(self):
        # urlparse on a bare string yields no host
        assert _reverse_dns_base("not-a-url") == "localhost"


# =============================================================================
# Round-trip: spec preserved
# =============================================================================


class TestRoundtripPreservesCanonicalFields:
    """When metadata.mcp_registry_spec is present, canonical fields come through verbatim."""

    def test_schema_name_version_repository_remotes_preserved(self, patch_registry_url):
        patch_registry_url("https://mcpgateway.mycorp.com")
        stored = _stored_remote_with_spec()
        spec = stored["metadata"]["mcp_registry_spec"]

        out, truncated = to_canonical(stored)

        assert truncated is False
        assert out["$schema"] == spec["$schema"]
        assert out["name"] == spec["original_name"]
        assert out["version"] == "1.2.0"
        assert out["repository"] == spec["repository"]
        assert out["remotes"] == spec["remotes"]

    def test_preserved_meta_merged_with_internal(self, patch_registry_url):
        patch_registry_url("https://mcpgateway.mycorp.com")
        stored = _stored_remote_with_spec()
        spec = stored["metadata"]["mcp_registry_spec"]

        out, _ = to_canonical(stored)

        meta = out["_meta"]
        # preserved upstream namespace is present
        assert (
            meta["io.modelcontextprotocol.registry/publisher-provided"]
            == spec["_meta"]["io.modelcontextprotocol.registry/publisher-provided"]
        )
        # internal namespace appears alongside, derived from REGISTRY_URL
        internal_ns = "com.mycorp.mcpgateway/internal"
        assert internal_ns in meta
        internal = meta[internal_ns]
        assert internal["id"] == "srv-1"
        assert internal["server_name"] == "Calculator"
        assert internal["path"] == "/calculator"
        assert internal["tool_list"] == [{"name": "add"}]
        # extra metadata (non-spec keys) preserved under "metadata"
        assert internal["metadata"] == {"extra_field": "kept"}

    def test_description_under_cap_unchanged(self):
        stored = _stored_remote_with_spec()
        out, truncated = to_canonical(stored)

        assert truncated is False
        assert out["description"] == stored["description"]


# =============================================================================
# Synthesis: no spec
# =============================================================================


class TestSynthesizesValidDocNoSpec:
    """A bespoke document without mcp_registry_spec gets a schema-shape doc synthesized."""

    def test_required_top_level_fields_present(self, patch_registry_url):
        patch_registry_url("https://mcpgateway.mycorp.com")
        stored = _stored_remote_no_spec()

        out, _ = to_canonical(stored)

        # The three top-level required canonical fields must always appear.
        assert {"name", "description", "version"}.issubset(out.keys())
        assert out["$schema"] == DEFAULT_SCHEMA_URL
        assert out["version"] == "0.5.0"

    def test_internal_meta_namespace_used(self, patch_registry_url):
        patch_registry_url("https://mcpgateway.mycorp.com")
        stored = _stored_remote_no_spec()

        out, _ = to_canonical(stored)

        assert "com.mycorp.mcpgateway/internal" in out["_meta"]

    def test_version_default_when_absent(self, patch_registry_url):
        patch_registry_url("http://localhost:8000")
        stored = _stored_remote_no_spec()
        stored.pop("version")
        # also strip metadata so neither path has it
        stored["metadata"] = {}

        out, _ = to_canonical(stored)

        assert out["version"] == "0.0.0"


class TestRemoteSynthesis:
    """No spec, remote => remotes synthesized from proxy_pass_url + supported_transports."""

    def test_remote_url_and_type_from_stored(self, patch_registry_url):
        patch_registry_url("http://localhost:8000")
        stored = _stored_remote_no_spec()

        out, _ = to_canonical(stored)

        assert out["remotes"] == [
            {"type": "sse", "url": "https://legacy.example/mcp"},
        ]
        # remote-only servers must not synthesize a packages array
        assert "packages" not in out

    def test_default_transport_when_unspecified(self, patch_registry_url):
        patch_registry_url("http://localhost:8000")
        stored = _stored_remote_no_spec()
        stored.pop("supported_transports")

        out, _ = to_canonical(stored)

        assert out["remotes"][0]["type"] == "streamable-http"


class TestLocalSynthesis:
    """No spec, local stdio => packages[0] synthesized from local_runtime."""

    def test_stdio_transport_and_environment_variables(self, patch_registry_url):
        patch_registry_url("http://localhost:8000")
        stored = _stored_local_no_spec()

        out, _ = to_canonical(stored)

        assert "remotes" not in out
        pkg = out["packages"][0]
        assert pkg["registryType"] == "npm"
        assert pkg["identifier"] == "@example/calc-mcp"
        assert pkg["version"] == "1.4.2"
        assert pkg["transport"] == {"type": "stdio"}
        assert pkg["runtimeHint"] == "npx"

        env_by_name = {ev["name"]: ev for ev in pkg["environmentVariables"]}
        assert env_by_name["API_KEY"]["isRequired"] is True
        assert "default" not in env_by_name["API_KEY"]
        assert env_by_name["LOG_LEVEL"]["default"] == "info"

    def test_required_and_env_overlap_merges(self, patch_registry_url):
        """Validator at ingest forbids overlap, but legacy data may have it; merge into one entry."""
        patch_registry_url("http://localhost:8000")
        stored = _stored_local_no_spec()
        stored["local_runtime"]["env"] = {"API_KEY": "default-value"}
        stored["local_runtime"]["required_env"] = ["API_KEY"]

        out, _ = to_canonical(stored)

        env_vars = out["packages"][0]["environmentVariables"]
        api_entries = [ev for ev in env_vars if ev["name"] == "API_KEY"]
        assert len(api_entries) == 1
        assert api_entries[0]["isRequired"] is True
        assert api_entries[0]["default"] == "default-value"

    def test_docker_extras_stashed_in_internal_meta(self, patch_registry_url):
        patch_registry_url("https://mcpgateway.mycorp.com")
        stored = _stored_docker_local()

        out, _ = to_canonical(stored)

        internal = out["_meta"]["com.mycorp.mcpgateway/internal"]
        assert internal["image_digest"] == "sha256:abc123"
        assert internal["platforms"] == ["linux/amd64", "linux/arm64"]


class TestCommandLocalRegistryType:
    """`command`-type local servers must emit packages[0].registryType = 'mcpb'."""

    def test_command_maps_to_mcpb(self, patch_registry_url):
        patch_registry_url("http://localhost:8000")
        stored = _stored_local_command()

        out, _ = to_canonical(stored)

        assert "packages" in out
        pkg = out["packages"][0]
        assert pkg["registryType"] == "mcpb"
        assert pkg["runtimeHint"] == "command"
        # registryType must never be omitted for a package
        assert "registryType" in pkg


# =============================================================================
# Description truncation
# =============================================================================


class TestDescriptionTruncation:
    """Description over 100 chars is truncated; full text moves to _meta.<ns>.description_full."""

    def test_long_description_truncates_and_flags(self, patch_registry_url):
        patch_registry_url("https://mcpgateway.mycorp.com")
        stored = _stored_remote_no_spec()
        stored["description"] = "x" * 200

        out, truncated = to_canonical(stored)

        assert truncated is True
        assert len(out["description"]) == MAX_CANONICAL_DESCRIPTION
        assert out["description"].endswith("…")
        internal = out["_meta"]["com.mycorp.mcpgateway/internal"]
        assert internal["description_full"] == "x" * 200

    def test_exactly_100_chars_not_truncated(self, patch_registry_url):
        patch_registry_url("http://localhost:8000")
        stored = _stored_remote_no_spec()
        stored["description"] = "y" * 100

        out, truncated = to_canonical(stored)

        assert truncated is False
        assert out["description"] == "y" * 100
        internal = out["_meta"]["localhost/internal"]
        assert "description_full" not in internal

    def test_empty_description_handled(self, patch_registry_url):
        patch_registry_url("http://localhost:8000")
        stored = _stored_remote_no_spec()
        stored["description"] = ""

        out, truncated = to_canonical(stored)

        assert truncated is False
        assert out["description"] == ""


# =============================================================================
# Name synthesis
# =============================================================================


class TestNameSynthesisReverseDns:
    """If server_name isn't already reverse-DNS, synthesize <vendor>/<slug>."""

    def test_bare_name_becomes_vendor_slug(self, patch_registry_url):
        patch_registry_url("https://mcpgateway.mycorp.com")
        stored = _stored_remote_no_spec()
        # server_name is "Legacy Remote" (no slash), path is "/legacy-remote"
        out, _ = to_canonical(stored)

        assert out["name"] == "com.mycorp.mcpgateway/legacy-remote"

    def test_already_reverse_dns_preserved(self, patch_registry_url):
        patch_registry_url("https://mcpgateway.mycorp.com")
        stored = _stored_remote_no_spec()
        stored["server_name"] = "io.example/already-canonical"

        out, _ = to_canonical(stored)

        assert out["name"] == "io.example/already-canonical"

    def test_unknown_slug_when_no_path(self, patch_registry_url):
        patch_registry_url("http://localhost:8000")
        stored = _stored_remote_no_spec()
        stored.pop("path")
        stored["server_name"] = "anonymous"

        out, _ = to_canonical(stored)

        assert out["name"] == "localhost/unknown"

    def test_spec_original_name_wins_over_synthesis(self, patch_registry_url):
        patch_registry_url("https://mcpgateway.mycorp.com")
        stored = _stored_remote_with_spec()

        out, _ = to_canonical(stored)

        assert out["name"] == "io.example/calculator-mcp"


# =============================================================================
# Round-trip byte-equality (modulo key order) for canonical input fields
# =============================================================================


class TestRoundtripByteEquality:
    """Canonical fields are byte-for-byte identical modulo key ordering."""

    def test_canonical_fields_byte_identical_modulo_order(self, patch_registry_url):
        patch_registry_url("https://mcpgateway.mycorp.com")
        stored = _stored_remote_with_spec()
        spec = stored["metadata"]["mcp_registry_spec"]

        out, truncated = to_canonical(stored)

        assert truncated is False
        # Canonical input fields are returned exactly as preserved.
        assert out["$schema"] == spec["$schema"]
        assert out["name"] == spec["original_name"]
        assert out["version"] == spec["version"]
        assert out["repository"] == spec["repository"]
        assert out["remotes"] == spec["remotes"]

    def test_preserved_packages_passed_through_verbatim(self, patch_registry_url):
        """A local server registered from canonical input round-trips its packages array."""
        patch_registry_url("https://mcpgateway.mycorp.com")
        preserved_packages = [
            {
                "registryType": "npm",
                "identifier": "@example/calc-mcp",
                "version": "1.4.2",
                "transport": {"type": "stdio"},
                "runtimeHint": "npx",
                "environmentVariables": [
                    {"name": "API_KEY", "isRequired": True},
                ],
            }
        ]
        stored = _stored_local_no_spec()
        stored["metadata"] = {
            "mcp_registry_spec": {
                "original_name": "io.example/calc-mcp",
                "version": "1.4.2",
                "packages": preserved_packages,
            }
        }

        out, _ = to_canonical(stored)

        # Synthesis path must not run; preserved array passes through unchanged.
        assert out["packages"] == preserved_packages
        assert "remotes" not in out


def test_redaction_does_not_mutate_source_server():
    """Redaction works on a copy: source untouched, output stripped."""
    stored = {
        "path": "/test",
        "description": "test server",
        "metadata": {
            "mcp_registry_spec": {
                "remotes": [{"type": "streamable-http", "url": "http://backend:8000/mcp"}]
            }
        },
    }
    before = copy.deepcopy(stored)

    canonical, _ = to_canonical(stored)
    redacted = redact_backend_urls(canonical)  # capture the returned copy

    # source the caller passed in is untouched (no cache corruption)...
    assert stored == before, "redaction leaked back into the stored server"
    # ...and the returned doc actually has the backend url stripped
    assert "url" not in redacted["remotes"][0], "backend url was not redacted"
