"""Tests for the OpenAPI MCP server service (issue #2022).

The repositories are in-memory fakes; the container, the lifecycle steps and the
network are mocked. Each test checks one behavior the design depends on.
"""

import copy
from unittest.mock import (
    AsyncMock,
    patch,
)

import pytest

from registry.schemas.openapi_server_models import (
    OpenApiServer,
    OpenApiServerCreateRequest,
    OpenApiTargetCreateRequest,
    OpenApiTargetUpdateRequest,
    StoredSpec,
)
from registry.services import openapi_server_service as service
from registry.services import openapi_spec_checks
from registry.services.openapi_errors import (
    OpenApiServerError,
    container_unavailable,
    unprocessable,
)
from registry.utils.credential_encryption import decrypt_credential

ADMIN: dict = {"username": "admin", "is_admin": True}
SPEC_BYTES: bytes = b'{"openapi": "3.1.0", "info": {"title": "Items", "version": "1"}, "paths": {"/items": {"get": {"operationId": "listItems"}}}}'
TOOLS: list[str] = ["petstore_listItems"]


class FakeServerRepo:
    def __init__(self) -> None:
        self.docs: dict[str, OpenApiServer] = {}
        self.lose_next_replace = 0

    async def count(self) -> int:
        return len(self.docs)

    async def list_all(self):
        return [copy.deepcopy(s) for s in self.docs.values()]

    async def get(self, slug):
        return copy.deepcopy(self.docs.get(slug))

    async def create(self, server) -> bool:
        if server.slug in self.docs:
            return False
        self.docs[server.slug] = copy.deepcopy(server)
        return True

    async def replace(self, server, expected_generation) -> bool:
        if self.lose_next_replace:
            self.lose_next_replace -= 1
            return False
        stored = self.docs.get(server.slug)
        if stored is None or stored.generation != expected_generation:
            return False
        self.docs[server.slug] = copy.deepcopy(server)
        return True

    async def delete(self, slug) -> bool:
        return self.docs.pop(slug, None) is not None

    async def count_spec_references(self, sha256) -> int:
        return sum(1 for s in self.docs.values() for t in s.targets if t.spec_sha256 == sha256)

    async def count_targets(self) -> int:
        return sum(len(s.targets) for s in self.docs.values())


class FakeSpecRepo:
    def __init__(self) -> None:
        self.docs: dict[str, StoredSpec] = {}

    async def upsert(self, spec) -> None:
        self.docs[spec.sha256] = spec

    async def get_many(self, sha256s):
        return {sha: self.docs[sha] for sha in sha256s if sha in self.docs}

    async def delete(self, sha256) -> None:
        self.docs.pop(sha256, None)


@pytest.fixture
def env(monkeypatch):
    """Fakes for everything the service touches outside itself."""
    servers, specs = FakeServerRepo(), FakeSpecRepo()
    mocks = {
        "put_server": AsyncMock(return_value={"generation": 1, "targets": {}}),
        "delete_server": AsyncMock(),
        "preview": AsyncMock(return_value={"tools": TOOLS, "tags": {"items": 1}, "warnings": []}),
        "register": AsyncMock(),
        "refresh": AsyncMock(),
        "remove_everywhere": AsyncMock(return_value=True),
        "get_server_info": AsyncMock(return_value=None),
        "update_server": AsyncMock(return_value=True),
        "fetch": AsyncMock(return_value=(SPEC_BYTES, "application/json")),
    }
    monkeypatch.setattr(service, "get_openapi_server_repository", lambda: servers)
    monkeypatch.setattr(service, "get_openapi_spec_repository", lambda: specs)
    monkeypatch.setattr(service.openapi_mcp_client, "put_server", mocks["put_server"])
    monkeypatch.setattr(service.openapi_mcp_client, "delete_server", mocks["delete_server"])
    monkeypatch.setattr(service.openapi_mcp_client, "preview", mocks["preview"])
    monkeypatch.setattr(
        service.server_lifecycle, "register_and_enable_managed_server", mocks["register"]
    )
    monkeypatch.setattr(service.server_lifecycle, "refresh_managed_server", mocks["refresh"])
    monkeypatch.setattr(
        service.server_lifecycle, "remove_server_everywhere", mocks["remove_everywhere"]
    )
    monkeypatch.setattr(service.server_service, "get_server_info", mocks["get_server_info"])
    monkeypatch.setattr(service.server_service, "update_server", mocks["update_server"])
    monkeypatch.setattr(
        service, "openapi_mcp_target_url", lambda slug: f"http://openapi-mcp:9876/s/{slug}/mcp"
    )
    monkeypatch.setattr(openapi_spec_checks, "fetch_spec_bytes", mocks["fetch"])
    monkeypatch.setattr(openapi_spec_checks, "validate_target_url", lambda url, field: None)
    return servers, specs, mocks


def _target_request(
    **overrides,
) -> OpenApiTargetCreateRequest:
    body = {
        "name": "petstore",
        "spec_url": "https://petstore.example.com/openapi.json",
        "base_url": "https://petstore.example.com/api/v3",
        "auth": {
            "scheme": "api_key_header",
            "header_name": "api_key",
            "credential": "sekrit-key-1",
        },
    }
    body.update(overrides)
    return OpenApiTargetCreateRequest(**body)


async def _create(
    name: str = "Weather APIs",
) -> OpenApiServer:
    server, _status = await service.create_server(OpenApiServerCreateRequest(name=name), ADMIN, [])
    return server


class TestCreateServer:
    async def test_create_stores_pushes_and_registers(self, env):
        servers, _specs, mocks = env

        server = await _create()

        assert (server.slug, server.path, server.generation) == ("weather-apis", "/weather-apis", 1)
        entry = mocks["register"].call_args.args[0]
        assert entry["proxy_pass_url"] == "http://openapi-mcp:9876/s/weather-apis/mcp"
        assert entry["visibility"] == "private"
        assert "weather-apis" in servers.docs

    @pytest.mark.parametrize("name", ["API", "virtual", "s", "!!!"])
    async def test_reserved_or_empty_names_rejected(self, env, name):
        with pytest.raises(OpenApiServerError) as info:
            await _create(name)

        assert info.value.status_code == 400

    async def test_max_servers_returns_limit_error(self, env, monkeypatch):
        monkeypatch.setattr(service.settings, "openapi_mcp_max_servers", 1)
        await _create("one")

        with pytest.raises(OpenApiServerError) as info:
            await _create("two")

        assert (info.value.code, info.value.extra["limit"]) == ("limit_reached", 1)

    async def test_path_taken_by_ordinary_server_is_conflict(self, env):
        env[2]["get_server_info"].return_value = {"path": "/weather-apis"}

        with pytest.raises(OpenApiServerError) as info:
            await _create()

        assert info.value.code == "duplicate_name"

    async def test_create_server_rolls_back_when_container_down(self, env):
        servers, _specs, mocks = env
        mocks["put_server"].side_effect = container_unavailable("down")

        with pytest.raises(OpenApiServerError):
            await _create()

        assert servers.docs == {}
        mocks["delete_server"].assert_awaited_once_with("weather-apis")
        mocks["register"].assert_not_awaited()

    async def test_rollback_removes_serverinfo_when_registration_half_done(self, env):
        servers, _specs, mocks = env
        mocks["register"].side_effect = OpenApiServerError(500, "enable_failed", "x")
        mocks["get_server_info"].side_effect = [None, {"path": "/weather-apis"}]

        with pytest.raises(OpenApiServerError):
            await _create()

        mocks["remove_everywhere"].assert_awaited_once()
        assert servers.docs == {}

    async def test_rollback_is_safe_to_run_twice(self, env):
        servers, _specs, mocks = env
        server = OpenApiServer(
            slug="ghost",
            name="ghost",
            path="/ghost",
            created_by="admin",
            created_at=openapi_spec_checks.now_utc(),
            updated_at=openapi_spec_checks.now_utc(),
        )
        mocks["delete_server"].side_effect = [None, container_unavailable("down")]

        await service._rollback_create(server, "admin")
        await service._rollback_create(server, "admin")

        assert servers.docs == {}


class TestAddTarget:
    async def test_add_target_stores_encrypted_credential(self, env):
        servers, specs, mocks = env
        await _create()

        updated, target, _warnings, status = await service.add_target(
            "weather-apis", _target_request(), ADMIN
        )

        stored = servers.docs["weather-apis"].targets[0]
        assert stored.auth.credential_encrypted and "sekrit" not in stored.auth.credential_encrypted
        assert decrypt_credential(stored.auth.credential_encrypted) == "sekrit-key-1"
        assert (updated.generation, target.tool_names, status) == (2, TOOLS, "applied")
        assert stored.spec_sha256 in specs.docs

    async def test_push_carries_decrypted_credential_and_parsed_spec(self, env):
        _servers, _specs, mocks = env
        await _create()

        await service.add_target("weather-apis", _target_request(), ADMIN)

        state = mocks["put_server"].call_args.args[0]
        assert state["generation"] == 2
        assert state["targets"][0]["auth_credential"] == "sekrit-key-1"
        assert state["targets"][0]["spec"]["info"]["title"] == "Items"

    async def test_add_target_stores_nothing_when_preview_fails(self, env):
        servers, specs, mocks = env
        await _create()
        mocks["preview"].side_effect = unprocessable("conversion_failed", "bad")

        with pytest.raises(OpenApiServerError):
            await service.add_target("weather-apis", _target_request(), ADMIN)

        assert servers.docs["weather-apis"].targets == []
        assert servers.docs["weather-apis"].generation == 1
        assert specs.docs == {}

    async def test_too_many_tools_stores_nothing(self, env, monkeypatch):
        servers, specs, mocks = env
        await _create()
        monkeypatch.setattr(service.settings, "openapi_mcp_max_tools_per_target", 1)
        mocks["preview"].return_value = {"tools": ["a_x", "a_y"], "tags": {"t": 2}, "warnings": []}

        with pytest.raises(OpenApiServerError) as info:
            await service.add_target("weather-apis", _target_request(), ADMIN)

        assert info.value.code == "too_many_tools"
        assert specs.docs == {}

    async def test_duplicate_target_name_is_conflict(self, env):
        await _create()
        await service.add_target("weather-apis", _target_request(), ADMIN)

        with pytest.raises(OpenApiServerError) as info:
            await service.add_target("weather-apis", _target_request(), ADMIN)

        assert info.value.code == "duplicate_name"

    async def test_optimistic_save_retries_after_losing_a_race(self, env):
        servers, _specs, _mocks = env
        await _create()
        servers.lose_next_replace = 2

        updated, *_rest = await service.add_target("weather-apis", _target_request(), ADMIN)

        assert updated.generation == 2

    async def test_save_gives_up_after_retries(self, env):
        servers, _specs, _mocks = env
        await _create()
        servers.lose_next_replace = 5

        with pytest.raises(OpenApiServerError) as info:
            await service.add_target("weather-apis", _target_request(), ADMIN)

        assert info.value.code == "concurrent_update"

    async def test_container_failure_after_store_reports_pending(self, env):
        _servers, _specs, mocks = env
        await _create()
        mocks["put_server"].side_effect = container_unavailable("down")

        *_rest, status = await service.add_target("weather-apis", _target_request(), ADMIN)

        assert status == "pending"


class TestUpdateAndDelete:
    async def test_rotate_credential(self, env):
        servers, _specs, _mocks = env
        await _create()
        await service.add_target("weather-apis", _target_request(), ADMIN)
        rotate = OpenApiTargetUpdateRequest(auth={"scheme": "bearer", "credential": "new-token"})

        await service.update_target("weather-apis", "petstore", rotate, ADMIN)

        auth = servers.docs["weather-apis"].targets[0].auth
        assert (auth.scheme, decrypt_credential(auth.credential_encrypted)) == (
            "bearer",
            "new-token",
        )

    async def test_update_without_auth_keeps_credential(self, env):
        servers, _specs, _mocks = env
        await _create()
        await service.add_target("weather-apis", _target_request(), ADMIN)
        before = servers.docs["weather-apis"].targets[0].auth.credential_encrypted

        await service.update_target(
            "weather-apis", "petstore", OpenApiTargetUpdateRequest(description="d"), ADMIN
        )

        assert servers.docs["weather-apis"].targets[0].auth.credential_encrypted == before

    async def test_delete_target_removes_unused_spec(self, env):
        servers, specs, _mocks = env
        await _create()
        await service.add_target("weather-apis", _target_request(), ADMIN)

        await service.delete_target("weather-apis", "petstore", ADMIN)

        assert servers.docs["weather-apis"].targets == []
        assert specs.docs == {}

    async def test_shared_spec_survives_one_target_delete(self, env):
        _servers, specs, _mocks = env
        await _create()
        await service.add_target("weather-apis", _target_request(), ADMIN)
        await service.add_target("weather-apis", _target_request(name="second"), ADMIN)

        await service.delete_target("weather-apis", "petstore", ADMIN)

        assert len(specs.docs) == 1

    async def test_delete_server_cleans_everything(self, env):
        servers, specs, mocks = env
        await _create()
        await service.add_target("weather-apis", _target_request(), ADMIN)
        mocks["get_server_info"].return_value = {"path": "/weather-apis"}

        await service.delete_server("weather-apis", ADMIN)

        mocks["remove_everywhere"].assert_awaited_once()
        assert servers.docs == {} and specs.docs == {}

    async def test_unknown_server_is_404(self, env):
        with pytest.raises(OpenApiServerError) as info:
            await service.delete_server("nope", ADMIN)

        assert info.value.status_code == 404


UPLOADED_YAML: str = """openapi: 3.0.3
info: {title: Uploaded, version: "2"}
paths:
  /things:
    get: {operationId: listThings}
"""


def _upload_request(
    **overrides,
) -> OpenApiTargetCreateRequest:
    body = {
        "name": "petstore",
        "spec_content": UPLOADED_YAML,
        "base_url": "https://petstore.example.com/api/v3",
    }
    body.update(overrides)
    return OpenApiTargetCreateRequest(**body)


class TestUploadedSpec:
    """A spec sent in spec_content skips only the fetch; every other check still runs."""

    async def test_upload_stores_spec_without_fetching(self, env):
        servers, specs, mocks = env
        await _create()

        _updated, target, _warnings, _status = await service.add_target(
            "weather-apis", _upload_request(), ADMIN
        )

        stored = servers.docs["weather-apis"].targets[0]
        assert (stored.spec_source, stored.spec_url, stored.spec_title) == (
            "upload",
            None,
            "Uploaded",
        )
        assert specs.docs[stored.spec_sha256].content == UPLOADED_YAML
        mocks["fetch"].assert_not_awaited()
        assert mocks["preview"].call_args.args[0]["spec"]["info"]["title"] == "Uploaded"

    @pytest.mark.parametrize(
        ("content", "code"),
        [
            ('{"swagger": "2.0", "paths": {"/a": {}}}', "spec_invalid"),
            (
                "openapi: 3.0.0\npaths: {/a: {get: {}}}\nx: {$ref: 'http://169.254.169.254/'}\n",
                "spec_invalid",
            ),
            ("   ", "spec_invalid"),
        ],
    )
    async def test_upload_runs_the_same_spec_checks(self, env, content, code):
        servers, specs, _mocks = env
        await _create()

        with pytest.raises(OpenApiServerError) as info:
            await service.add_target("weather-apis", _upload_request(spec_content=content), ADMIN)

        assert info.value.code == code
        assert servers.docs["weather-apis"].targets == [] and specs.docs == {}

    async def test_upload_over_byte_cap_is_spec_too_large(self, env, monkeypatch):
        await _create()
        monkeypatch.setattr(service.settings, "openapi_mcp_spec_max_bytes", 64)

        with pytest.raises(OpenApiServerError) as info:
            await service.add_target("weather-apis", _upload_request(), ADMIN)

        assert info.value.code == "spec_too_large"

    async def test_upload_still_checks_tool_count(self, env, monkeypatch):
        _servers, _specs, mocks = env
        await _create()
        mocks["preview"].return_value = {"tools": ["a", "b", "c"], "tags": {}, "warnings": []}
        monkeypatch.setattr(service.settings, "openapi_mcp_max_tools_per_target", 2)

        with pytest.raises(OpenApiServerError) as info:
            await service.add_target("weather-apis", _upload_request(), ADMIN)

        assert info.value.code == "too_many_tools"

    async def test_patch_replaces_spec_and_drops_old_one(self, env):
        servers, specs, _mocks = env
        await _create()
        await service.add_target("weather-apis", _target_request(), ADMIN)
        old_sha = servers.docs["weather-apis"].targets[0].spec_sha256

        await service.update_target(
            "weather-apis",
            "petstore",
            OpenApiTargetUpdateRequest(spec_content=UPLOADED_YAML),
            ADMIN,
        )

        stored = servers.docs["weather-apis"].targets[0]
        assert (stored.spec_source, stored.spec_url) == ("upload", None)
        assert old_sha not in specs.docs and stored.spec_sha256 in specs.docs

    async def test_refresh_of_uploaded_spec_is_conflict(self, env):
        _servers, _specs, mocks = env
        await _create()
        await service.add_target("weather-apis", _upload_request(), ADMIN)

        with pytest.raises(OpenApiServerError) as info:
            await service.update_target(
                "weather-apis", "petstore", OpenApiTargetUpdateRequest(refresh_spec=True), ADMIN
            )

        assert (info.value.status_code, info.value.code) == (409, "spec_url_missing")
        mocks["fetch"].assert_not_awaited()

    async def test_patch_spec_url_switches_upload_back_to_fetching(self, env):
        servers, _specs, mocks = env
        await _create()
        await service.add_target("weather-apis", _upload_request(), ADMIN)

        await service.update_target(
            "weather-apis",
            "petstore",
            OpenApiTargetUpdateRequest(spec_url="https://petstore.example.com/v2.json"),
            ADMIN,
        )

        stored = servers.docs["weather-apis"].targets[0]
        assert (stored.spec_source, stored.spec_url) == (
            "url",
            "https://petstore.example.com/v2.json",
        )
        mocks["fetch"].assert_awaited_once()


class TestSpecSourceValidation:
    def test_create_needs_exactly_one_source(self):
        with pytest.raises(ValueError, match="exactly one"):
            _target_request(spec_content=UPLOADED_YAML)
        with pytest.raises(ValueError, match="exactly one"):
            OpenApiTargetCreateRequest(name="petstore", base_url="https://a.example.com")

    def test_update_takes_at_most_one_spec_change(self):
        with pytest.raises(ValueError, match="at most one"):
            OpenApiTargetUpdateRequest(spec_content=UPLOADED_YAML, refresh_spec=True)

    def test_stored_target_without_source_reads_as_url(self):
        assert service.OpenApiTarget.model_fields["spec_source"].default == "url"


async def test_push_and_refresh_reports_pending_past_budget(env, monkeypatch):
    import asyncio

    _servers, _specs, mocks = env
    server = await _create()

    async def _slow(*args, **kwargs):
        await asyncio.sleep(1)

    mocks["refresh"].side_effect = _slow
    monkeypatch.setattr(service, "REQUEST_WAIT_BUDGET_SECONDS", 0.05)

    with patch.object(service.logger, "warning"):
        status = await service._push_and_refresh(server, rescan=False)

    assert status == "pending"
