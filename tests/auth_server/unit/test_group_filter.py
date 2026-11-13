"""Unit tests for auth_server.group_filter.

Covers the login-time IdP group filter:
- Design B (ALLOWED_IDP_GROUPS allowlist) takes precedence when set.
- Design C (auto-derive from scope mappings) is the default.
- Fail-open only when Design C cannot run (repo error / no mappings seeded).
- A successful-but-empty Design C result stores empty (genuinely no-access).
- Authorization is unchanged (scope-equivalence) when dropped groups are
  unmapped.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from auth_server import group_filter


def _make_scope_repo(mapped: set[str]) -> AsyncMock:
    """Build a scope-repo mock whose get_all_mapped_group_names returns mapped."""
    repo = AsyncMock()
    repo.get_all_mapped_group_names.return_value = mapped
    return repo


@pytest.fixture(autouse=True)
def _reset_mapped_group_names_cache():
    """Keep process-local mapping state isolated across test cases."""
    group_filter._clear_mapped_group_names_cache()
    yield
    group_filter._clear_mapped_group_names_cache()


class TestFilterSessionGroups:
    """Tests for filter_session_groups."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        """An empty group list short-circuits to empty."""
        result = await group_filter.filter_session_groups([], username_hash="h")
        assert result == []

    @pytest.mark.asyncio
    async def test_scope_derived_keeps_only_mapped(self):
        """Design C keeps only groups present in the scope-mapped set."""
        repo = _make_scope_repo({"registry-admins", "registry-readonly"})
        groups = ["registry-admins", "noise-1", "registry-readonly"] + [f"g{i}" for i in range(500)]
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                return_value=repo,
            ):
                result = await group_filter.filter_session_groups(groups, username_hash="h")
        assert set(result) == {"registry-admins", "registry-readonly"}

    @pytest.mark.asyncio
    async def test_allowlist_mode_overrides_scope_derived(self):
        """Design B (allowlist) takes precedence; scope repo is not consulted."""
        repo = _make_scope_repo({"should-not-be-used"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", ["keep-a", "keep-b"]):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                return_value=repo,
            ) as mock_get_repo:
                result = await group_filter.filter_session_groups(
                    ["keep-a", "drop-x", "keep-b"],
                    username_hash="h",
                )
        assert set(result) == {"keep-a", "keep-b"}
        mock_get_repo.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_mappings_fail_open(self):
        """No scope mappings seeded -> fail open (store full list)."""
        repo = _make_scope_repo(set())
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                return_value=repo,
            ):
                result = await group_filter.filter_session_groups(
                    ["a", "b", "c"],
                    username_hash="h",
                )
        assert result == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_repo_error_fail_open(self):
        """Repository error -> fail open (login must not break)."""
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                side_effect=RuntimeError("db down"),
            ):
                result = await group_filter.filter_session_groups(
                    ["a", "b"],
                    username_hash="h",
                )
        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_scope_derived_normalizes_leading_slash(self):
        """Design C: a full-group-path IdP claim matches an unslashed mapping.

        Keycloak's Group Membership mapper with "Full group path" enabled
        emits '/mcp-admins' instead of 'mcp-admins'; scope mappings are seeded
        without the leading slash, so the comparison must normalize both.
        """
        repo = _make_scope_repo({"registry-admins", "registry-readonly"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                return_value=repo,
            ):
                result = await group_filter.filter_session_groups(
                    ["/registry-admins", "some-other-group"],
                    username_hash="h",
                )
        assert result == ["registry-admins"]

    @pytest.mark.asyncio
    async def test_allowlist_mode_normalizes_leading_slash(self):
        """Design B: the allowlist comparison also normalizes leading slashes."""
        repo = _make_scope_repo({"should-not-be-used"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", ["keep-a", "keep-b"]):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                return_value=repo,
            ) as mock_get_repo:
                result = await group_filter.filter_session_groups(
                    ["/keep-a", "drop-x"],
                    username_hash="h",
                )
        assert result == ["keep-a"]
        mock_get_repo.assert_not_called()

    @pytest.mark.asyncio
    async def test_identifier_mismatch_stores_empty(self):
        """Mappings keyed by names but token emits GUIDs -> empty intersection.

        A successful Design C run with no overlap stores empty (no-access); it
        does NOT chain to the allowlist.
        """
        repo = _make_scope_repo({"registry-admins"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                return_value=repo,
            ):
                result = await group_filter.filter_session_groups(
                    ["7f3a-guid", "9b2c-guid"],
                    username_hash="h",
                )
        assert result == []


class TestScopeMappedGroupCache:
    """Successful mappings are cached briefly; errors never become cached policy."""

    @pytest.mark.asyncio
    async def test_concurrent_cold_misses_share_one_repository_read(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def read_mappings():
            started.set()
            await release.wait()
            return {"reader"}

        repo = _make_scope_repo(set())
        repo.get_all_mapped_group_names.side_effect = read_mappings
        with (
            patch.object(group_filter, "ALLOWED_IDP_GROUPS", []),
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
        ):
            first = asyncio.create_task(
                group_filter.filter_session_groups(["reader", "admin"], "a")
            )
            second = asyncio.create_task(
                group_filter.filter_session_groups(["admin", "reader"], "b")
            )
            await started.wait()
            await asyncio.sleep(0)
            assert repo.get_all_mapped_group_names.await_count == 1
            release.set()
            assert await asyncio.gather(first, second) == [["reader"], ["reader"]]
            assert await group_filter.filter_session_groups(["reader"], "c") == ["reader"]

        repo.get_all_mapped_group_names.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fails", [True, False], ids=["repository-error", "empty-mappings"])
    async def test_concurrent_unavailable_mappings_do_not_cache(self, fails):
        started = asyncio.Event()
        release = asyncio.Event()

        async def unavailable_mappings():
            started.set()
            await release.wait()
            if fails:
                raise RuntimeError("repository unavailable")
            return set()

        repo = _make_scope_repo(set())
        repo.get_all_mapped_group_names.side_effect = unavailable_mappings
        with (
            patch.object(group_filter, "ALLOWED_IDP_GROUPS", []),
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
        ):
            login = asyncio.create_task(group_filter.filter_session_groups(["reader"], "login"))
            bearer = asyncio.create_task(
                group_filter.filter_session_groups(["reader"], "bearer", fail_closed=True)
            )
            await started.wait()
            await asyncio.sleep(0)
            assert repo.get_all_mapped_group_names.await_count == 1
            release.set()
            assert await login == ["reader"]
            with pytest.raises(group_filter.ScopeGroupMappingsUnavailable):
                await bearer
            repo.get_all_mapped_group_names.side_effect = None
            repo.get_all_mapped_group_names.return_value = {"admin"}
            assert await group_filter.filter_session_groups(["reader", "admin"], "retry") == [
                "admin"
            ]

        assert repo.get_all_mapped_group_names.await_count == 2

    @pytest.mark.asyncio
    async def test_cancelled_waiter_does_not_cancel_shared_lookup(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def read_mappings():
            started.set()
            await release.wait()
            return {"reader"}

        repo = _make_scope_repo(set())
        repo.get_all_mapped_group_names.side_effect = read_mappings
        with (
            patch.object(group_filter, "ALLOWED_IDP_GROUPS", []),
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
        ):
            cancelled = asyncio.create_task(group_filter.filter_session_groups(["reader"], "a"))
            await started.wait()
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            surviving = asyncio.create_task(group_filter.filter_session_groups(["reader"], "b"))
            await asyncio.sleep(0)
            assert repo.get_all_mapped_group_names.await_count == 1
            release.set()
            assert await surviving == ["reader"]

        repo.get_all_mapped_group_names.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reset_detaches_old_lookup_without_overwriting_new_cache(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def read_mappings():
            if repo.get_all_mapped_group_names.await_count == 1:
                started.set()
                await release.wait()
                return {"old"}
            return {"new"}

        repo = _make_scope_repo(set())
        repo.get_all_mapped_group_names.side_effect = read_mappings
        with (
            patch.object(group_filter, "ALLOWED_IDP_GROUPS", []),
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
        ):
            old_waiter = asyncio.create_task(group_filter.filter_session_groups(["old"], "old"))
            await started.wait()
            group_filter._clear_mapped_group_names_cache()
            assert await group_filter.filter_session_groups(["new"], "new") == ["new"]
            release.set()
            assert await old_waiter == ["old"]
            assert await group_filter.filter_session_groups(["new", "old"], "cache") == ["new"]

        assert repo.get_all_mapped_group_names.await_count == 2

    @pytest.mark.asyncio
    async def test_reuses_mappings_without_rescanning_scopes(self):
        repo = _make_scope_repo({"reader"})
        with (
            patch.object(group_filter, "ALLOWED_IDP_GROUPS", []),
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
            patch.object(group_filter.time, "monotonic", return_value=100),
        ):
            first = await group_filter.filter_session_groups(["reader", "admin"], "first")
            second = await group_filter.filter_session_groups(["admin", "reader"], "second")

        assert first == ["reader"]
        assert second == ["reader"]
        repo.get_all_mapped_group_names.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_expiry_refreshes_changed_mappings(self):
        repo = _make_scope_repo({"reader"})
        with (
            patch.object(group_filter, "ALLOWED_IDP_GROUPS", []),
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
            patch.object(group_filter.time, "monotonic", return_value=100) as clock,
        ):
            assert await group_filter.filter_session_groups(["reader", "admin"], "h") == ["reader"]
            repo.get_all_mapped_group_names.return_value = {"admin"}
            clock.return_value = 100 + group_filter.MAPPED_GROUP_NAMES_CACHE_TTL_SECONDS
            refreshed = await group_filter.filter_session_groups(["reader", "admin"], "h")

        assert refreshed == ["admin"]
        assert repo.get_all_mapped_group_names.await_count == 2

    @pytest.mark.asyncio
    async def test_failed_refresh_is_not_cached_or_accepted_for_bearer(self):
        repo = _make_scope_repo({"reader"})
        with (
            patch.object(group_filter, "ALLOWED_IDP_GROUPS", []),
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
            patch.object(group_filter.time, "monotonic", return_value=100) as clock,
        ):
            assert await group_filter.filter_session_groups(["reader"], "h") == ["reader"]
            repo.get_all_mapped_group_names.side_effect = RuntimeError("db down")
            clock.return_value = 100 + group_filter.MAPPED_GROUP_NAMES_CACHE_TTL_SECONDS
            with pytest.raises(group_filter.ScopeGroupMappingsUnavailable):
                await group_filter.filter_session_groups(["reader", "admin"], "h", fail_closed=True)
            repo.get_all_mapped_group_names.side_effect = None
            repo.get_all_mapped_group_names.return_value = {"admin"}
            recovered = await group_filter.filter_session_groups(
                ["reader", "admin"], "h", fail_closed=True
            )

        assert recovered == ["admin"]
        assert repo.get_all_mapped_group_names.await_count == 3

    @pytest.mark.asyncio
    async def test_empty_mapping_state_is_not_cached_or_accepted_for_bearer(self):
        repo = _make_scope_repo(set())
        with (
            patch.object(group_filter, "ALLOWED_IDP_GROUPS", []),
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
            patch.object(group_filter.time, "monotonic", return_value=100),
        ):
            with pytest.raises(group_filter.ScopeGroupMappingsUnavailable):
                await group_filter.filter_session_groups(["admin"], "h", fail_closed=True)
            repo.get_all_mapped_group_names.return_value = {"reader"}
            recovered = await group_filter.filter_session_groups(["admin"], "h", fail_closed=True)

        assert recovered == []  # confirmed no overlap, not a failed lookup
        assert repo.get_all_mapped_group_names.await_count == 2

    @pytest.mark.asyncio
    async def test_explicit_invalidation_forces_refresh_before_ttl(self):
        mapped = {"reader"}
        repo = _make_scope_repo(mapped)
        with (
            patch.object(group_filter, "ALLOWED_IDP_GROUPS", []),
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
            patch.object(group_filter.time, "monotonic", return_value=100),
        ):
            assert await group_filter.filter_session_groups(["reader", "admin"], "h") == ["reader"]
            mapped.clear()
            mapped.add("admin")
            # Cached names are an immutable snapshot, not the repository's mutable set.
            assert await group_filter.filter_session_groups(["reader", "admin"], "h") == ["reader"]
            group_filter._clear_mapped_group_names_cache()
            refreshed = await group_filter.filter_session_groups(["reader", "admin"], "h")

        assert refreshed == ["admin"]
        assert repo.get_all_mapped_group_names.await_count == 2

    @pytest.mark.asyncio
    async def test_allowlist_still_takes_precedence_over_cached_mappings(self):
        repo = _make_scope_repo({"admin"})
        with (
            patch.object(group_filter, "ALLOWED_IDP_GROUPS", []),
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
            patch.object(group_filter.time, "monotonic", return_value=100),
        ):
            await group_filter.filter_session_groups(["admin"], "h")
            with patch.object(group_filter, "ALLOWED_IDP_GROUPS", ["reader"]):
                result = await group_filter.filter_session_groups(
                    ["admin", "reader"], "h", fail_closed=True
                )

        assert result == ["reader"]
        repo.get_all_mapped_group_names.assert_awaited_once()
