"""Unit tests for the observed auth-path mix in telemetry payloads (issue #1753).

Three things these pin, in descending order of consequence:

1. The mix is reported as *shares*, never as request counts. That is a privacy
   decision, so a raw total must not be reachable from the payload.
2. An empty mix must not read as a measured one -- all three fields are None
   together, never {} and never 0.
3. The numbers must not be wrong in a plausible-looking way: shares summing to
   99, a 40-minute window reported as a day, or a window past the collector's
   `le=48` bound, which would make Pydantic reject the whole heartbeat.
"""

import contextlib
import json
import random
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.core.telemetry import (
    _AUTH_PATH_WINDOW_HOURS_MAX,
    _KNOWN_AUTH_PATHS,
    _auth_path_fields,
    _auth_path_volume_bucket,
    _build_heartbeat_payload,
    _build_startup_payload,
)

AUTH_PATH_KEYS = (
    "auth_path_share_24h",
    "auth_path_volume_bucket_24h",
    "auth_path_window_hours",
)


@contextlib.contextmanager
def _payload_mocks(auth_path_counts=None, window_hours=0.0, search_counts=None):
    """Patch everything the payload builders reach outside this module.

    Cloud resolution and the registry id are pinned so no probe, no filesystem
    read and no database read can make a payload assertion flaky.
    """
    search_counts = search_counts or {"total": 7, "last_24h": 5, "last_1h": 1}
    with contextlib.ExitStack() as stack:
        enter = stack.enter_context
        mock_settings = enter(patch("registry.core.telemetry.settings"))
        enter(
            patch(
                "registry.core.telemetry._resolve_cloud",
                new_callable=AsyncMock,
                return_value=("unknown", "unknown"),
            )
        )
        enter(
            patch(
                "registry.core.telemetry._get_registry_id",
                new_callable=AsyncMock,
                return_value="11111111-2222-3333-4444-555555555555",
            )
        )
        enter(
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value=search_counts,
            )
        )
        enter(
            patch(
                "registry.repositories.stats_repository.get_auth_path_counts",
                new_callable=AsyncMock,
                return_value={
                    "counts": dict(auth_path_counts or {}),
                    "window_hours": window_hours,
                },
            )
        )
        enter(
            patch(
                "registry.api.system_routes.get_server_start_time",
                return_value=datetime.now(UTC),
            )
        )
        for factory in (
            "get_server_repository",
            "get_agent_repository",
            "get_skill_repository",
            "get_peer_federation_repository",
        ):
            repo = MagicMock()
            repo.list_all = AsyncMock(return_value=[])
            repo.list_peers = AsyncMock(return_value=[])
            enter(patch(f"registry.repositories.factory.{factory}", return_value=repo))

        mock_settings.deployment_mode.value = "with-gateway"
        mock_settings.registry_mode.value = "full"
        mock_settings.storage_backend = "documentdb"
        mock_settings.auth_provider = "keycloak"
        mock_settings.federation_static_token_auth_enabled = False
        mock_settings.embeddings_provider = "sentence-transformers"
        mock_settings.embeddings_model_name = "all-MiniLM-L6-v2"
        mock_settings.internal_only_deployment = False
        mock_settings.internal_deployment_type.value = "none"
        yield mock_settings


class TestAuthPathShares:
    """Tests for the share computation in _auth_path_fields()."""

    def test_shares_sum_to_exactly_100(self):
        """Largest-remainder rounding must never report a mix adding up to 99.

        Fuzzed with a seeded RNG: the rounding leftover depends on how the
        fractional parts fall, so a single hand-picked input proves nothing.
        """
        rng = random.Random(1753)
        paths = sorted(_KNOWN_AUTH_PATHS)

        for _ in range(2000):
            chosen = rng.sample(paths, rng.randint(1, len(paths)))
            counts = {path: rng.randint(1, 1_000_000) for path in chosen}
            shares = _auth_path_fields(counts, 24.0)["auth_path_share_24h"]

            assert sum(shares.values()) == 100, (counts, shares)
            assert set(shares) == set(counts), (counts, shares)
            assert all(isinstance(v, int) and v >= 0 for v in shares.values()), shares

    def test_shares_reflect_proportions(self):
        """A known mix maps to the known percentages."""
        fields = _auth_path_fields({"session_cookie": 620, "keycloak": 310, "unknown": 70}, 24.0)

        assert fields["auth_path_share_24h"] == {
            "session_cookie": 62,
            "keycloak": 31,
            "unknown": 7,
        }

    def test_shares_are_deterministic_for_tied_remainders(self):
        """Three equal paths give 34/33/33, and the same input twice agrees.

        Every fractional part ties here, so only a total ordering on the
        tie-break makes the output reproducible. A fleet view that reports a
        different path as the 34 on each heartbeat is not comparable.
        """
        counts = {"session_cookie": 1, "keycloak": 1, "unknown": 1}

        first = _auth_path_fields(counts, 24.0)["auth_path_share_24h"]
        second = _auth_path_fields(counts, 24.0)["auth_path_share_24h"]

        assert sum(first.values()) == 100
        assert sorted(first.values()) == [33, 33, 34]
        assert first == second

    def test_zero_volume_returns_none_for_all_three_fields(self):
        """No traffic reports None, not {} and not 0.

        An empty dict or a zero would read downstream as a measured mix of
        nothing, which is a different claim from "nothing was measured".
        """
        for counts in ({}, {"session_cookie": 0}, {"session_cookie": 0, "keycloak": 0}):
            fields = _auth_path_fields(counts, 12.5)

            assert set(fields) == set(AUTH_PATH_KEYS), fields
            for key in AUTH_PATH_KEYS:
                assert fields[key] is None, (counts, key, fields[key])

    def test_unknown_paths_excluded_from_shares(self):
        """A key outside the closed set is dropped, denominator included.

        The share dict must not become a free-text channel, and leaving a
        stray key in the denominator would silently halve every real share.
        """
        fields = _auth_path_fields({"session_cookie": 50, "wat": 50}, 24.0)

        assert fields["auth_path_share_24h"] == {"session_cookie": 100}
        # 50 in the denominator, not 100: the dropped key contributes nothing.
        assert fields["auth_path_volume_bucket_24h"] == "10-99"

    def test_window_hours_from_reset_timestamp(self):
        """A 40-minute window reports 0 hours, not 24.

        The daily window is reset lazily on write, so the heartbeat's phase
        against it is arbitrary. A reader has to be able to discard a sample
        too short to mean anything.
        """
        assert _auth_path_fields({"session_cookie": 3}, 40 / 60)["auth_path_window_hours"] == 0
        assert _auth_path_fields({"session_cookie": 3}, 21.9)["auth_path_window_hours"] == 21

    def test_window_hours_clamped_to_48(self):
        """A 200-hour window reports 48.

        The collector bounds the field at le=48, so an unclamped value makes
        Pydantic reject the entire heartbeat -- fail-closed behaviour inside a
        fail-open path. A coarse "at least 48" beats a dropped payload.
        """
        fields = _auth_path_fields({"session_cookie": 3}, 200.0)

        assert fields["auth_path_window_hours"] == _AUTH_PATH_WINDOW_HOURS_MAX == 48

    def test_negative_window_hours_floored_at_zero(self):
        """Clock skew must not emit a negative hour count the collector rejects."""
        assert _auth_path_fields({"session_cookie": 3}, -5.0)["auth_path_window_hours"] == 0


class TestAuthPathVolumeBucket:
    """Tests for the order-of-magnitude denominator bucket."""

    @pytest.mark.parametrize(
        "total,expected",
        [
            (1, "1-9"),
            (9, "1-9"),
            (10, "10-99"),
            (99, "10-99"),
            (100, "100-999"),
            (999, "100-999"),
            (1000, "1k-9k"),
            (9999, "1k-9k"),
            (10000, "10k+"),
            (1_000_000, "10k+"),
        ],
    )
    def test_volume_bucket_boundaries(self, total, expected):
        """Every boundary maps to its exact bucket string.

        The strings are matched by a regex in the collector, so an off-by-one
        or a renamed bucket is a rejected heartbeat, not a mislabelled one.
        """
        assert _auth_path_volume_bucket(total) == expected
        fields = _auth_path_fields({"session_cookie": total}, 24.0)
        assert fields["auth_path_volume_bucket_24h"] == expected

    def test_zero_volume_bucket_unreachable(self):
        """The "0" bucket exists in the collector regex but must never be sent.

        Zero volume reports None for all three fields, so nothing can emit
        "0" -- which would read as a measured window that saw no traffic.
        """
        assert _auth_path_fields({}, 24.0)["auth_path_volume_bucket_24h"] is None
        assert _auth_path_fields({"session_cookie": 0}, 24.0)["auth_path_volume_bucket_24h"] is None
        for total in (1, 2, 9, 10, 100, 1000, 10000):
            assert _auth_path_volume_bucket(total) != "0"


class TestAuthPathPayloadPlacement:
    """Tests for where the auth-path fields appear, and what they carry."""

    @pytest.mark.asyncio
    async def test_heartbeat_carries_auth_path_fields(self):
        """The heartbeat carries the mix, the bucket and the window."""
        with _payload_mocks(
            auth_path_counts={"session_cookie": 620, "keycloak": 310, "unknown": 70},
            window_hours=21.4,
        ):
            payload = await _build_heartbeat_payload()

        assert payload["auth_path_share_24h"] == {
            "session_cookie": 62,
            "keycloak": 31,
            "unknown": 7,
        }
        assert payload["auth_path_volume_bucket_24h"] == "1k-9k"
        assert payload["auth_path_window_hours"] == 21

    @pytest.mark.asyncio
    async def test_heartbeat_sends_nulls_when_window_saw_no_traffic(self):
        """No traffic sends all three keys as None, and still sends them."""
        with _payload_mocks(auth_path_counts={}, window_hours=3.0):
            payload = await _build_heartbeat_payload()

        for key in AUTH_PATH_KEYS:
            assert key in payload
            assert payload[key] is None, (key, payload[key])

    @pytest.mark.asyncio
    async def test_startup_payload_has_no_auth_path_fields(self):
        """Windowed observations belong to the heartbeat; a fresh process has none."""
        with _payload_mocks(auth_path_counts={"session_cookie": 10}, window_hours=24.0):
            payload = await _build_startup_payload()

        for key in AUTH_PATH_KEYS:
            assert key not in payload, key

    @pytest.mark.asyncio
    async def test_schema_version_is_6_in_both_builders(self):
        """Both builders say "6", and they say the same thing.

        The two values are separate string literals with no shared constant, so
        bumping one and forgetting the other is the actual failure mode.
        """
        with _payload_mocks():
            startup = await _build_startup_payload()
            heartbeat = await _build_heartbeat_payload()

        assert startup["schema_version"] == heartbeat["schema_version"]
        assert heartbeat["schema_version"] == "6"

    @pytest.mark.asyncio
    async def test_heartbeat_carries_no_raw_request_volume(self):
        """Shares, not counts: no per-path count and no total may be recoverable.

        Pinned so a later change cannot quietly turn a mix report into a
        traffic meter bound to a stable registry_id.
        """
        counts = {"session_cookie": 3000, "keycloak": 1242}
        total = sum(counts.values())

        with _payload_mocks(auth_path_counts=counts, window_hours=24.0):
            payload = await _build_heartbeat_payload()

        forbidden = {str(total), *(str(n) for n in counts.values())}
        rendered = json.dumps(payload)
        for value in forbidden:
            assert value not in rendered, f"{value} recoverable from {rendered}"
        assert total not in [v for v in payload.values() if isinstance(v, int)]
