"""Unit tests for the ARD Registry adapter service (issue #1295).

Covers the pure helpers (filter mapping, pageToken, tag filter, score rescale,
ordering) and the access-scoping behavior of search_and_scope (restricted users
see a strict subset; the access-filtered count is reported).
"""

from unittest.mock import AsyncMock, patch

import pytest

from registry.services import ard_search_service as s
from registry.services.ard_search_service import ArdValidationError


class TestFilterToEngine:
    def test_none(self):
        assert s.filter_to_engine(None) == (None, None)

    def test_type_and_tags(self):
        assert s.filter_to_engine({"type": ["mcp_server"], "tags": ["finance"]}) == (
            ["mcp_server"],
            ["finance"],
        )

    def test_type_or_within_key(self):
        ent, _ = s.filter_to_engine({"type": ["server", "agent"]})
        assert ent == ["mcp_server", "a2a_agent"]

    def test_media_string_accepted(self):
        ent, _ = s.filter_to_engine({"type": "application/ai-skill"})
        assert ent == ["skill"]

    def test_unknown_key_raises(self):
        with pytest.raises(ArdValidationError):
            s.filter_to_engine({"bogus": "x"})

    def test_unknown_type_value_raises(self):
        with pytest.raises(ArdValidationError):
            s.filter_to_engine({"type": "not-a-type"})


class TestPageToken:
    def test_round_trip(self):
        assert s.decode_page_token(s.encode_page_token(40)) == 40

    def test_none_is_zero(self):
        assert s.decode_page_token(None) == 0

    def test_bad_token_raises(self):
        with pytest.raises(ArdValidationError):
            s.decode_page_token("!!not-base64!!")

    def test_negative_offset_raises(self):
        with pytest.raises(ArdValidationError):
            s.decode_page_token(s.encode_page_token(-5))


class TestTagFilter:
    def test_has_all_case_insensitive(self):
        assert s._has_all_tags(["Finance", "X"], ["finance"]) is True

    def test_missing_tag(self):
        assert s._has_all_tags(["x"], ["finance"]) is False

    def test_no_wanted_tags_passes(self):
        assert s._has_all_tags([], None) is True


class TestScoreRescale:
    def test_rescale_and_clamp(self):
        from registry.schemas.ard_models import ArdCatalogEntry

        entry = ArdCatalogEntry(
            identifier="urn:air:x:server:y", display_name="Y",
            type="application/mcp-server-card+json", url="http://x",
        )
        assert s._to_result(entry, 0.923, "http://src").score == 92
        assert s._to_result(entry, 0.0, "http://src").score == 0
        assert s._to_result(entry, 1.0, "http://src").score == 100
        assert s._to_result(entry, 2.0, "http://src").score == 100  # clamp


class TestParseFilterPairs:
    def test_repeated_key_accumulates(self):
        assert s._parse_filter_pairs(["type=server", "type=agent"]) == {
            "type": ["server", "agent"]
        }

    def test_single(self):
        assert s._parse_filter_pairs(["tags=finance"]) == {"tags": "finance"}

    def test_bad_pair_raises(self):
        with pytest.raises(ArdValidationError):
            s._parse_filter_pairs(["noequals"])


class TestSearchAndScopeAccessScoping:
    """search_and_scope must access-scope hits and report the filtered count."""

    async def test_restricted_user_sees_subset_and_counts_filtered(self):
        # Two server hits; the restricted user can access only one.
        raw = {
            "servers": [
                {"path": "/allowed/", "server_name": "Allowed", "tags": [],
                 "relevance_score": 0.9, "description": "a"},
                {"path": "/denied/", "server_name": "Denied", "tags": [],
                 "relevance_score": 0.8, "description": "b"},
            ],
            "agents": [],
            "skills": [],
        }
        mock_repo = AsyncMock()
        mock_repo.search = AsyncMock(return_value=raw)

        async def fake_access(path, name, ctx):
            return path == "/allowed/"

        with (
            patch.object(s, "get_search_repository", return_value=mock_repo),
            patch.object(s, "user_can_access_server", side_effect=fake_access),
            patch.object(s, "_resolve_publisher_domain", return_value="reg.example.com"),
        ):
            results, scoped_out = await s.search_and_scope(
                "q", None, None, 10, {"username": "u"}, "http://h/api/ard/search"
            )

        assert [r.display_name for r in results] == ["Allowed"]
        assert scoped_out == 1
        assert results[0].score == 90
        assert results[0].source == "http://h/api/ard/search"
        assert results[0].type == "application/mcp-server-card+json"

    async def test_results_ordered_by_score_desc(self):
        raw = {
            "servers": [
                {"path": "/low/", "server_name": "Low", "tags": [], "relevance_score": 0.2},
                {"path": "/high/", "server_name": "High", "tags": [], "relevance_score": 0.95},
            ],
            "agents": [],
            "skills": [],
        }
        mock_repo = AsyncMock()
        mock_repo.search = AsyncMock(return_value=raw)
        with (
            patch.object(s, "get_search_repository", return_value=mock_repo),
            patch.object(s, "user_can_access_server", AsyncMock(return_value=True)),
            patch.object(s, "_resolve_publisher_domain", return_value="reg.example.com"),
        ):
            results, _ = await s.search_and_scope(
                "q", None, None, 10, {}, "http://h/api/ard/search"
            )
        assert [r.display_name for r in results] == ["High", "Low"]
