"""
Unit tests for the audit read path over durable identity claims.

Covers CSV export column coverage, the broadened username filter, and the
regression guard that keeps dashboard grouping on a single display identity.

Validates: Issue #1642
"""

import csv
import io
import re
from unittest.mock import AsyncMock, MagicMock, patch

from registry.audit.routes import (
    _build_query,
    _count_distinct_usernames,
    _generate_csv,
    _generate_jsonl,
    get_statistics,
)

# The six claim columns, in the order the CSV must emit them.
CLAIM_COLUMNS = [
    "principal_name",
    "canonical_id",
    "subject",
    "object_id",
    "tenant_id",
    "app_id",
]

# Fields the username filter searches, without the stream-specific prefix, split
# by HOW they match: readable identities as a case-insensitive regex, opaque
# identifiers by equality. Equality is the only form an index can serve, though
# no claim index exists yet -- see _identity_search_clause.
READABLE_FIELDS = ["username", "principal_name"]
OPAQUE_FIELDS = ["subject", "canonical_id", "object_id"]
SEARCHED_FIELDS = READABLE_FIELDS + OPAQUE_FIELDS


def _expected_clause(prefix: str, value: str, anchored: bool = False) -> list[dict]:
    """The $or the read path must build for ``value``."""
    escaped = re.escape(value)
    readable = {"$regex": f"^{escaped}$" if anchored else escaped, "$options": "i"}
    lowered = value.lower()
    opaque = value if lowered == value else {"$in": [value, lowered]}
    return [
        *({f"{prefix}{field}": readable} for field in READABLE_FIELDS),
        *({f"{prefix}{field}": opaque} for field in OPAQUE_FIELDS),
    ]


CLAIMS = {
    "subject": "sub-opaque-123",
    "canonical_id": "oid-abc@tid-xyz",
    "principal_name": "alice@contoso.com",
    "object_id": "oid-abc",
    "tenant_id": "tid-xyz",
    "app_id": "app-789",
}


def _claim_query(**overrides):
    """Call _build_query on a stream that actually carries the claim fields.

    ``registry_api`` never populates them (see _identity_search_clause), so its
    filter is the display username alone and is not the stream to assert claim
    matching against.
    """
    kwargs = {"stream": "mcp_access"}
    kwargs.update(overrides)
    return _query(**kwargs)


def _query(**overrides):
    """Call _build_query with registry_api defaults, overriding as needed."""
    kwargs = {
        "stream": "registry_api",
        "from_time": None,
        "to_time": None,
        "username": None,
        "operation": None,
        "resource_type": None,
        "resource_id": None,
        "status_min": None,
        "status_max": None,
        "auth_decision": None,
    }
    kwargs.update(overrides)
    return _build_query(**kwargs)


def _csv_rows(events):
    """Render events through _generate_csv and parse them back."""
    content = "".join(_generate_csv(events))
    return list(csv.DictReader(io.StringIO(content)))


def _header(events):
    """Return the CSV header field order for events."""
    content = "".join(_generate_csv(events))
    return next(csv.reader(io.StringIO(content)))


# =============================================================================
# CSV export: claim columns
# =============================================================================


class TestCsvClaimColumns:
    """The compliance export must carry every stored identity claim."""

    def test_claim_columns_follow_username(self):
        """Claim columns are emitted directly after username, in order."""
        header = _header([{"request_id": "req-1"}])
        start = header.index("username") + 1
        assert header[start : start + len(CLAIM_COLUMNS)] == CLAIM_COLUMNS

    def test_populates_claims_from_identity_block(self):
        """registry_api/mcp_access records expose claims under `identity`."""
        rows = _csv_rows(
            [
                {
                    "request_id": "req-1",
                    "log_type": "registry_api_access",
                    "identity": {"username": "alice", **CLAIMS},
                }
            ]
        )

        assert rows[0]["username"] == "alice"
        for column, value in CLAIMS.items():
            assert rows[0][column] == value

    def test_populates_claims_from_token_mint_top_level(self):
        """token_mint records have no identity block: claims are top level."""
        rows = _csv_rows(
            [
                {
                    "request_id": "req-2",
                    "log_type": "token_mint",
                    "username": "alice@contoso.com",
                    **CLAIMS,
                }
            ]
        )

        assert rows[0]["username"] == "alice@contoso.com"
        for column, value in CLAIMS.items():
            assert rows[0][column] == value

    def test_absent_claims_render_empty(self):
        """IdPs that omit the claims store None; the export writes "" not None."""
        rows = _csv_rows(
            [
                {
                    "request_id": "req-3",
                    "identity": {
                        "username": "bob",
                        "subject": None,
                        "canonical_id": None,
                    },
                }
            ]
        )

        for column in CLAIM_COLUMNS:
            assert rows[0][column] == ""

    def test_jsonl_export_carries_claims_verbatim(self):
        """JSONL dumps the whole record, so claims need no explicit mapping."""
        event = {"request_id": "req-4", "identity": {"username": "alice", **CLAIMS}}
        line = next(iter(_generate_jsonl([event])))

        for value in CLAIMS.values():
            assert value in line


class TestCsvFormulaNeutralization:
    """The compliance export is the artifact most likely to be opened in a
    spreadsheet, and an audit record stores caller-authored strings: the request
    `path` is any URI a client sent, including one that never routed. A cell
    starting with =, +, -, @, tab or CR is evaluated as a formula by Excel,
    LibreOffice and Sheets, so an unauthenticated request to `/=HYPERLINK(...)`
    would otherwise become live content in an admin's spreadsheet.
    """

    def _path_cell(self, path: str) -> str:
        rows = _csv_rows(
            [{"request_id": "r", "identity": {"username": "a"}, "request": {"path": path}}]
        )
        return rows[0]["path"]

    def test_formula_path_is_quoted_as_text(self):
        cell = self._path_cell('=HYPERLINK("http://attacker/","click")')

        assert cell.startswith("'="), cell

    def test_every_dangerous_prefix_is_neutralized(self):
        for prefix in ("=", "+", "-", "@", "\t", "\r"):
            cell = self._path_cell(f"{prefix}cmd")

            assert cell.startswith(f"'{prefix}"), (prefix, cell)

    def test_ordinary_values_are_untouched(self):
        """Neutralization must not rewrite normal exports."""
        assert self._path_cell("/api/servers") == "/api/servers"

    def test_claim_values_are_neutralized_too(self):
        """The claim columns go through the same writer as the pre-existing ones."""
        rows = _csv_rows(
            [{"request_id": "r", "identity": {"username": "a", "principal_name": "=1+1"}}]
        )

        assert rows[0]["principal_name"] == "'=1+1"


# =============================================================================
# Username filter: search across identity claims
# =============================================================================


class TestUsernameFilterBreadth:
    """An operator holding only an IdP-side value must still find the caller."""

    def test_identity_streams_match_every_searched_field(self):
        """Claim-bearing nested streams search the display + claim fields."""
        query = _claim_query(username="oid-abc")

        assert query["$or"] == _expected_clause("identity.", "oid-abc")
        # The filter now lives entirely in $or.
        assert "identity.username" not in query

    def test_registry_api_searches_the_display_username_alone(self):
        """registry_api records carry the claim fields as permanent nulls.

        The auth server hands the registry a thin signed assertion rather than raw
        IdP claims, so no registry_api record can ever hold a claim value. Every
        claim branch there is a per-document predicate that cannot match, on the
        stream that is both the bulk of the collection and the UI's default view.
        Dropping them cannot lose a row: no equality or regex predicate matches
        null.
        """
        query = _query(username="oid-abc")

        assert query["$or"] == [{"identity.username": {"$regex": "oid\\-abc", "$options": "i"}}]

    def test_registry_api_still_finds_a_caller_by_display_name(self):
        """The narrowing must not change what registry_api can find."""
        query = _query(username="alice")
        searched = {field for clause in query["$or"] for field in clause}

        assert searched == {"identity.username"}

    def test_token_mint_matches_top_level_fields(self):
        """token_mint stores the display username and claims at top level."""
        query = _query(stream="token_mint", username="oid-abc")

        assert query["$or"] == _expected_clause("", "oid-abc")
        assert "username" not in query

    def test_opaque_claims_match_exactly_not_as_a_substring(self):
        """An opaque id is pasted whole, so exact matching stops a short filter
        value from sweeping in unrelated users' records. It is also the only form
        an index could ever serve, for the day one is added."""
        clauses = _claim_query(username="oid-abc")["$or"]
        by_field = {field: match for clause in clauses for field, match in clause.items()}

        for field in OPAQUE_FIELDS:
            assert by_field[f"identity.{field}"] == "oid-abc", field
        # Readable identities keep substring search: operators type "alice".
        assert by_field["identity.username"] == {"$regex": "oid\\-abc", "$options": "i"}

    def test_tenant_and_app_are_not_searched(self):
        """tenant_id/app_id identify a tenant or app, not a person."""
        searched = {field for clause in _claim_query(username="x")["$or"] for field in clause}

        assert "identity.tenant_id" not in searched
        assert "identity.app_id" not in searched

    def test_opaque_claims_tolerate_a_differently_cased_paste(self):
        """Equality would silently miss an operator who pasted an Entra Object ID
        in a different case, and a case-insensitive regex could never use an
        index. So a lowercase variant is matched via $in: no silent miss, and the
        predicate stays index-eligible."""
        guid = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
        clauses = _claim_query(username=guid)["$or"]
        by_field = {field: match for clause in clauses for field, match in clause.items()}

        assert by_field["identity.object_id"] == {"$in": [guid, guid.lower()]}
        assert by_field["identity.canonical_id"] == {"$in": [guid, guid.lower()]}

    def test_already_lowercase_value_stays_a_plain_equality_match(self):
        """The common case emits the tightest possible query, not a 1-element $in."""
        by_field = {
            field: match
            for clause in _claim_query(username="oid-abc")["$or"]
            for field, match in clause.items()
        }

        assert by_field["identity.object_id"] == "oid-abc"

    def test_regex_metacharacters_stay_escaped(self):
        """Every regex branch escapes the filter value (emails contain "." )."""
        query = _query(username="alice@contoso.com")
        regex_matches = [
            match for clause in query["$or"] for match in clause.values() if isinstance(match, dict)
        ]

        assert regex_matches, "expected at least one regex branch"
        for match in regex_matches:
            assert match["$regex"] == re.escape("alice@contoso.com")

    def test_no_or_key_without_a_username_filter(self):
        """Unfiltered queries stay a plain equality match."""
        assert _query() == {"log_type": "registry_api_access"}

    def test_other_filters_are_unaffected(self):
        """Broadening identity matching does not disturb sibling filters."""
        query = _query(username="alice", operation="create", status_min=400, status_max=499)

        assert query["action.operation"] == "create"
        assert query["response.status_code"] == {"$gte": 400, "$lte": 499}


class TestStatisticsUsernameFilter:
    """The statistics drill-down filter broadens with the same field set."""

    async def _matches(self, stream: str, username: str | None):
        """Run get_statistics and return the $match of every pipeline."""
        mock_repo = MagicMock()
        mock_repo.count = AsyncMock(return_value=0)
        mock_repo.aggregate = AsyncMock(return_value=[])

        with patch("registry.audit.routes.get_audit_repository", return_value=mock_repo):
            await get_statistics(
                user_context={"is_admin": True, "username": "admin"},
                stream=stream,
                days=7,
                username=username,
            )

        return [
            stage["$match"]
            for call in mock_repo.aggregate.call_args_list
            for stage in call.args[0]
            if "$match" in stage
        ]

    async def test_identity_stream_keeps_exact_match_semantics(self):
        """Claim-bearing nested statistics anchor the readable regex; claims stay
        exact."""
        matches = await self._matches("mcp_access", "alice")

        assert matches
        for match in matches:
            assert match["$or"] == _expected_clause("identity.", "alice", anchored=True)
            assert "identity.username" not in match

    async def test_registry_api_statistics_anchor_the_username_alone(self):
        """The claim narrowing applies to the statistics pipelines too, and both
        windows get the same clause."""
        matches = await self._matches("registry_api", "alice")

        assert matches
        for match in matches:
            assert match["$or"] == [{"identity.username": {"$regex": "^alice$", "$options": "i"}}]

    async def test_token_mint_keeps_partial_match_semantics(self):
        """token_mint statistics stay unanchored on the readable fields."""
        matches = await self._matches("token_mint", "alice")

        assert matches
        for match in matches:
            assert match["$or"] == _expected_clause("", "alice")

    async def test_prior_window_filter_matches_current_window(self):
        """The week-over-week comparison window uses the same identity clause."""
        matches = await self._matches("registry_api", "alice")
        clauses = {repr(match["$or"]) for match in matches}

        assert len(clauses) == 1


# =============================================================================
# Regression guard: grouping stays on the display identity
# =============================================================================


class TestGroupingNotBroadened:
    """Dashboards must keep one row per human, not one per claim value."""

    async def test_statistics_groups_on_display_username_only(self):
        """$group keys stay on identity.username; claims never appear."""
        mock_repo = MagicMock()
        mock_repo.count = AsyncMock(return_value=0)
        mock_repo.aggregate = AsyncMock(return_value=[])

        with patch("registry.audit.routes.get_audit_repository", return_value=mock_repo):
            await get_statistics(
                user_context={"is_admin": True, "username": "admin"},
                stream="registry_api",
                days=7,
                username="alice",
            )

        group_ids = [
            stage["$group"]["_id"]
            for call in mock_repo.aggregate.call_args_list
            for stage in call.args[0]
            if "$group" in stage
        ]

        assert "$identity.username" in group_ids
        for claim in CLAIM_COLUMNS:
            assert not any(claim in str(group_id) for group_id in group_ids)

    async def test_active_user_counts_distinct_display_username_only(self):
        """DAU/WAU counts stay keyed on the single display identity field."""
        mock_repo = MagicMock()
        mock_repo.distinct = AsyncMock(return_value=["alice", "anonymous", ""])

        count = await _count_distinct_usernames(mock_repo, {"log_type": "registry_api_access"})

        assert count == 1
        assert mock_repo.distinct.call_args.args[0] == "identity.username"
