"""Security-blocked tools must not leak through registry read surfaces.

The per-tool block refuses `tools/call` and hides the tool from the proxy's
`tools/list`. Every other read projection ignored block state, which leaked two
ways:

1. `POST /api/search/semantic` returned the blocked tool AND its description.
   `intelligent_tool_finder` is built on that endpoint, so an agent searching for
   a capability got the description handed to the model. A tool earns a
   `HIGH:PROMPT INJECTION` block precisely because its description carries the
   injection, so search routed around the block completely.
2. `GET /api/servers` reported the tool as present with no marking, so the UI
   told an operator a tool was available while every call returned 403.

Those want opposite treatments, hence two functions. Model-facing projections
hide; operator-facing ones annotate so somebody can still see and act on it.
"""

from unittest.mock import AsyncMock, patch

import pytest

from registry.services.tool_blocks import annotate_blocked_tools, hide_blocked_tools

pytestmark = [pytest.mark.unit]

SERVER = "/context7"

# The real shape: two tools, one blocked by a scan.
TOOLS = [
    {"name": "resolve-library-id", "description": "You MUST call this function ..."},
    {"name": "query-docs", "description": "Retrieves documentation"},
]
OVERRIDES = {
    "resolve-library-id": {
        "blocked": True,
        "source": "security_scan",
        "reason": "HIGH:PROMPT INJECTION",
    }
}


def _repo(overrides: dict):
    repo = AsyncMock()
    repo.get_tool_overrides = AsyncMock(return_value=overrides)
    return repo


class TestHideBlockedTools:
    """Model-facing projections drop the tool entirely."""

    async def test_drops_the_blocked_tool(self):
        with patch(
            "registry.services.tool_blocks.get_server_repository",
            return_value=_repo(OVERRIDES),
        ):
            out = await hide_blocked_tools(SERVER, TOOLS)

        assert [t["name"] for t in out] == ["query-docs"]

    async def test_description_does_not_survive(self):
        """The description is the payload, so its absence is the actual control."""
        with patch(
            "registry.services.tool_blocks.get_server_repository",
            return_value=_repo(OVERRIDES),
        ):
            out = await hide_blocked_tools(SERVER, TOOLS)

        assert "You MUST call this function" not in str(out)

    async def test_search_result_shape_uses_tool_name(self):
        """Search results key the name as `tool_name`, server docs as `name`."""
        search_shaped = [{"tool_name": "resolve-library-id"}, {"tool_name": "query-docs"}]

        with patch(
            "registry.services.tool_blocks.get_server_repository",
            return_value=_repo(OVERRIDES),
        ):
            out = await hide_blocked_tools(SERVER, search_shaped)

        assert [t["tool_name"] for t in out] == ["query-docs"]

    async def test_nothing_blocked_passes_everything(self):
        with patch(
            "registry.services.tool_blocks.get_server_repository",
            return_value=_repo({}),
        ):
            out = await hide_blocked_tools(SERVER, TOOLS)

        assert len(out) == 2

    async def test_unblocked_override_is_not_treated_as_blocked(self):
        """An admin unblock stores blocked=False; it must not hide the tool."""
        with patch(
            "registry.services.tool_blocks.get_server_repository",
            return_value=_repo({"resolve-library-id": {"blocked": False, "source": "admin"}}),
        ):
            out = await hide_blocked_tools(SERVER, TOOLS)

        assert len(out) == 2

    async def test_lookup_failure_does_not_break_the_listing(self):
        """These are read projections, so a lookup error must not 500 the page.

        Enforcement still fails closed on tools/call, which is the control that
        matters; taking the whole registry UI down over an annotation would not
        make anyone safer.
        """
        repo = AsyncMock()
        repo.get_tool_overrides = AsyncMock(side_effect=RuntimeError("mongo down"))

        with patch("registry.services.tool_blocks.get_server_repository", return_value=repo):
            out = await hide_blocked_tools(SERVER, TOOLS)

        assert len(out) == 2


class TestAnnotateBlockedTools:
    """Operator-facing projections keep the tool and mark it."""

    async def test_marks_the_blocked_tool_with_a_reason(self):
        with patch(
            "registry.services.tool_blocks.get_server_repository",
            return_value=_repo(OVERRIDES),
        ):
            out = await annotate_blocked_tools(SERVER, TOOLS)

        by_name = {t["name"]: t for t in out}
        assert by_name["resolve-library-id"]["blocked"] is True
        assert by_name["resolve-library-id"]["block_reason"] == "HIGH:PROMPT INJECTION"
        assert by_name["resolve-library-id"]["block_source"] == "security_scan"

    async def test_keeps_every_tool(self):
        """Hiding here would leave nobody able to see what was blocked."""
        with patch(
            "registry.services.tool_blocks.get_server_repository",
            return_value=_repo(OVERRIDES),
        ):
            out = await annotate_blocked_tools(SERVER, TOOLS)

        assert len(out) == 2

    async def test_unblocked_tool_is_marked_false_not_omitted(self):
        """The field is always present, so the UI never has to guess."""
        with patch(
            "registry.services.tool_blocks.get_server_repository",
            return_value=_repo(OVERRIDES),
        ):
            out = await annotate_blocked_tools(SERVER, TOOLS)

        clean = [t for t in out if t["name"] == "query-docs"][0]
        assert clean["blocked"] is False
        assert "block_reason" not in clean

    async def test_does_not_mutate_the_input(self):
        """The caller's document may be shared; annotate must copy."""
        with patch(
            "registry.services.tool_blocks.get_server_repository",
            return_value=_repo(OVERRIDES),
        ):
            await annotate_blocked_tools(SERVER, TOOLS)

        assert "blocked" not in TOOLS[0]
