"""
Unit tests for FGAC evaluation engine (scopes + allowed_tools).
"""

import uuid
from pathlib import Path

import pytest

from auth_server.enforceai.fgac.evaluate import (
    evaluate_tool_call,
    evaluate_tool_visibility,
    resolve_callable_tools_for_server,
)
from auth_server.enforceai.fgac.models import (
    MethodPolicy,
    ScopeCatalog,
    ScopeDefinition,
    ServerPermission,
    ToolPolicy,
)
from auth_server.enforceai.identity import (
    IdentityContext,
)


def _identity(
    *,
    scopes: list[str],
) -> IdentityContext:
    return IdentityContext(
        user_id="https://issuer.example|sub",
        agent_id=str(uuid.uuid4()),
        provider="oidc",
        scopes=scopes,
    )


def _catalog() -> ScopeCatalog:
    scope = ScopeDefinition(
        name="scope-1",
        server_permissions=(
            ServerPermission(
                server="mcpgw",
                methods=MethodPolicy(
                    all_methods=False,
                    methods=frozenset({"tools/list", "tools/call"}),
                ),
                tools=ToolPolicy(
                    all_tools=False,
                    tools=frozenset({"intelligent_tool_finder"}),
                ),
            ),
        ),
        agent_permissions=tuple(),
    )

    return ScopeCatalog(
        path=Path("in-memory"),
        ui_scopes={},
        group_mappings={},
        scopes={"scope-1": scope},
    )


@pytest.mark.unit
class TestFgacEvaluate:
    def test_scope_allow_deny_matrix(self) -> None:
        catalog = _catalog()
        identity = _identity(scopes=["scope-1"])

        allowed = evaluate_tool_call(
            identity=identity,
            catalog=catalog,
            server="mcpgw",
            tool="intelligent_tool_finder",
        )
        assert allowed.allowed is True
        assert allowed.reason == "allowed"
        assert allowed.matched_scope == "scope-1"

        denied_unknown_tool = evaluate_tool_call(
            identity=identity,
            catalog=catalog,
            server="mcpgw",
            tool="other_tool",
        )
        assert denied_unknown_tool.allowed is False

        denied_unknown_server = evaluate_tool_call(
            identity=identity,
            catalog=catalog,
            server="fininfo",
            tool="print_stock_data",
        )
        assert denied_unknown_server.allowed is False

    def test_allowed_tools_restriction_denies_even_if_scopes_allow(self) -> None:
        catalog = _catalog()
        identity = _identity(scopes=["scope-1"])

        denied = evaluate_tool_call(
            identity=identity,
            catalog=catalog,
            server="mcpgw",
            tool="intelligent_tool_finder",
            allowed_tools=["something_else"],
        )
        assert denied.allowed is False
        assert denied.reason == "tool_not_in_allowed_tools"

    def test_visibility_implies_callable(self) -> None:
        catalog = _catalog()
        identity = _identity(scopes=["scope-1"])

        decision_visibility = evaluate_tool_visibility(
            identity=identity,
            catalog=catalog,
            server="mcpgw",
            tool="intelligent_tool_finder",
        )
        decision_call = evaluate_tool_call(
            identity=identity,
            catalog=catalog,
            server="mcpgw",
            tool="intelligent_tool_finder",
        )

        if decision_visibility.allowed:
            assert decision_call.allowed is True

    def test_unknown_scopes_fail_closed(self) -> None:
        catalog = _catalog()
        identity = _identity(scopes=["scope-does-not-exist"])

        decision = evaluate_tool_call(
            identity=identity,
            catalog=catalog,
            server="mcpgw",
            tool="intelligent_tool_finder",
        )

        assert decision.allowed is False
        assert decision.reason == "unknown_scope"

    def test_resolve_callable_tools_for_server_obeys_allowed_tools(self) -> None:
        catalog = _catalog()
        identity = _identity(scopes=["scope-1"])

        all_callable = resolve_callable_tools_for_server(
            identity=identity,
            catalog=catalog,
            server="mcpgw",
        )
        assert all_callable.all_tools is False
        assert all_callable.tools == frozenset({"intelligent_tool_finder"})

        restricted = resolve_callable_tools_for_server(
            identity=identity,
            catalog=catalog,
            server="mcpgw",
            allowed_tools=["something_else"],
        )
        assert restricted.all_tools is False
        assert restricted.tools == frozenset()
