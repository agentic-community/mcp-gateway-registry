"""Every airegistry-tools (mcpgw) tool declares a title and MCP annotations.

Clients read readOnlyHint and destructiveHint from tools/list to decide whether
to ask a person before a call, and the registry shows them in its UI. The
fastmcp decorator is stubbed in the unit tests, so this test reads the
decorators from the source instead: a new tool added without annotations fails
here rather than shipping without them.
"""

import ast
from pathlib import Path

import pytest

SERVER_FILE: Path = Path(__file__).resolve().parents[4] / "servers" / "mcpgw" / "server.py"
ANNOTATIONS_CONSTANT: str = "READ_ONLY_TOOL_ANNOTATIONS"
EXPECTED_TOOL_COUNT: int = 8


def _tool_decorators() -> dict[str, ast.Call]:
    """Return each @mcp.tool(...) decorator call, keyed by the function name."""
    tree = ast.parse(SERVER_FILE.read_text())
    decorators: dict[str, ast.Call] = {}
    for node in tree.body:
        if not isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            is_call = isinstance(decorator, ast.Call)
            func = decorator.func if is_call else decorator
            if isinstance(func, ast.Attribute) and func.attr == "tool":
                decorators[node.name] = decorator
    return decorators


def _keyword(
    call: ast.Call,
    name: str,
) -> ast.expr | None:
    """Return the value passed for one keyword argument, or None."""
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _annotations_constant() -> dict[str, bool]:
    """Read the READ_ONLY_TOOL_ANNOTATIONS literal from the source."""
    tree = ast.parse(SERVER_FILE.read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == (
            ANNOTATIONS_CONSTANT
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{ANNOTATIONS_CONSTANT} not found in {SERVER_FILE}")


@pytest.mark.unit
def test_every_tool_is_found():
    """Guards the test itself: it must see all of the server's tools."""
    assert len(_tool_decorators()) == EXPECTED_TOOL_COUNT


@pytest.mark.unit
def test_every_tool_declares_read_only_annotations():
    for name, call in _tool_decorators().items():
        assert isinstance(call, ast.Call), f"{name}: @mcp.tool needs arguments"
        value = _keyword(call, "annotations")
        assert isinstance(value, ast.Name) and value.id == ANNOTATIONS_CONSTANT, name


@pytest.mark.unit
def test_every_tool_declares_a_title():
    for name, call in _tool_decorators().items():
        value = _keyword(call, "title")
        assert isinstance(value, ast.Constant) and value.value.strip(), name


@pytest.mark.unit
def test_annotation_values_match_what_the_tools_do():
    """All tools only read from the registry, so no tool is destructive."""
    assert _annotations_constant() == {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
