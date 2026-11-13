"""Tests for spec conversion: filters, overrides, annotations and preview."""

import asyncio

import conversion
import pytest
from conversion import (
    ConversionError,
    preview,
    tag_counts,
)
from models import PreviewRequest

TIMEOUT: float = 10.0


def _request(
    spec: dict,
    **options,
) -> PreviewRequest:
    return PreviewRequest(target_name="items", spec=spec, **options)


async def _tools_by_name(
    spec: dict,
    **options,
) -> dict:
    import httpx2

    request = _request(spec, **options)
    target_options = request.model_copy(update={"name": "items"})
    async with httpx2.AsyncClient(base_url=conversion.PREVIEW_BASE_URL) as client:
        sub, _names = await conversion.convert_target(spec, target_options, client, TIMEOUT)
        tools = await sub.list_tools(run_middleware=False)
    return {tool.name: tool for tool in tools}


class TestPreview:
    """What the preview reports."""

    async def test_every_operation_becomes_a_prefixed_tool(self, items_spec):
        result = await preview(_request(items_spec), TIMEOUT)

        assert result.tools == [
            "items_Echo_Things",
            "items_createItem",
            "items_deleteItem",
            "items_getItem",
            "items_listItems",
        ]

    async def test_tag_counts(self, items_spec):
        assert tag_counts(items_spec) == {"admin": 1, "items": 3}

    async def test_include_tags_and_operations_union(self, items_spec):
        result = await preview(
            _request(
                items_spec, include_tags=["admin"], include_operations=["getItem", "GET /echo"]
            ),
            TIMEOUT,
        )

        assert result.tools == ["items_Echo_Things", "items_deleteItem", "items_getItem"]

    async def test_unmatched_entries_become_warnings(self, items_spec):
        result = await preview(
            _request(
                items_spec,
                include_tags=["nope"],
                include_operations=["GET /missing"],
                tool_name_overrides={"noSuchOp": "x"},
                description_overrides={"noSuchTool": "x"},
            ),
            TIMEOUT,
        )

        assert len(result.warnings) == 4
        assert result.tools == []

    async def test_tool_name_override(self, items_spec):
        result = await preview(
            _request(items_spec, tool_name_overrides={"getItem": "fetch_item"}), TIMEOUT
        )

        assert "items_fetch_item" in result.tools
        assert result.warnings == []

    async def test_invalid_spec_raises_conversion_error(self):
        with pytest.raises(ConversionError):
            await preview(
                _request({"openapi": "3.1.0", "paths": {"/x": {"get": "not a mapping"}}}), TIMEOUT
            )

    async def test_slow_conversion_times_out(self, items_spec, monkeypatch):
        original = conversion._build_sub_server

        def _slow(*args):
            import time

            time.sleep(0.5)
            return original(*args)

        monkeypatch.setattr(conversion, "_build_sub_server", _slow)

        with pytest.raises(ConversionError, match="longer than"):
            await preview(_request(items_spec), 0.1)
        await asyncio.sleep(0.5)


class TestComponentAdjustments:
    """Annotations, input schema, descriptions and output schemas."""

    async def test_annotations_follow_http_method(self, items_spec):
        tools = await _tools_by_name(items_spec)

        assert tools["getItem"].annotations.read_only_hint is True
        assert tools["deleteItem"].annotations.destructive_hint is True
        assert tools["createItem"].annotations.read_only_hint is False
        assert tools["createItem"].annotations.destructive_hint is False

    async def test_input_schema_is_closed(self, items_spec):
        tools = await _tools_by_name(items_spec)

        assert all(tool.parameters["additionalProperties"] is False for tool in tools.values())

    async def test_description_override(self, items_spec):
        tools = await _tools_by_name(
            items_spec, description_overrides={"getItem": "Return one item."}
        )

        assert tools["getItem"].description == "Return one item."

    async def test_output_schema_permissive_by_default(self, items_spec):
        tools = await _tools_by_name(items_spec)

        assert "required" not in (tools["getItem"].output_schema or {})

    async def test_output_schema_kept_on_request(self, items_spec):
        tools = await _tools_by_name(items_spec, keep_output_schemas=True)

        assert tools["getItem"].output_schema["required"] == ["id", "photoUrls"]
