"""The one error type for OpenAPI MCP servers (issue #2022).

Each error carries the HTTP status and the machine-readable code the API
returns, so the routes map every failure the same way:
``{"error": "<code>", "detail": "<message>"}``.
"""

from typing import Any


class OpenApiServerError(Exception):
    """A request about an OpenAPI MCP server failed in a way the caller can act on."""

    def __init__(
        self,
        status_code: int,
        code: str,
        detail: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.extra = extra or {}

    def to_body(self) -> dict[str, Any]:
        return {"error": self.code, "detail": self.detail, **self.extra}


def not_found(
    what: str,
) -> OpenApiServerError:
    return OpenApiServerError(404, "not_found", f"{what} not found")


def conflict(
    code: str,
    detail: str,
    extra: dict[str, Any] | None = None,
) -> OpenApiServerError:
    return OpenApiServerError(409, code, detail, extra)


def unprocessable(
    code: str,
    detail: str,
    extra: dict[str, Any] | None = None,
) -> OpenApiServerError:
    return OpenApiServerError(422, code, detail, extra)


def container_unavailable(
    detail: str,
) -> OpenApiServerError:
    return OpenApiServerError(
        503,
        "container_unavailable",
        f"{detail}. Check that the openapi-mcp container is running and reachable "
        "at OPENAPI_MCP_INTERNAL_URL, and that both sides share OPENAPI_MCP_SYNC_SECRET.",
    )
