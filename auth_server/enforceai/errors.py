from __future__ import annotations

from typing import Optional

from fastapi import HTTPException


class EnforceAIError(Exception):
    status_code: int = 500
    error_code: str = "enforceai_error"
    default_public_message: str = "Internal error"

    def __init__(
        self,
        message: str,
        *,
        public_message: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.public_message = public_message or self.default_public_message

    def as_http_exception(self) -> HTTPException:
        return HTTPException(
            status_code=self.status_code,
            detail=self.public_message,
        )


class UnauthorizedError(EnforceAIError):
    status_code = 401
    error_code = "unauthorized"
    default_public_message = "Unauthorized"


class ForbiddenError(EnforceAIError):
    status_code = 403
    error_code = "forbidden"
    default_public_message = "Forbidden"


class DependencyUnavailableError(EnforceAIError):
    status_code = 503
    error_code = "dependency_unavailable"
    default_public_message = "Enforcement dependency unavailable"

