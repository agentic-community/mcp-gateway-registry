from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from .._validation import (
    _validate_user_id,
)


UserAuthMethod = Literal["oidc", "password"]
UserRole = Literal["admin", "user"]


class UserRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    user_id: str
    auth_method: UserAuthMethod
    username: Optional[str] = None
    email: str
    password_hash: Optional[str] = None
    role: UserRole = "user"

    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None
    disabled_at: Optional[datetime] = None

    @field_validator("user_id")
    @classmethod
    def _user_id_is_canonical(
        cls,
        value: str,
    ) -> str:
        return _validate_user_id(value)

    @field_validator("username")
    @classmethod
    def _username_is_non_empty_if_present(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("username must not be empty")
        return stripped

    @field_validator("email")
    @classmethod
    def _email_is_non_empty(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("email must not be empty")
        return stripped

    @field_validator("password_hash")
    @classmethod
    def _password_hash_is_non_empty_if_present(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("password_hash must not be empty")
        return stripped

