"""
Unit tests for EnforceAI IdentityContext core contract.
"""

import uuid

import pytest
from pydantic import ValidationError

from auth_server.enforceai.identity import (
    IdentityContext,
    build_user_id,
)


@pytest.mark.unit
class TestIdentityContext:
    """Test suite for IdentityContext model behavior."""

    def test_required_fields_and_serialization(self):
        agent_id = str(uuid.uuid4())
        user_id = build_user_id(
            "https://issuer.example",
            "subject-123",
        )

        context = IdentityContext(
            user_id=user_id,
            agent_id=agent_id,
            provider="oidc",
            scopes=[
                "mcp:tools:list",
                "mcp:tools:call",
            ],
            user_roles=[
                "role-a",
            ],
            metadata={
                "source": "unit-test",
            },
        )

        dumped = context.model_dump()
        assert dumped["user_id"] == user_id
        assert dumped["agent_id"] == agent_id
        assert dumped["provider"] == "oidc"
        assert dumped["scopes"] == [
            "mcp:tools:list",
            "mcp:tools:call",
        ]
        assert dumped["user_roles"] == [
            "role-a",
        ]
        assert dumped["metadata"] == {
            "source": "unit-test",
        }

        dumped_json = context.model_dump(mode="json")
        assert dumped_json["agent_id"] == agent_id

    def test_user_id_requires_issuer_namespaced_format(self):
        with pytest.raises(ValidationError, match="user_id must be in '<iss>\\|<sub>' format"):
            IdentityContext(
                user_id="subject-only",
                agent_id=str(uuid.uuid4()),
                provider="oidc",
                scopes=["mcp:tools:list"],
            )

    def test_agent_id_requires_uuid4(self):
        with pytest.raises(ValidationError, match="agent_id must be a UUIDv4 string"):
            IdentityContext(
                user_id=build_user_id(
                    "https://issuer.example",
                    "subject-123",
                ),
                agent_id=str(uuid.uuid1()),
                provider="oidc",
                scopes=["mcp:tools:list"],
            )

