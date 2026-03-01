"""
Test helper functions for MCP Gateway Registry tests.

This module provides utility functions for common test operations.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

from registry.schemas import AgentCard, ServerDetail


def create_temp_directory() -> Path:
    """
    Create a temporary directory for test files.

    Returns:
        Path to the temporary directory
    """
    temp_dir = tempfile.mkdtemp()
    return Path(temp_dir)


def write_json_file(file_path: Path, data: dict[str, Any]) -> None:
    """
    Write data to a JSON file.

    Args:
        file_path: Path to the JSON file
        data: Dictionary to write as JSON
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_json_file(file_path: Path) -> dict[str, Any]:
    """
    Read data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary loaded from JSON
    """
    with open(file_path) as f:
        return json.load(f)


def create_test_server_file(
    servers_dir: Path, server_name: str, server_data: dict[str, Any]
) -> Path:
    """
    Create a server JSON file in the test servers directory.

    Args:
        servers_dir: Path to servers directory
        server_name: Name of the server
        server_data: Server data dictionary

    Returns:
        Path to the created server file
    """
    servers_dir.mkdir(parents=True, exist_ok=True)
    server_file = servers_dir / f"{server_name}.json"
    write_json_file(server_file, server_data)
    return server_file


def create_test_agent_file(agents_dir: Path, agent_name: str, agent_data: dict[str, Any]) -> Path:
    """
    Create an agent JSON file in the test agents directory.

    Args:
        agents_dir: Path to agents directory
        agent_name: Name of the agent
        agent_data: Agent data dictionary

    Returns:
        Path to the created agent file
    """
    agents_dir.mkdir(parents=True, exist_ok=True)
    agent_file = agents_dir / f"{agent_name}.json"
    write_json_file(agent_file, agent_data)
    return agent_file


def assert_server_equals(
    actual: ServerDetail, expected: ServerDetail, check_meta: bool = False
) -> None:
    """
    Assert that two ServerDetail objects are equal.

    Args:
        actual: Actual server detail
        expected: Expected server detail
        check_meta: Whether to check the _meta field
    """
    assert actual.name == expected.name
    assert actual.description == expected.description
    assert actual.version == expected.version
    assert actual.title == expected.title

    if check_meta:
        assert actual.meta == expected.meta


def assert_agent_equals(
    actual: AgentCard, expected: AgentCard, check_timestamps: bool = False
) -> None:
    """
    Assert that two AgentCard objects are equal.

    Args:
        actual: Actual agent card
        expected: Expected agent card
        check_timestamps: Whether to check timestamp fields
    """
    assert actual.name == expected.name
    assert actual.description == expected.description
    assert actual.url == expected.url
    assert actual.version == expected.version
    assert actual.path == expected.path

    if check_timestamps:
        assert actual.registered_at == expected.registered_at
        assert actual.updated_at == expected.updated_at


def create_mock_jwt_payload(
    username: str,
    groups: list[str] | None = None,
    scopes: list[str] | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a mock JWT payload for testing.

    Args:
        username: Username for the token
        groups: Optional list of groups
        scopes: Optional list of scopes
        extra_claims: Optional extra claims to add

    Returns:
        JWT payload dictionary
    """
    payload = {
        "sub": username,
        "username": username,
        "token_use": "access",
        "iat": 1000000000,
        "exp": 2000000000,
    }

    if groups:
        payload["cognito:groups"] = groups
        payload["groups"] = groups

    if scopes:
        payload["scope"] = " ".join(scopes)

    if extra_claims:
        payload.update(extra_claims)

    return payload


def create_test_state_file(
    state_path: Path, server_states: dict[str, dict[str, Any]] | None = None
) -> None:
    """
    Create a server state JSON file for testing.

    Args:
        state_path: Path to the state file
        server_states: Dictionary mapping server names to their state data
    """
    if server_states is None:
        server_states = {}

    write_json_file(state_path, server_states)


def create_test_agent_state_file(
    state_path: Path, agent_states: dict[str, bool] | None = None
) -> None:
    """
    Create an agent state JSON file for testing.

    Args:
        state_path: Path to the state file
        agent_states: Dictionary mapping agent paths to enabled status
    """
    if agent_states is None:
        agent_states = {}

    write_json_file(state_path, agent_states)


def normalize_text_for_comparison(text: str) -> str:
    """
    Normalize text for comparison in tests.

    Removes extra whitespace and converts to lowercase.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    return " ".join(text.lower().split())


def extract_error_message(response_data: dict[str, Any]) -> str:
    """
    Extract error message from API response.

    Args:
        response_data: Response data dictionary

    Returns:
        Error message string
    """
    if isinstance(response_data, dict):
        return response_data.get("error") or response_data.get("detail") or "Unknown error"
    return str(response_data)


def create_minimal_server_dict(
    name: str, description: str = "Test server", version: str = "1.0.0"
) -> dict[str, Any]:
    """
    Create a minimal server dictionary for testing.

    Args:
        name: Server name
        description: Server description
        version: Server version

    Returns:
        Minimal server dictionary
    """
    return {"name": name, "description": description, "version": version}


def create_minimal_agent_dict(
    name: str, url: str, description: str = "Test agent", version: str = "1.0"
) -> dict[str, Any]:
    """
    Create a minimal agent dictionary for testing.

    Args:
        name: Agent name
        url: Agent URL
        description: Agent description
        version: Agent version

    Returns:
        Minimal agent dictionary
    """
    return {
        "name": name,
        "url": url,
        "description": description,
        "version": version,
        "protocolVersion": "1.0",
        "capabilities": {},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [],
    }
