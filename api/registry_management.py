#!/usr/bin/env python3
"""
MCP Gateway Registry Management CLI.

High-level wrapper for the RegistryClient providing command-line interface
for server registration, management, group operations, and A2A agent management.

Server Management:
    # Register a server from JSON config
    uv run python registry_management.py register --config /path/to/config.json

    # List all servers
    uv run python registry_management.py list

    # Toggle server status
    uv run python registry_management.py toggle --path /cloudflare-docs

    # Remove server
    uv run python registry_management.py remove --path /cloudflare-docs

    # Health check
    uv run python registry_management.py healthcheck

    # Get registry configuration (deployment mode, features)
    uv run python registry_management.py config

    # Get registry configuration as JSON
    uv run python registry_management.py config --json

    # Rate a server (1-5 stars)
    uv run python registry_management.py server-rate --path /cloudflare-docs --rating 5

    # Get server rating information
    uv run python registry_management.py server-rating --path /cloudflare-docs

    # Get security scan results for a server
    uv run python registry_management.py security-scan --path /cloudflare-docs

    # Trigger manual security scan (admin only)
    uv run python registry_management.py rescan --path /cloudflare-docs

Group Management:
    # Add server to groups
    uv run python registry_management.py add-to-groups --server my-server --groups group1,group2

    # List all groups
    uv run python registry_management.py list-groups

Agent Management (A2A):
    # Register an agent
    uv run python registry_management.py agent-register --config /path/to/agent.json

    # List all agents
    uv run python registry_management.py agent-list

    # Get agent details
    uv run python registry_management.py agent-get --path /code-reviewer

    # Toggle agent status
    uv run python registry_management.py agent-toggle --path /code-reviewer --enabled true

    # Delete agent
    uv run python registry_management.py agent-delete --path /code-reviewer

    # Rate an agent (1-5 stars)
    uv run python registry_management.py agent-rate --path /code-reviewer --rating 5

    # Get agent rating information
    uv run python registry_management.py agent-rating --path /code-reviewer

    # Discover agents by skills
    uv run python registry_management.py agent-discover --skills code_analysis,bug_detection

    # Semantic agent search
    uv run python registry_management.py agent-search --query "agents that analyze code"

Anthropic Registry API (v0.1):
    # List all servers
    uv run python registry_management.py anthropic-list

    # List all servers with raw JSON output
    uv run python registry_management.py anthropic-list --raw

    # List versions for a specific server
    uv run python registry_management.py anthropic-versions --server-name "io.mcpgateway/example-server"

    # Get server details
    uv run python registry_management.py anthropic-get --server-name "io.mcpgateway/example-server" --version latest

User Management (IAM):
    # List all Keycloak users
    uv run python registry_management.py user-list

    # Search for specific users
    uv run python registry_management.py user-list --search admin

    # Create M2M service account
    uv run python registry_management.py user-create-m2m --name my-service --groups registry-admins

    # Create human user
    uv run python registry_management.py user-create-human --username john.doe --email john@example.com --first-name John --last-name Doe --groups registry-admins

    # Delete user
    uv run python registry_management.py user-delete --username john.doe

Group Management (IAM):
    # List IAM groups
    uv run python registry_management.py group-list

    # Create a new IAM group
    uv run python registry_management.py group-create --name developers --description "Developer team group"

    # Delete an IAM group
    uv run python registry_management.py group-delete --name developers --force

Federation Management:
    # Get federation configuration
    uv run python registry_management.py federation-get

    # Save federation configuration from JSON file
    uv run python registry_management.py federation-save --config federation-config.json

    # List all federation configurations
    uv run python registry_management.py federation-list

    # Add Anthropic server to federation config
    uv run python registry_management.py federation-add-anthropic-server --server-name io.github.jgador/websharp

    # Remove Anthropic server from federation config
    uv run python registry_management.py federation-remove-anthropic-server --server-name io.github.jgador/websharp

    # Add ASOR agent to federation config
    uv run python registry_management.py federation-add-asor-agent --agent-id aws_assistant

    # Remove ASOR agent from federation config
    uv run python registry_management.py federation-remove-asor-agent --agent-id aws_assistant

    # Delete federation configuration
    uv run python registry_management.py federation-delete --config-id default --force

Virtual MCP Server Management:
    # Create a virtual server from JSON config
    uv run python registry_management.py vs-create --config /path/to/virtual-server.json

    # List all virtual servers
    uv run python registry_management.py vs-list

    # List only enabled virtual servers
    uv run python registry_management.py vs-list --enabled-only

    # Get virtual server details
    uv run python registry_management.py vs-get --path /virtual/dev-tools

    # Update a virtual server from JSON config
    uv run python registry_management.py vs-update --path /virtual/dev-tools --config updated-config.json

    # Enable/disable a virtual server
    uv run python registry_management.py vs-toggle --path /virtual/dev-tools --enabled true

    # Delete a virtual server
    uv run python registry_management.py vs-delete --path /virtual/dev-tools --force

    # Rate a virtual server (1-5 stars)
    uv run python registry_management.py vs-rate --path /virtual/dev-tools --rating 5

    # Get virtual server rating
    uv run python registry_management.py vs-rating --path /virtual/dev-tools

Global Options (can be set via environment variables or command-line arguments):
    --registry-url URL       Registry base URL (overrides REGISTRY_URL env var)
    --aws-region REGION      AWS region (overrides AWS_REGION env var)
    --keycloak-url URL       Keycloak base URL (overrides KEYCLOAK_URL env var)
    --token-file PATH        Path to file containing JWT token (bypasses token script)

Environment Variables (used if command-line options not provided):
    REGISTRY_URL: Registry base URL (e.g., https://registry.mycorp.click)
    AWS_REGION: AWS region where Keycloak and SSM are deployed (e.g., us-east-1)
    KEYCLOAK_URL: Keycloak base URL (e.g., https://kc.us-east-1.mycorp.click)

Environment Variables (Optional):
    CLIENT_NAME: Keycloak client name (default: registry-admin-bot)
    GET_TOKEN_SCRIPT: Path to get-m2m-token.sh script

Local Development (running against local Docker Compose setup):
    When running the solution locally with Docker Compose, you can use the --token-file
    option to provide a pre-generated JWT token instead of dynamically fetching one.

    Step 1: Generate credentials using the credentials provider script:
        cd credentials-provider
        ./generate_creds.sh

    Step 2: Use the generated token file with the CLI:
        uv run python api/registry_management.py --debug \\
            --registry-url http://localhost \\
            --token-file .oauth-tokens/ingress.json \\
            list 2>&1 | tee debug.log

    The credentials-provider/generate_creds.sh script creates tokens in .oauth-tokens/
    directory. The ingress.json token file contains the admin JWT token that can be
    used with the registry management CLI.

    Other examples for local development:
        # List users
        uv run python api/registry_management.py --debug \\
            --registry-url http://localhost \\
            --token-file .oauth-tokens/ingress.json \\
            user-list

        # Health check
        uv run python api/registry_management.py --debug \\
            --registry-url http://localhost \\
            --token-file .oauth-tokens/ingress.json \\
            healthcheck

        # Create M2M account
        uv run python api/registry_management.py --debug \\
            --registry-url http://localhost \\
            --token-file .oauth-tokens/ingress.json \\
            user-create-m2m --name test-bot --groups developers
"""

import argparse
import json
import logging
import os
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from registry_client import (
    RegistryClient,
    InternalServiceRegistration,
    ServerListResponse,
    ToggleResponse,
    GroupListResponse,
    AgentRegistration,
    AgentProvider,
    AgentVisibility,
    Skill,
    AgentListResponse,
    AgentDetail,
    AgentToggleResponse,
    AgentDiscoveryResponse,
    AgentSemanticDiscoveryResponse,
    RatingResponse,
    RatingInfoResponse,
    AgentSecurityScanResponse,
    AgentRescanResponse,
    AnthropicServerList,
    AnthropicServerResponse,
    M2MAccountRequest,
    HumanUserRequest,
    UserSummary,
    UserListResponse,
    UserDeleteResponse,
    M2MAccountResponse,
    GroupCreateRequest,
    GroupSummary,
    GroupDeleteResponse,
    SkillRegistrationRequest,
    SkillCard,
    SkillListResponse,
    SkillHealthResponse,
    SkillContentResponse,
    SkillSearchResponse,
    SkillToggleResponse,
    SkillRatingResponse,
    SkillSecurityScanResponse,
    SkillRescanResponse,
    ToolMapping,
    ToolScopeOverride,
    VirtualServerCreateRequest,
    VirtualServerConfig,
    VirtualServerListResponse,
    VirtualServerToggleResponse,
    VirtualServerDeleteResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _get_registry_url(cli_value: Optional[str] = None) -> str:
    """
    Get registry URL from command-line argument or environment variable.

    Args:
        cli_value: Command-line argument value (overrides environment variable)

    Returns:
        Registry base URL

    Raises:
        ValueError: If REGISTRY_URL is not provided
    """
    registry_url = cli_value or os.getenv("REGISTRY_URL")
    if not registry_url:
        raise ValueError(
            "REGISTRY_URL is required.\n"
            "Set via environment variable or --registry-url option:\n"
            "  export REGISTRY_URL=https://registry.mycorp.click\n"
            "  OR\n"
            "  --registry-url https://registry.mycorp.click"
        )

    logger.debug(f"Using registry URL: {registry_url}")
    return registry_url


def _mask_sensitive_fields(
    data: Any,
    fields_to_mask: Optional[List[str]] = None,
) -> Any:
    """
    Mask sensitive fields in response data for safe logging/printing.

    Args:
        data: Response data (dict, list, or other)
        fields_to_mask: List of field names to mask (default: federation_token)

    Returns:
        Data with sensitive fields masked
    """
    if fields_to_mask is None:
        fields_to_mask = ["federation_token"]

    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            if key in fields_to_mask and value:
                # Show first 3 chars followed by ...
                if isinstance(value, str) and len(value) > 3:
                    masked[key] = f"{value[:3]}..."
                else:
                    masked[key] = "***"
            else:
                masked[key] = _mask_sensitive_fields(value, fields_to_mask)
        return masked
    elif isinstance(data, list):
        return [_mask_sensitive_fields(item, fields_to_mask) for item in data]
    else:
        return data


def _get_client_name() -> str:
    """
    Get Keycloak client name from environment variable or default.

    Returns:
        Client name
    """
    client_name = os.getenv("CLIENT_NAME", "registry-admin-bot")
    logger.debug(f"Using client name: {client_name}")
    return client_name


def _get_token_script() -> str:
    """
    Get path to get-m2m-token.sh script.

    Returns:
        Script path
    """
    # Default to get-m2m-token.sh in the same directory as this script
    script_dir = Path(__file__).parent
    default_script = str(script_dir / "get-m2m-token.sh")
    script_path = os.getenv("GET_TOKEN_SCRIPT", default_script)
    logger.debug(f"Using token script: {script_path}")
    return script_path


def _get_jwt_token(aws_region: Optional[str] = None, keycloak_url: Optional[str] = None) -> str:
    """
    Retrieve JWT token using get-m2m-token.sh script.

    Args:
        aws_region: AWS region (passed to script via --aws-region)
        keycloak_url: Keycloak URL (passed to script via --keycloak-url)

    Returns:
        JWT access token

    Raises:
        RuntimeError: If token retrieval fails
    """
    client_name = _get_client_name()
    script_path = _get_token_script()

    try:
        # Redact client name in logs for security
        logger.debug(f"Retrieving token for client: {client_name}")

        # Build command with optional arguments
        cmd = [script_path]
        if aws_region:
            cmd.extend(["--aws-region", aws_region])
        if keycloak_url:
            cmd.extend(["--keycloak-url", keycloak_url])
        cmd.append(client_name)

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        token = result.stdout.strip()

        if not token:
            raise RuntimeError("Empty token returned from get-m2m-token.sh")

        # Redact token in logs - show only first 8 characters
        redacted_token = f"{token[:8]}..." if len(token) > 8 else "***"
        logger.debug(f"Successfully retrieved JWT token: {redacted_token}")
        return token

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to retrieve token: {e.stderr}")
        raise RuntimeError(f"Token retrieval failed: {e.stderr}") from e
    except Exception as e:
        logger.error(f"Unexpected error retrieving token: {e}")
        raise RuntimeError(f"Token retrieval error: {e}") from e


def _load_json_config(config_path: str) -> Dict[str, Any]:
    """
    Load JSON configuration file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = json.load(f)

    logger.debug(f"Loaded configuration from {config_path}")
    return config


def _create_client(args: argparse.Namespace) -> RegistryClient:
    """
    Create and return a configured RegistryClient instance.

    Args:
        args: Command arguments containing optional CLI values

    Returns:
        RegistryClient instance

    Raises:
        RuntimeError: If token retrieval fails
        FileNotFoundError: If token file not found
        ValueError: If required configuration is missing
    """
    # Check all required configuration upfront
    missing_params = []

    # Check REGISTRY_URL
    registry_url = args.registry_url or os.getenv("REGISTRY_URL")
    if not registry_url:
        missing_params.append("REGISTRY_URL")

    # Check if token file is provided
    if hasattr(args, "token_file") and args.token_file:
        token_path = Path(args.token_file)
        if not token_path.exists():
            raise FileNotFoundError(f"Token file not found: {args.token_file}")

        logger.debug(f"Loading token from file: {args.token_file}")

        # Try to parse as JSON first (token files from generate-agent-token.sh or UI)
        try:
            with open(token_path, "r") as f:
                token_data = json.load(f)
            # Extract access_token - handle multiple JSON formats:
            # Format 1: {"access_token": "..."} (from generate-agent-token.sh)
            # Format 2: {"tokens": {"access_token": "..."}, ...} (from UI "Get JWT Token")
            # Format 3: {"token_data": {"access_token": "..."}, ...} (alternative UI format)
            token = token_data.get("access_token")
            if not token and "tokens" in token_data:
                token = token_data["tokens"].get("access_token")
            if not token and "token_data" in token_data:
                token = token_data["token_data"].get("access_token")
            if not token:
                raise RuntimeError(
                    f"No 'access_token' field found in token file: {args.token_file}"
                )
        except json.JSONDecodeError:
            # Fall back to plain text token file
            token = token_path.read_text().strip()

        if not token:
            raise RuntimeError(f"Empty token in file: {args.token_file}")

        # Redact token in logs - show only first 8 characters
        redacted_token = f"{token[:8]}..." if len(token) > 8 else "***"
        logger.debug(f"Successfully loaded token from file: {redacted_token}")
    else:
        # Check parameters needed for token script
        aws_region = args.aws_region or os.getenv("AWS_REGION")
        keycloak_url = args.keycloak_url or os.getenv("KEYCLOAK_URL")

        if not aws_region:
            missing_params.append("AWS_REGION")
        if not keycloak_url:
            missing_params.append("KEYCLOAK_URL")

        # If any parameters are missing, raise comprehensive error
        if missing_params:
            error_msg = "Missing required configuration:\n\n"
            for param in missing_params:
                error_msg += f"  - {param}\n"
            error_msg += "\nSet via environment variables or command-line options:\n\n"
            if "REGISTRY_URL" in missing_params:
                error_msg += "  export REGISTRY_URL=https://registry.example.com\n"
                error_msg += "  OR use --registry-url https://registry.example.com\n\n"
            if "AWS_REGION" in missing_params:
                error_msg += "  export AWS_REGION=us-east-1\n"
                error_msg += "  OR use --aws-region us-east-1\n\n"
            if "KEYCLOAK_URL" in missing_params:
                error_msg += "  export KEYCLOAK_URL=https://keycloak.example.com\n"
                error_msg += "  OR use --keycloak-url https://keycloak.example.com\n\n"
            error_msg += "Alternatively, use --token-file to provide a pre-generated JWT token."
            raise ValueError(error_msg)

        token = _get_jwt_token(aws_region=aws_region, keycloak_url=keycloak_url)

    # Final check for registry URL (in case token file path was provided)
    if missing_params and "REGISTRY_URL" in missing_params:
        raise ValueError(
            "REGISTRY_URL is required.\n"
            "Set via environment variable or --registry-url option:\n"
            "  export REGISTRY_URL=https://registry.example.com\n"
            "  OR\n"
            "  --registry-url https://registry.example.com"
        )

    return RegistryClient(registry_url=registry_url, token=token)


def cmd_register(args: argparse.Namespace) -> int:
    """
    Register a new server from JSON configuration.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        config = _load_json_config(args.config)

        # Convert config to InternalServiceRegistration
        # Handle both old and new config formats
        registration = InternalServiceRegistration(
            service_path=config.get("path") or config.get("service_path"),
            name=config.get("server_name") or config.get("name"),
            description=config.get("description"),
            proxy_pass_url=config.get("proxy_pass_url"),
            version=config.get("version"),
            status=config.get("status"),
            auth_provider=config.get("auth_provider"),
            auth_scheme=config.get("auth_scheme", config.get("auth_type")),
            supported_transports=config.get("supported_transports"),
            headers=config.get("headers"),
            tool_list_json=config.get("tool_list_json"),
            tags=config.get("tags"),
            overwrite=args.overwrite,
            mcp_endpoint=config.get("mcp_endpoint"),
            sse_endpoint=config.get("sse_endpoint"),
            metadata=config.get("metadata", {}),
        )

        client = _create_client(args)
        response = client.register_service(registration)

        logger.info(f"Server registered successfully: {response.path}")
        logger.info(f"Message: {response.message}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {e}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON configuration: {e}")
        return 1
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """
    List all registered servers.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)

        # Print raw JSON if requested - fetch directly from API to get all fields
        if hasattr(args, "json") and args.json:
            import json

            raw_response = client._make_request(method="GET", endpoint="/api/servers")
            print(json.dumps(raw_response.json(), indent=2, default=str))
            return 0

        response = client.list_services()

        if not response.servers:
            logger.info("No servers registered")
            return 0

        logger.info(f"Found {len(response.servers)} registered servers:\n")

        for server in response.servers:
            status_icon = "✓" if server.is_enabled else "✗"
            health_icon = {
                "healthy": "🟢",
                "unhealthy": "🔴",
                "unknown": "⚪",
                "disabled": "⚫",
            }.get(server.health_status.value, "⚪")

            print(f"{status_icon} {health_icon} {server.path}")
            print(f"   Name: {server.display_name}")
            print(f"   Description: {server.description}")
            print(f"   Enabled: {server.is_enabled}")
            print(f"   Health: {server.health_status.value}")
            print()

        return 0

    except Exception as e:
        logger.error(f"List operation failed: {e}")
        return 1


def cmd_toggle(args: argparse.Namespace) -> int:
    """
    Toggle server enabled/disabled status.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.toggle_service(args.path)

        status = "enabled" if response.is_enabled else "disabled"
        logger.info(f"Server {response.path} is now {status}")
        logger.info(f"Message: {response.message}")
        return 0

    except Exception as e:
        logger.error(f"Toggle operation failed: {e}")
        return 1


def cmd_remove(args: argparse.Namespace) -> int:
    """
    Remove a server from the registry.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        if not args.force:
            confirmation = input(f"Remove server {args.path}? (yes/no): ")
            if confirmation.lower() != "yes":
                logger.info("Operation cancelled")
                return 0

        client = _create_client(args)
        response = client.remove_service(args.path)

        logger.info(f"Server removed successfully: {args.path}")
        return 0

    except Exception as e:
        logger.error(f"Remove operation failed: {e}")
        return 1


def cmd_healthcheck(args: argparse.Namespace) -> int:
    """
    Perform health check on all servers.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.healthcheck()

        logger.info(f"Health check status: {response.get('status', 'unknown')}")
        logger.info("\nHealth check results:")
        print(json.dumps(response, indent=2))
        return 0

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    """
    Get registry configuration including deployment mode and features.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.get_config()

        logger.info(f"Deployment Mode: {response.get('deployment_mode', 'unknown')}")
        logger.info(f"Registry Mode: {response.get('registry_mode', 'unknown')}")
        logger.info(f"Nginx Updates Enabled: {response.get('nginx_updates_enabled', 'unknown')}")

        if args.json:
            print(json.dumps(response, indent=2))
        else:
            print("\nRegistry Configuration:")
            print(f"  Deployment Mode:       {response.get('deployment_mode')}")
            print(f"  Registry Mode:         {response.get('registry_mode')}")
            print(f"  Nginx Updates Enabled: {response.get('nginx_updates_enabled')}")
            print("\nEnabled Features:")
            features = response.get("features", {})
            for feature, enabled in features.items():
                status = "enabled" if enabled else "disabled"
                print(f"  {feature}: {status}")

        return 0

    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        return 1


def cmd_add_to_groups(args: argparse.Namespace) -> int:
    """
    Add server to user groups.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        groups = [g.strip() for g in args.groups.split(",")]
        client = _create_client(args)
        response = client.add_server_to_groups(args.server, groups)

        logger.info(f"Server {args.server} added to groups: {', '.join(groups)}")
        return 0

    except Exception as e:
        logger.error(f"Add to groups failed: {e}")
        return 1


def cmd_remove_from_groups(args: argparse.Namespace) -> int:
    """
    Remove server from user groups.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        groups = [g.strip() for g in args.groups.split(",")]
        client = _create_client(args)
        response = client.remove_server_from_groups(args.server, groups)

        logger.info(f"Server {args.server} removed from groups: {', '.join(groups)}")
        return 0

    except Exception as e:
        logger.error(f"Remove from groups failed: {e}")
        return 1


def cmd_create_group(args: argparse.Namespace) -> int:
    """
    Create a new user group.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.create_group(
            group_name=args.name, description=args.description, create_in_idp=args.idp
        )

        logger.info(f"Group created successfully: {args.name}")
        return 0

    except Exception as e:
        logger.error(f"Create group failed: {e}")
        return 1


def cmd_delete_group(args: argparse.Namespace) -> int:
    """
    Delete a user group.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        if not args.force:
            confirmation = input(f"Delete group {args.name}? (yes/no): ")
            if confirmation.lower() != "yes":
                logger.info("Operation cancelled")
                return 0

        client = _create_client(args)
        response = client.delete_group(
            group_name=args.name, delete_from_idp=args.idp, force=args.force
        )

        logger.info(f"Group deleted successfully: {args.name}")
        return 0

    except Exception as e:
        logger.error(f"Delete group failed: {e}")
        return 1


def cmd_import_group(args: argparse.Namespace) -> int:
    """
    Import a complete group definition from JSON file.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import json

    try:
        # Read JSON file
        with open(args.file, "r") as f:
            group_definition = json.load(f)

        # Validate required field
        if "scope_name" not in group_definition:
            logger.error("JSON file must contain 'scope_name' field")
            return 1

        client = _create_client(args)
        response = client.import_group(group_definition)

        logger.info(f"Group imported successfully: {group_definition['scope_name']}")
        logger.info(f"Response: {json.dumps(response, indent=2)}")
        return 0

    except FileNotFoundError:
        logger.error(f"File not found: {args.file}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file: {e}")
        return 1
    except Exception as e:
        logger.error(f"Import group failed: {e}")
        return 1


def cmd_list_groups(args: argparse.Namespace) -> int:
    """
    List all user groups.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import json

    try:
        client = _create_client(args)
        response = client.list_groups(
            include_keycloak=not args.no_keycloak, include_scopes=not args.no_scopes
        )

        # If JSON output requested, print raw response and exit
        if hasattr(args, "json") and args.json:
            print(json.dumps(response.model_dump(), indent=2, default=str))
            return 0

        # Display synchronized groups
        if response.synchronized:
            print("\n=== Synchronized Groups (in both Keycloak and Scopes) ===")
            for group_name in response.synchronized:
                print(f"  - {group_name}")
                # Show details from scopes if available
                if group_name in response.scopes_groups:
                    group_info = response.scopes_groups[group_name]
                    if "description" in group_info:
                        print(f"    Description: {group_info['description']}")
                    if "server_count" in group_info:
                        print(f"    Servers: {group_info['server_count']}")

        # Display Keycloak-only groups
        if response.keycloak_only:
            print("\n=== Keycloak-Only Groups (not in Scopes) ===")
            for group_name in response.keycloak_only:
                print(f"  - {group_name}")

        # Display Scopes-only groups
        if response.scopes_only:
            print("\n=== Scopes-Only Groups (not in Keycloak) ===")
            for group_name in response.scopes_only:
                print(f"  - {group_name}")
                if group_name in response.scopes_groups:
                    group_info = response.scopes_groups[group_name]
                    if "description" in group_info:
                        print(f"    Description: {group_info['description']}")

        # Summary
        total_keycloak = len(response.keycloak_groups)
        total_scopes = len(response.scopes_groups)
        print(f"\n=== Summary ===")
        print(f"Total Keycloak groups: {total_keycloak}")
        print(f"Total Scopes groups: {total_scopes}")
        print(f"Synchronized: {len(response.synchronized)}")
        print(f"Keycloak-only: {len(response.keycloak_only)}")
        print(f"Scopes-only: {len(response.scopes_only)}")

        return 0

    except Exception as e:
        logger.error(f"List groups failed: {e}")
        return 1


def cmd_describe_group(args: argparse.Namespace) -> int:
    """
    Describe a specific group with all details.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import json

    try:
        client = _create_client(args)
        group_name = args.name

        # Get full group details from scopes storage
        try:
            group_data = client.get_group(group_name)
        except Exception as e:
            if "404" in str(e):
                logger.error(f"Group '{group_name}' not found in scopes storage")
                group_data = None
            else:
                raise

        # If JSON output requested
        if hasattr(args, "json") and args.json:
            if group_data:
                print(json.dumps(group_data, indent=2, default=str))
                return 0
            else:
                print(json.dumps({"error": "Group not found", "group_name": group_name}, indent=2))
                return 1

        # Human-readable output
        if not group_data:
            print(f"\nGroup '{group_name}' not found in scopes storage\n")
            return 1

        print(f"\n=== Group: {group_name} ===\n")
        print(f"Scope Type: {group_data.get('scope_type', 'N/A')}")
        print(f"Description: {group_data.get('description', 'N/A')}")
        print(f"Created: {group_data.get('created_at', 'N/A')}")
        print(f"Updated: {group_data.get('updated_at', 'N/A')}")

        print("\nServer Access:")
        server_access = group_data.get("server_access", [])
        if server_access:
            for idx, access in enumerate(server_access, 1):
                print(f"  {idx}. Server: {access.get('server', 'N/A')}")
                if "methods" in access:
                    print(f"     Methods: {', '.join(access['methods'])}")
                if "tools" in access:
                    print(f"     Tools: {', '.join(access['tools'])}")
                if "agents" in access:
                    print(f"     Agents: {json.dumps(access['agents'], indent=6)}")
        else:
            print("  None")

        print("\nGroup Mappings:")
        group_mappings = group_data.get("group_mappings", [])
        if group_mappings:
            for mapping in group_mappings:
                print(f"  - {mapping}")
        else:
            print("  None")

        print("\nUI Permissions:")
        ui_permissions = group_data.get("ui_permissions", {})
        if ui_permissions:
            print(json.dumps(ui_permissions, indent=2))
        else:
            print("  None")

        return 0

    except Exception as e:
        logger.error(f"Describe group failed: {e}")
        return 1


def cmd_server_rate(args: argparse.Namespace) -> int:
    """
    Rate a server (1-5 stars).

    Args:
        args: Command arguments with path and rating

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response: RatingResponse = client.rate_server(path=args.path, rating=args.rating)

        logger.info(f"✓ {response.message}")
        logger.info(f"Average rating: {response.average_rating:.2f} stars")

        return 0

    except Exception as e:
        logger.error(f"Failed to rate server: {e}")
        return 1


def cmd_server_rating(args: argparse.Namespace) -> int:
    """
    Get rating information for a server.

    Args:
        args: Command arguments with path

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response: RatingInfoResponse = client.get_server_rating(path=args.path)

        logger.info(f"\nRating for server '{args.path}':")
        logger.info(f"  Average: {response.num_stars:.2f} stars")
        logger.info(f"  Total ratings: {len(response.rating_details)}")

        if response.rating_details:
            logger.info("\nIndividual ratings (most recent):")
            # Show first 10 ratings
            for detail in response.rating_details[:10]:
                logger.info(f"  {detail.user}: {detail.rating} stars")

            if len(response.rating_details) > 10:
                logger.info(f"  ... and {len(response.rating_details) - 10} more")

        return 0

    except Exception as e:
        logger.error(f"Failed to get ratings: {e}")
        return 1


def cmd_security_scan(args: argparse.Namespace) -> int:
    """
    Get security scan results for a server.

    Args:
        args: Command arguments with path and optional json flag

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response: SecurityScanResult = client.get_security_scan(path=args.path)

        if args.json:
            # Output raw JSON
            print(json.dumps(response.model_dump(), indent=2, default=str))
        else:
            # Pretty print results
            logger.info(f"\nSecurity scan results for server '{args.path}':")

            # Display analysis results by analyzer
            if response.analysis_results:
                for analyzer_name, analyzer_data in response.analysis_results.items():
                    logger.info(f"\n  Analyzer: {analyzer_name}")
                    if isinstance(analyzer_data, dict) and "findings" in analyzer_data:
                        findings = analyzer_data["findings"]
                        logger.info(f"    Findings: {len(findings)}")
                        for finding in findings[:5]:  # Show first 5
                            severity = finding.get("severity", "UNKNOWN")
                            tool_name = finding.get("tool_name", "unknown")
                            logger.info(f"      - {tool_name}: {severity}")
                        if len(findings) > 5:
                            logger.info(f"      ... and {len(findings) - 5} more")

            # Display tool results summary
            if response.tool_results:
                logger.info(f"\n  Total tools scanned: {len(response.tool_results)}")
                safe_count = sum(1 for tool in response.tool_results if tool.get("is_safe", False))
                unsafe_count = len(response.tool_results) - safe_count
                logger.info(f"  Safe tools: {safe_count}")
                if unsafe_count > 0:
                    logger.info(f"  Unsafe tools: {unsafe_count}")
                    logger.warning("\n  WARNING: Some tools flagged as potentially unsafe!")

        return 0

    except Exception as e:
        logger.error(f"Failed to get security scan results: {e}")
        return 1


def cmd_rescan(args: argparse.Namespace) -> int:
    """
    Trigger manual security scan for a server (admin only).

    Args:
        args: Command arguments with path and optional json flag

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response: RescanResponse = client.rescan_server(path=args.path)

        if args.json:
            # Output raw JSON
            print(json.dumps(response.model_dump(), indent=2, default=str))
        else:
            # Pretty print results
            safety_status = "SAFE" if response.is_safe else "UNSAFE"
            logger.info(f"\nSecurity scan completed for server '{args.path}':")
            logger.info(f"  Status: {safety_status}")
            logger.info(f"  Scan timestamp: {response.scan_timestamp}")
            logger.info(f"  Analyzers used: {', '.join(response.analyzers_used)}")
            logger.info(f"\n  Severity counts:")
            logger.info(f"    Critical: {response.critical_issues}")
            logger.info(f"    High: {response.high_severity}")
            logger.info(f"    Medium: {response.medium_severity}")
            logger.info(f"    Low: {response.low_severity}")

            if response.scan_failed:
                logger.error(f"\n  Scan failed: {response.error_message}")
                return 1

            if not response.is_safe:
                logger.warning("\n  WARNING: Server flagged as potentially unsafe!")

        return 0

    except Exception as e:
        logger.error(f"Failed to trigger security scan: {e}")
        return 1


def cmd_server_update_credential(args: argparse.Namespace) -> int:
    """
    Update authentication credentials for a server.

    Args:
        args: Command arguments with path, auth-scheme, credential, etc.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Validate that credential is provided when auth_scheme is not 'none'
        if args.auth_scheme != "none" and not args.credential:
            logger.error("--credential is required when --auth-scheme is not 'none'")
            return 1

        client = _create_client(args)
        response = client.update_server_credential(
            service_path=args.path,
            auth_scheme=args.auth_scheme,
            auth_credential=args.credential,
            auth_header_name=args.auth_header_name,
        )

        if args.json:
            # Output raw JSON
            print(json.dumps(response, indent=2, default=str))
        else:
            # Pretty print results
            logger.info(f"\nAuth credential updated successfully for '{args.path}':")
            logger.info(f"  Auth scheme: {response.get('auth_scheme')}")
            if response.get("auth_header_name"):
                logger.info(f"  Header name: {response.get('auth_header_name')}")
            logger.info(f"  Message: {response.get('message')}")

        return 0

    except Exception as e:
        logger.error(f"Failed to update server credential: {e}")
        return 1


def cmd_server_search(args: argparse.Namespace) -> int:
    """
    Perform semantic search across all entity types.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.semantic_search(query=args.query, max_results=args.max_results)

        if args.json:
            # Output raw JSON
            print(json.dumps(response.model_dump(), indent=2, default=str))
            return 0

        total_results = (
            len(response.servers)
            + len(response.tools)
            + len(response.agents)
            + len(response.skills)
            + len(response.virtual_servers)
        )

        if total_results == 0:
            logger.info("No results found matching the query")
            return 0

        logger.info(f"Search mode: {response.search_mode}")

        # Display MCP Servers
        if response.servers:
            print(f"\n--- MCP Servers ({len(response.servers)}) ---")
            for server in response.servers:
                print(f"  {server.server_name} ({server.path})")
                print(f"    Relevance: {server.relevance_score:.2%}")
                if server.tags:
                    print(f"    Tags: {', '.join(server.tags[:5])}")
                if server.description:
                    desc = (
                        server.description[:100] + "..."
                        if len(server.description) > 100
                        else server.description
                    )
                    print(f"    {desc}")
                print()

        # Display Tools
        if response.tools:
            print(f"\n--- Tools ({len(response.tools)}) ---")
            for tool in response.tools:
                print(f"  {tool.tool_name} (from {tool.server_path})")
                print(f"    Relevance: {tool.relevance_score:.2%}")
                if tool.description:
                    desc = (
                        tool.description[:100] + "..."
                        if len(tool.description) > 100
                        else tool.description
                    )
                    print(f"    {desc}")
                print()

        # Display A2A Agents
        if response.agents:
            print(f"\n--- A2A Agents ({len(response.agents)}) ---")
            for agent in response.agents:
                agent_name = agent.agent_card.get("name", "Unknown")
                agent_desc = agent.agent_card.get("description", "")
                agent_skills = agent.agent_card.get("skills", [])
                print(f"  {agent_name} ({agent.path})")
                print(f"    Relevance: {agent.relevance_score:.2%}")
                if agent_skills:
                    skill_names = [
                        s.get("name", "") if isinstance(s, dict) else str(s)
                        for s in agent_skills[:5]
                    ]
                    print(f"    Skills: {', '.join(skill_names)}")
                if agent_desc:
                    desc = agent_desc[:100] + "..." if len(agent_desc) > 100 else agent_desc
                    print(f"    {desc}")
                print()

        # Display Skills
        if response.skills:
            print(f"\n--- Skills ({len(response.skills)}) ---")
            for skill in response.skills:
                print(f"  {skill.skill_name} ({skill.path})")
                print(f"    Relevance: {skill.relevance_score:.2%}")
                if skill.author:
                    print(f"    Author: {skill.author}")
                if skill.tags:
                    print(f"    Tags: {', '.join(skill.tags[:5])}")
                if skill.description:
                    desc = (
                        skill.description[:100] + "..."
                        if len(skill.description) > 100
                        else skill.description
                    )
                    print(f"    {desc}")
                print()

        # Display Virtual MCP Servers
        if response.virtual_servers:
            print(f"\n--- Virtual MCP Servers ({len(response.virtual_servers)}) ---")
            for vs in response.virtual_servers:
                print(f"  {vs.server_name} ({vs.path})")
                print(f"    Relevance: {vs.relevance_score:.2%}")
                print(f"    Tools: {vs.num_tools}, Backends: {vs.backend_count}")
                if vs.backend_paths:
                    print(f"    Backend paths: {', '.join(vs.backend_paths)}")
                if vs.tags:
                    print(f"    Tags: {', '.join(vs.tags[:5])}")
                if vs.description:
                    desc = (
                        vs.description[:100] + "..."
                        if len(vs.description) > 100
                        else vs.description
                    )
                    print(f"    {desc}")
                print()

        return 0

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return 1


# Server Version Management Command Handlers


def cmd_list_versions(args: argparse.Namespace) -> int:
    """
    List all versions for a server.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.get_server_versions(path=args.path)

        if args.json:
            print(json.dumps(response, indent=2, default=str))
            return 0

        logger.info(f"Versions for server {response['path']}:\n")
        logger.info(f"Default version: {response['default_version']}\n")

        for v in response.get("versions", []):
            default_marker = " (DEFAULT)" if v.get("is_default") else ""
            status = v.get("status", "stable")
            print(f"  {v['version']}{default_marker}")
            print(f"    Status: {status}")
            print(f"    URL: {v.get('proxy_pass_url', 'N/A')}")
            print()

        return 0

    except Exception as e:
        logger.error(f"Failed to list versions: {e}")
        return 1


def cmd_remove_version(args: argparse.Namespace) -> int:
    """
    Remove a version from a server.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.remove_server_version(path=args.path, version=args.version)

        if args.json:
            print(json.dumps(response, indent=2, default=str))
            return 0

        logger.info(f"Successfully removed version {args.version} from {args.path}")
        return 0

    except Exception as e:
        logger.error(f"Failed to remove version: {e}")
        return 1


def cmd_set_default_version(args: argparse.Namespace) -> int:
    """
    Set the default version for a server.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.set_default_version(path=args.path, version=args.version)

        if args.json:
            print(json.dumps(response, indent=2, default=str))
            return 0

        logger.info(f"Successfully set default version to {args.version} for {args.path}")
        return 0

    except Exception as e:
        logger.error(f"Failed to set default version: {e}")
        return 1


# Agent Management Command Handlers


def cmd_agent_register(args: argparse.Namespace) -> int:
    """
    Register a new A2A agent from JSON configuration.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1

        with open(config_path, "r") as f:
            config = json.load(f)

        # Convert skills list of dicts to Skill objects
        # Handle both 'input_schema' and 'parameters' field names
        # Also handle 'id' vs 'name' field for skill identifier
        skills = []
        for skill_data in config.get("skills", []):
            # Get skill identifier - prefer 'id', fall back to 'name'
            skill_id = skill_data.get("id") or skill_data.get("name", "")
            skill_name = skill_data.get("name", skill_id)

            # Normalize field names
            skill_dict = {
                "id": skill_id,  # Always include id field
                "name": skill_name,
                "description": skill_data.get("description", ""),
                "tags": skill_data.get("tags", []),  # Include tags field
            }
            # Use 'input_schema' if present, otherwise use 'parameters'
            if "input_schema" in skill_data:
                skill_dict["input_schema"] = skill_data["input_schema"]
            elif "parameters" in skill_data:
                skill_dict["input_schema"] = skill_data["parameters"]

            skills.append(Skill(**skill_dict))
        config["skills"] = skills

        # Provider is now a dict object per A2A spec {organization, url}
        # No conversion needed - pass it through as-is

        # Convert visibility string to enum if present
        if "visibility" in config:
            try:
                config["visibility"] = AgentVisibility(config["visibility"].lower())
            except ValueError:
                logger.warning(f"Unknown visibility '{config['visibility']}', using 'public'")
                config["visibility"] = AgentVisibility.PUBLIC

        # Handle security_schemes conversion
        # Normalize common security type variations to A2A spec values
        if "security_schemes" in config:
            transformed_schemes = {}
            for scheme_name, scheme_data in config["security_schemes"].items():
                scheme_type = scheme_data.get("type", "").lower()
                # Normalize to A2A spec values: apiKey, http, oauth2, openIdConnect
                # Keep 'http' as is (for bearer auth), not 'bearer'
                type_map = {
                    "http": "http",  # HTTP auth (including bearer)
                    "bearer": "http",  # Bearer is a type of HTTP auth
                    "apikey": "apiKey",
                    "api_key": "apiKey",
                    "oauth2": "oauth2",
                    "openidconnect": "openIdConnect",
                    "openid": "openIdConnect",
                }
                mapped_type = type_map.get(scheme_type, "http")

                # Preserve all fields from the original scheme data
                transformed_scheme = dict(scheme_data)
                transformed_scheme["type"] = mapped_type

                transformed_schemes[scheme_name] = transformed_scheme
            config["security_schemes"] = transformed_schemes

        # Remove fields that aren't in AgentRegistration model
        valid_fields = {
            "protocol_version",
            "name",
            "description",
            "path",
            "url",
            "version",
            "capabilities",
            "default_input_modes",
            "default_output_modes",
            "provider",
            "security_schemes",
            "skills",
            "tags",
            "visibility",
            "license",
        }
        config = {k: v for k, v in config.items() if k in valid_fields}

        agent = AgentRegistration(**config)
        client = _create_client(args)
        response = client.register_agent(agent)

        logger.info(
            f"Agent registered successfully: {response.agent.name} at {response.agent.path}"
        )
        print(
            json.dumps(
                {
                    "message": response.message,
                    "agent": {
                        "name": response.agent.name,
                        "path": response.agent.path,
                        "url": response.agent.url,
                        "num_skills": response.agent.num_skills,
                        "is_enabled": response.agent.is_enabled,
                    },
                },
                indent=2,
            )
        )
        return 0

    except Exception as e:
        logger.error(f"Agent registration failed: {e}")
        logger.debug(f"Full error details:", exc_info=True)
        return 1


def cmd_agent_list(args: argparse.Namespace) -> int:
    """
    List all A2A agents.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.list_agents(
            query=args.query if hasattr(args, "query") else None,
            enabled_only=args.enabled_only if hasattr(args, "enabled_only") else False,
            visibility=args.visibility if hasattr(args, "visibility") else None,
        )

        # Debug mode: print full JSON response
        if args.debug:
            logger.debug("Full JSON response from API:")
            print(json.dumps(response.model_dump(by_alias=True), indent=2, default=str))
            print()

        if not response.agents:
            logger.info("No agents found")
            return 0

        logger.info(f"Found {len(response.agents)} agents:\n")
        for agent in response.agents:
            status = "✓" if agent.is_enabled else "✗"
            print(f"{status} {agent.name} ({agent.path})")
            print(f"  {agent.description}")
            print()

        return 0

    except Exception as e:
        logger.error(f"List agents failed: {e}")
        return 1


def cmd_agent_get(args: argparse.Namespace) -> int:
    """
    Get detailed information about a specific agent.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        agent = client.get_agent(args.path)

        logger.info(f"Retrieved agent: {agent.name}")
        print(
            json.dumps(
                {
                    "name": agent.name,
                    "path": agent.path,
                    "description": agent.description,
                    "url": agent.url,
                    "version": agent.version,
                    "provider": agent.provider.model_dump() if agent.provider else None,
                    "is_enabled": agent.is_enabled,
                    "visibility": agent.visibility,
                    "skills": [
                        {"name": skill.name, "description": skill.description}
                        for skill in agent.skills
                    ],
                },
                indent=2,
            )
        )
        return 0

    except Exception as e:
        logger.error(f"Get agent failed: {e}")
        return 1


def cmd_agent_update(args: argparse.Namespace) -> int:
    """
    Update an existing agent.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1

        with open(config_path, "r") as f:
            config = json.load(f)

        # Convert skills list of dicts to Skill objects
        # Handle both 'input_schema' and 'parameters' field names
        skills = []
        for skill_data in config.get("skills", []):
            skill_dict = {
                "name": skill_data.get("name", skill_data.get("id", "")),
                "description": skill_data.get("description", ""),
            }
            if "input_schema" in skill_data:
                skill_dict["input_schema"] = skill_data["input_schema"]
            elif "parameters" in skill_data:
                skill_dict["input_schema"] = skill_data["parameters"]
            skills.append(Skill(**skill_dict))
        config["skills"] = skills

        # Convert provider string to enum with validation
        if "provider" in config:
            provider_value = config["provider"].lower()
            provider_map = {
                "anthropic": AgentProvider.ANTHROPIC,
                "custom": AgentProvider.CUSTOM,
                "other": AgentProvider.OTHER,
                "example corp": AgentProvider.CUSTOM,
                "example": AgentProvider.CUSTOM,
            }
            if provider_value in provider_map:
                config["provider"] = provider_map[provider_value]
            else:
                logger.warning(f"Unknown provider '{config['provider']}', using 'custom'")
                config["provider"] = AgentProvider.CUSTOM

        # Convert visibility string to enum if present
        if "visibility" in config:
            try:
                config["visibility"] = AgentVisibility(config["visibility"].lower())
            except ValueError:
                logger.warning(f"Unknown visibility '{config['visibility']}', using 'public'")
                config["visibility"] = AgentVisibility.PUBLIC

        # Handle security_schemes conversion
        if "security_schemes" in config:
            transformed_schemes = {}
            for scheme_name, scheme_data in config["security_schemes"].items():
                scheme_type = scheme_data.get("type", "").lower()
                type_map = {
                    "http": "bearer",
                    "bearer": "bearer",
                    "apikey": "api_key",
                    "api_key": "api_key",
                    "oauth2": "oauth2",
                }
                mapped_type = type_map.get(scheme_type, "bearer")
                transformed_schemes[scheme_name] = {
                    "type": mapped_type,
                    "description": scheme_data.get("description", ""),
                }
            config["security_schemes"] = transformed_schemes

        # Remove fields that aren't in AgentRegistration model
        valid_fields = {
            "name",
            "description",
            "path",
            "url",
            "version",
            "provider",
            "security_schemes",
            "skills",
            "tags",
            "visibility",
            "license",
        }
        config = {k: v for k, v in config.items() if k in valid_fields}

        agent = AgentRegistration(**config)
        client = _create_client(args)
        response = client.update_agent(args.path, agent)

        logger.info(f"Agent updated successfully: {response.name}")
        return 0

    except Exception as e:
        logger.error(f"Agent update failed: {e}")
        logger.debug(f"Full error details:", exc_info=True)
        return 1


def cmd_agent_delete(args: argparse.Namespace) -> int:
    """
    Delete an agent from the registry.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        if not args.force:
            confirmation = input(f"Delete agent {args.path}? (yes/no): ")
            if confirmation.lower() != "yes":
                logger.info("Operation cancelled")
                return 0

        client = _create_client(args)
        client.delete_agent(args.path)

        logger.info(f"Agent deleted successfully: {args.path}")
        return 0

    except Exception as e:
        logger.error(f"Agent deletion failed: {e}")
        return 1


def cmd_agent_toggle(args: argparse.Namespace) -> int:
    """
    Toggle agent enabled/disabled status.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.toggle_agent(args.path, args.enabled)

        logger.info(
            f"Agent {response.path} is now {'enabled' if response.is_enabled else 'disabled'}"
        )
        return 0

    except Exception as e:
        logger.error(f"Agent toggle failed: {e}")
        return 1


def cmd_agent_discover(args: argparse.Namespace) -> int:
    """
    Discover agents by required skills.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        skills = [s.strip() for s in args.skills.split(",")]
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None

        client = _create_client(args)
        response = client.discover_agents_by_skills(
            skills=skills, tags=tags, max_results=args.max_results
        )

        if not response.agents:
            logger.info("No agents found matching the required skills")
            return 0

        logger.info(f"Found {len(response.agents)} matching agents:\n")
        for agent in response.agents:
            print(f"{agent.name} ({agent.path})")
            print(f"  Relevance: {agent.relevance_score:.2%}")
            print(f"  Matching skills: {', '.join(agent.matching_skills)}")
            print()

        return 0

    except Exception as e:
        logger.error(f"Agent discovery failed: {e}")
        return 1


def cmd_agent_search(args: argparse.Namespace) -> int:
    """
    Perform semantic search for agents.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.discover_agents_semantic(query=args.query, max_results=args.max_results)

        if not response.agents:
            if args.json:
                print(json.dumps({"agents": [], "query": args.query}, indent=2))
            else:
                logger.info("No agents found matching the query")
            return 0

        if args.json:
            # Output full JSON response
            output = {
                "query": args.query,
                "agents": [agent.model_dump() for agent in response.agents],
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            # Human-readable output
            logger.info(f"Found {len(response.agents)} matching agents:\n")
            for agent in response.agents:
                print(f"{agent.name} ({agent.path})")
                print(f"  Relevance: {agent.relevance_score:.2%}")
                print(f"  {agent.description[:100]}...")
                print()

        return 0

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return 1


def cmd_agent_rate(args: argparse.Namespace) -> int:
    """
    Rate an agent (1-5 stars).

    Args:
        args: Command arguments with path and rating

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response: RatingResponse = client.rate_agent(path=args.path, rating=args.rating)

        logger.info(f"✓ {response.message}")
        logger.info(f"Average rating: {response.average_rating:.2f} stars")

        return 0

    except Exception as e:
        logger.error(f"Failed to rate agent: {e}")
        return 1


def cmd_agent_rating(args: argparse.Namespace) -> int:
    """
    Get rating information for an agent.

    Args:
        args: Command arguments with path

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response: RatingInfoResponse = client.get_agent_rating(path=args.path)

        logger.info(f"\nRating for agent '{args.path}':")
        logger.info(f"  Average: {response.num_stars:.2f} stars")
        logger.info(f"  Total ratings: {len(response.rating_details)}")

        if response.rating_details:
            logger.info("\nIndividual ratings (most recent):")
            # Show first 10 ratings
            for detail in response.rating_details[:10]:
                logger.info(f"  {detail.user}: {detail.rating} stars")

            if len(response.rating_details) > 10:
                logger.info(f"  ... and {len(response.rating_details) - 10} more")

        return 0

    except Exception as e:
        logger.error(f"Failed to get ratings: {e}")
        return 1


def cmd_agent_security_scan(args: argparse.Namespace) -> int:
    """
    Get security scan results for an agent.

    Args:
        args: Command arguments with path and optional json flag

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response: AgentSecurityScanResponse = client.get_agent_security_scan(path=args.path)

        # Always output as JSON since the response structure is complex
        print(json.dumps(response.model_dump(), indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Failed to get security scan results: {e}")
        return 1


def cmd_agent_rescan(args: argparse.Namespace) -> int:
    """
    Trigger manual security scan for an agent (admin only).

    Args:
        args: Command arguments with path and optional json flag

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response: AgentRescanResponse = client.rescan_agent(path=args.path)

        if hasattr(args, "json") and args.json:
            # Output raw JSON
            print(json.dumps(response.model_dump(), indent=2, default=str))
        else:
            # Pretty print results
            safety_status = "SAFE" if response.is_safe else "UNSAFE"
            logger.info(f"\nSecurity scan completed for agent '{args.path}':")
            logger.info(f"  Status: {safety_status}")
            logger.info(f"  Scan timestamp: {response.scan_timestamp}")
            logger.info(f"  Analyzers used: {', '.join(response.analyzers_used)}")
            logger.info(f"\n  Severity counts:")
            logger.info(f"    Critical: {response.critical_issues}")
            logger.info(f"    High: {response.high_severity}")
            logger.info(f"    Medium: {response.medium_severity}")
            logger.info(f"    Low: {response.low_severity}")

            if response.output_file:
                logger.info(f"\n  Output file: {response.output_file}")

            if response.scan_failed:
                logger.error(f"\n  Scan failed: {response.error_message}")
                return 1

            if not response.is_safe:
                logger.warning("\n  WARNING: Agent flagged as potentially unsafe!")

        return 0

    except Exception as e:
        logger.error(f"Failed to trigger security scan: {e}")
        return 1


# ==========================================
# Agent Skills Command Handlers
# ==========================================


def cmd_skill_register(args: argparse.Namespace) -> int:
    """
    Register a new Agent Skill.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        request = SkillRegistrationRequest(
            name=args.name,
            skill_md_url=args.url,
            description=args.description if hasattr(args, "description") else None,
            tags=args.tags.split(",") if hasattr(args, "tags") and args.tags else [],
            visibility=args.visibility if hasattr(args, "visibility") else "public",
        )

        client = _create_client(args)
        skill = client.register_skill(request)

        logger.info(f"Skill registered successfully: {skill.name} at {skill.path}")
        print(
            json.dumps(
                {
                    "message": "Skill registered successfully",
                    "skill": {
                        "name": skill.name,
                        "path": skill.path,
                        "description": skill.description,
                        "skill_md_url": skill.skill_md_url,
                        "is_enabled": skill.is_enabled,
                    },
                },
                indent=2,
            )
        )
        return 0

    except Exception as e:
        logger.error(f"Skill registration failed: {e}")
        return 1


def cmd_skill_list(args: argparse.Namespace) -> int:
    """
    List all Agent Skills.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.list_skills(
            include_disabled=args.include_disabled if hasattr(args, "include_disabled") else False,
            tag=args.tag if hasattr(args, "tag") else None,
        )

        if args.debug:
            print(json.dumps([s.model_dump() for s in response.skills], indent=2, default=str))
            return 0

        if not response.skills:
            logger.info("No skills found")
            return 0

        logger.info(f"Found {len(response.skills)} skills:\n")
        for skill in response.skills:
            status = "[+]" if skill.is_enabled else "[-]"
            health = f"({skill.health_status})" if skill.health_status else ""
            print(f"{status} {skill.name} {health}")
            print(f"    Path: {skill.path}")
            if skill.description:
                print(f"    {skill.description[:80]}...")
            if skill.tags:
                print(f"    Tags: {', '.join(skill.tags)}")
            print()

        return 0

    except Exception as e:
        logger.error(f"List skills failed: {e}")
        return 1


def cmd_skill_get(args: argparse.Namespace) -> int:
    """
    Get details for a specific skill.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        skill = client.get_skill(args.path)

        logger.info(f"Retrieved skill: {skill.name}")
        print(
            json.dumps(
                {
                    "name": skill.name,
                    "path": skill.path,
                    "description": skill.description,
                    "skill_md_url": skill.skill_md_url,
                    "skill_md_raw_url": skill.skill_md_raw_url,
                    "version": skill.version,
                    "author": skill.author,
                    "visibility": skill.visibility,
                    "is_enabled": skill.is_enabled,
                    "tags": skill.tags,
                    "owner": skill.owner,
                    "num_stars": skill.num_stars,
                    "health_status": skill.health_status,
                    "created_at": skill.created_at,
                    "updated_at": skill.updated_at,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    except Exception as e:
        logger.error(f"Get skill failed: {e}")
        return 1


def cmd_skill_delete(args: argparse.Namespace) -> int:
    """
    Delete a skill.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        client.delete_skill(args.path)

        logger.info(f"Skill deleted: {args.path}")
        print(json.dumps({"message": "Skill deleted successfully", "path": args.path}, indent=2))
        return 0

    except Exception as e:
        logger.error(f"Delete skill failed: {e}")
        return 1


def cmd_skill_toggle(args: argparse.Namespace) -> int:
    """
    Toggle skill enabled/disabled state.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.toggle_skill(args.path, args.enable)

        state = "enabled" if response.is_enabled else "disabled"
        logger.info(f"Skill {state}: {response.path}")
        print(json.dumps({"path": response.path, "is_enabled": response.is_enabled}, indent=2))
        return 0

    except Exception as e:
        logger.error(f"Toggle skill failed: {e}")
        return 1


def cmd_skill_health(args: argparse.Namespace) -> int:
    """
    Check skill health (SKILL.md accessibility).

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.check_skill_health(args.path)

        status = "HEALTHY" if response.healthy else "UNHEALTHY"
        logger.info(f"Skill health: {status}")
        print(
            json.dumps(
                {
                    "path": response.path,
                    "healthy": response.healthy,
                    "status_code": response.status_code,
                    "error": response.error,
                    "response_time_ms": response.response_time_ms,
                },
                indent=2,
            )
        )
        return 0 if response.healthy else 1

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return 1


def cmd_skill_content(args: argparse.Namespace) -> int:
    """
    Get SKILL.md content for a skill.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.get_skill_content(args.path)

        if hasattr(args, "raw") and args.raw:
            # Output raw content only
            print(response.content)
        else:
            logger.info(f"Retrieved content from: {response.url}")
            print(f"--- SKILL.md ({len(response.content)} chars) ---")
            print(response.content)
            print("--- END ---")

        return 0

    except Exception as e:
        logger.error(f"Get content failed: {e}")
        return 1


def cmd_skill_search(args: argparse.Namespace) -> int:
    """
    Search for skills.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.search_skills(
            query=args.query, tags=args.tags if hasattr(args, "tags") else None
        )

        if args.debug:
            print(json.dumps(response.model_dump(), indent=2, default=str))
            return 0

        logger.info(f"Found {response.total_count} skills matching '{args.query}':\n")
        for skill in response.skills:
            print(f"  {skill.get('name')} ({skill.get('path')})")
            if skill.get("description"):
                print(f"      {skill.get('description')[:60]}...")
            print(f"      Score: {skill.get('relevance_score', 0):.2f}")
            print()

        return 0

    except Exception as e:
        logger.error(f"Search skills failed: {e}")
        return 1


def cmd_skill_rate(args: argparse.Namespace) -> int:
    """
    Rate a skill (1-5 stars).

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        if not 1 <= args.rating <= 5:
            logger.error("Rating must be between 1 and 5")
            return 1

        client = _create_client(args)
        response = client.rate_skill(args.path, args.rating)

        logger.info(f"Skill rated: {args.rating} stars")
        print(
            json.dumps(
                {
                    "message": response.get("message"),
                    "average_rating": response.get("average_rating"),
                },
                indent=2,
            )
        )
        return 0

    except Exception as e:
        logger.error(f"Rate skill failed: {e}")
        return 1


def cmd_skill_rating(args: argparse.Namespace) -> int:
    """
    Get rating information for a skill.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.get_skill_rating(args.path)

        logger.info(f"Skill rating: {response.num_stars} stars")
        print(
            json.dumps(
                {
                    "num_stars": response.num_stars,
                    "rating_details": response.rating_details,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    except Exception as e:
        logger.error(f"Get rating failed: {e}")
        return 1


def cmd_skill_security_scan(args: argparse.Namespace) -> int:
    """
    Get security scan results for a skill.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.get_skill_security_scan(path=args.path)

        print(json.dumps(response.model_dump(), indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Failed to get security scan results: {e}")
        return 1


def cmd_skill_rescan(args: argparse.Namespace) -> int:
    """
    Trigger manual security scan for a skill (admin only).

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.rescan_skill(path=args.path)

        if not args.json_output:
            safety_status = "SAFE" if response.is_safe else "UNSAFE"
            logger.info(f"\nSecurity scan completed for skill '{args.path}':")
            logger.info(f"  Status: {safety_status}")
            logger.info(f"  Critical: {response.critical_issues}")
            logger.info(f"  High: {response.high_severity}")
            logger.info(f"  Medium: {response.medium_severity}")
            logger.info(f"  Low: {response.low_severity}")
            logger.info(f"  Analyzers: {', '.join(response.analyzers_used)}")

        print(json.dumps(response.model_dump(), indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Failed to trigger security scan: {e}")
        return 1


def cmd_anthropic_list_servers(args: argparse.Namespace) -> int:
    """
    List all servers using Anthropic Registry API v0.1.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        result: AnthropicServerList = client.anthropic_list_servers(limit=args.limit)

        # Print raw JSON if requested
        if args.raw:
            print(json.dumps(result.model_dump(), indent=2, default=str))
            return 0

        logger.info(f"Retrieved {len(result.servers)} servers\n")

        if result.metadata:
            logger.info(f"Next cursor: {result.metadata.nextCursor}")
            logger.info(f"Count: {result.metadata.count}\n")

        # Print server details
        for idx, server_response in enumerate(result.servers, 1):
            server = server_response.server
            print(f"{idx}. {server.name}")
            print(f"   Title: {server.title or 'N/A'}")
            print(f"   Description: {server.description[:100]}...")
            print(f"   Version: {server.version}")
            print(f"   Website: {server.websiteUrl or 'N/A'}")

            if server.repository:
                print(f"   Repository: {server.repository.url}")

            if server.packages:
                print(f"   Packages: {len(server.packages)} package(s)")
            print()

        return 0

    except Exception as e:
        logger.error(f"Failed to list servers: {e}")
        return 1


def cmd_anthropic_list_versions(args: argparse.Namespace) -> int:
    """
    List versions for a specific server using Anthropic Registry API v0.1.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        result: AnthropicServerList = client.anthropic_list_server_versions(
            server_name=args.server_name
        )

        # Print raw JSON if requested
        if args.raw:
            print(json.dumps(result.model_dump(), indent=2, default=str))
            return 0

        logger.info(f"Found {len(result.servers)} version(s) for {args.server_name}\n")

        for idx, server_response in enumerate(result.servers, 1):
            server = server_response.server
            print(f"{idx}. Version {server.version}")
            print(f"   Name: {server.name}")
            print(f"   Description: {server.description[:100]}...")
            print()

        return 0

    except Exception as e:
        logger.error(f"Failed to list server versions: {e}")
        return 1


def cmd_anthropic_get_server(args: argparse.Namespace) -> int:
    """
    Get detailed information about a specific server version using Anthropic Registry API v0.1.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        result: AnthropicServerResponse = client.anthropic_get_server_version(
            server_name=args.server_name,
            version=args.version,
        )

        # Print raw JSON if requested
        if args.raw:
            print(json.dumps(result.model_dump(), indent=2, default=str))
            return 0

        server = result.server

        print(f"\nServer: {server.name}")
        print(f"Title: {server.title or 'N/A'}")
        print(f"Version: {server.version}")
        print(f"Description: {server.description}")
        print(f"Website: {server.websiteUrl or 'N/A'}")

        if server.repository:
            print(f"\nRepository:")
            print(f"  URL: {server.repository.url}")
            print(f"  Source: {server.repository.source}")
            if server.repository.id:
                print(f"  ID: {server.repository.id}")
            if server.repository.subfolder:
                print(f"  Subfolder: {server.repository.subfolder}")

        if server.packages:
            print(f"\nPackages ({len(server.packages)}):")
            for idx, package in enumerate(server.packages, 1):
                print(f"  {idx}. {package.registryType}: {package.identifier}")
                print(f"     Version: {package.version}")
                if package.runtimeHint:
                    print(f"     Runtime: {package.runtimeHint}")

        if server.meta:
            print(f"\nMetadata:")
            print(json.dumps(server.meta, indent=2))

        if result.meta:
            print(f"\nRegistry Metadata:")
            print(json.dumps(result.meta, indent=2))

        return 0

    except Exception as e:
        logger.error(f"Failed to get server version: {e}")
        return 1


# User Management Command Handlers (Management API)


def cmd_user_list(args: argparse.Namespace) -> int:
    """
    List Keycloak users.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.list_users(
            search=args.search if hasattr(args, "search") and args.search else None,
            limit=args.limit if hasattr(args, "limit") else 500,
        )

        if not response.users:
            logger.info("No users found")
            return 0

        logger.info(f"Found {response.total} users\n")

        for user in response.users:
            enabled_icon = "✓" if user.enabled else "✗"
            print(f"{enabled_icon} {user.username} (ID: {user.id})")
            print(f"  Email: {user.email or 'N/A'}")
            if user.firstName or user.lastName:
                name = f"{user.firstName or ''} {user.lastName or ''}".strip()
                print(f"  Name: {name}")
            print(f"  Groups: {', '.join(user.groups) if user.groups else 'None'}")
            print(f"  Enabled: {user.enabled}")
            print()

        return 0

    except Exception as e:
        logger.error(f"List users failed: {e}")
        return 1


def cmd_user_create_m2m(args: argparse.Namespace) -> int:
    """
    Create M2M service account.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        groups = [g.strip() for g in args.groups.split(",")]
        client = _create_client(args)
        result = client.create_m2m_account(
            name=args.name,
            groups=groups,
            description=args.description
            if hasattr(args, "description") and args.description
            else None,
        )

        logger.info(f"M2M account created successfully\n")
        print(f"Client ID: {result.client_id}")
        print(f"Client Secret: {result.client_secret}")
        print(f"Groups: {', '.join(result.groups)}")
        if result.service_principal_id:
            print(f"Service Principal ID: {result.service_principal_id}")
        print()
        print("IMPORTANT: Save the client secret securely - it cannot be retrieved later.")

        return 0

    except Exception as e:
        logger.error(f"Create M2M account failed: {e}")
        return 1


def cmd_user_create_human(args: argparse.Namespace) -> int:
    """
    Create human user account.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        groups = [g.strip() for g in args.groups.split(",")]
        client = _create_client(args)
        result = client.create_human_user(
            username=args.username,
            email=args.email,
            first_name=args.first_name,
            last_name=args.last_name,
            groups=groups,
            password=args.password if hasattr(args, "password") and args.password else None,
        )

        logger.info(f"User created successfully\n")
        print(f"Username: {result.username}")
        print(f"User ID: {result.id}")
        print(f"Email: {result.email or 'N/A'}")
        if result.firstName or result.lastName:
            name = f"{result.firstName or ''} {result.lastName or ''}".strip()
            print(f"Name: {name}")
        print(f"Groups: {', '.join(result.groups)}")
        print(f"Enabled: {result.enabled}")

        return 0

    except Exception as e:
        logger.error(f"Create user failed: {e}")
        return 1


def cmd_user_delete(args: argparse.Namespace) -> int:
    """
    Delete a user.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        if not args.force:
            confirmation = input(f"Delete user '{args.username}'? (yes/no): ")
            if confirmation.lower() != "yes":
                logger.info("Operation cancelled")
                return 0

        client = _create_client(args)
        result = client.delete_user(args.username)

        logger.info(f"User '{result.username}' deleted successfully")
        return 0

    except Exception as e:
        logger.error(f"Delete user failed: {e}")
        return 1


def cmd_group_create(args: argparse.Namespace) -> int:
    """
    Create a new IAM group.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        result = client.create_keycloak_group(name=args.name, description=args.description)

        logger.info(f"IAM group created successfully: {result.name}")
        print(f"\nGroup: {result.name}")
        print(f"  ID: {result.id}")
        print(f"  Path: {result.path}")
        if result.attributes:
            print(f"  Attributes: {json.dumps(result.attributes, indent=4)}")

        return 0

    except Exception as e:
        logger.error(f"Create IAM group failed: {e}")
        return 1


def cmd_group_delete(args: argparse.Namespace) -> int:
    """
    Delete an IAM group.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        if not args.force:
            confirmation = input(f"Delete IAM group '{args.name}'? (yes/no): ")
            if confirmation.lower() != "yes":
                logger.info("Operation cancelled")
                return 0

        client = _create_client(args)
        result = client.delete_keycloak_group(name=args.name)

        logger.info(f"IAM group deleted successfully: {result.name}")
        return 0

    except Exception as e:
        logger.error(f"Delete IAM group failed: {e}")
        return 1


def cmd_group_list(args: argparse.Namespace) -> int:
    """
    List IAM groups.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.list_keycloak_iam_groups()

        if not response.groups:
            logger.info("No IAM groups found")
            return 0

        logger.info(f"Found {response.total} IAM groups:\n")

        for group in response.groups:
            print(f"Group: {group['name']}")
            print(f"  ID: {group['id']}")
            print(f"  Path: {group['path']}")
            if group.get("attributes"):
                print(f"  Attributes: {json.dumps(group['attributes'], indent=4)}")
            print()

        return 0

    except Exception as e:
        logger.error(f"List IAM groups failed: {e}")
        return 1


def cmd_federation_get(args: argparse.Namespace) -> int:
    """
    Get federation configuration.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        config = client.get_federation_config(config_id=args.config_id)

        print(json.dumps(config, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Get federation config failed: {e}")
        return 1


def cmd_federation_save(args: argparse.Namespace) -> int:
    """
    Save federation configuration from JSON file.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)

        # Load config from file
        with open(args.config, "r") as f:
            config_data = json.load(f)

        response = client.save_federation_config(config=config_data, config_id=args.config_id)

        logger.info(f"Federation config saved successfully: {args.config_id}")
        print(json.dumps(response, indent=2, default=str))
        return 0

    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        return 1
    except Exception as e:
        logger.error(f"Save federation config failed: {e}")
        return 1


def cmd_federation_delete(args: argparse.Namespace) -> int:
    """
    Delete federation configuration.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)

        if not args.force:
            confirm = input(f"Delete federation config '{args.config_id}'? (y/N): ")
            if confirm.lower() != "y":
                logger.info("Cancelled")
                return 0

        response = client.delete_federation_config(config_id=args.config_id)

        logger.info(f"Federation config deleted: {args.config_id}")
        print(json.dumps(response, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Delete federation config failed: {e}")
        return 1


def cmd_federation_list(args: argparse.Namespace) -> int:
    """
    List all federation configurations.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.list_federation_configs()

        if args.json:
            # Output raw JSON
            print(json.dumps(response, indent=2, default=str))
            return 0

        if not response.get("configs"):
            logger.info("No federation configs found")
            return 0

        logger.info(f"Found {response.get('total', 0)} federation configs:\n")

        for config in response["configs"]:
            print(f"Config ID: {config.get('id')}")
            print(f"  Created: {config.get('created_at')}")
            print(f"  Updated: {config.get('updated_at')}")
            print()

        return 0

    except Exception as e:
        logger.error(f"List federation configs failed: {e}")
        return 1


def cmd_federation_add_anthropic_server(args: argparse.Namespace) -> int:
    """
    Add Anthropic server to federation config.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.add_anthropic_server(
            server_name=args.server_name, config_id=args.config_id
        )

        logger.info(f"Anthropic server added: {args.server_name}")
        print(json.dumps(response, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Add Anthropic server failed: {e}")
        return 1


def cmd_federation_remove_anthropic_server(args: argparse.Namespace) -> int:
    """
    Remove Anthropic server from federation config.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.remove_anthropic_server(
            server_name=args.server_name, config_id=args.config_id
        )

        logger.info(f"Anthropic server removed: {args.server_name}")
        print(json.dumps(response, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Remove Anthropic server failed: {e}")
        return 1


def cmd_federation_add_asor_agent(args: argparse.Namespace) -> int:
    """
    Add ASOR agent to federation config.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.add_asor_agent(agent_id=args.agent_id, config_id=args.config_id)

        logger.info(f"ASOR agent added: {args.agent_id}")
        print(json.dumps(response, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Add ASOR agent failed: {e}")
        return 1


def cmd_federation_remove_asor_agent(args: argparse.Namespace) -> int:
    """
    Remove ASOR agent from federation config.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.remove_asor_agent(agent_id=args.agent_id, config_id=args.config_id)

        logger.info(f"ASOR agent removed: {args.agent_id}")
        print(json.dumps(response, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Remove ASOR agent failed: {e}")
        return 1


def cmd_federation_sync(args: argparse.Namespace) -> int:
    """
    Trigger manual federation sync to import servers/agents.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.sync_federation(config_id=args.config_id, source=args.source)

        if args.json:
            # Output raw JSON
            print(json.dumps(response, indent=2, default=str))
        else:
            # Formatted output
            logger.info(f"Federation sync completed: {response.get('message')}")
            print(f"\nSync Results:")
            print(f"  Config ID: {response.get('config_id')}")
            print(f"  Total Synced: {response.get('total_synced', 0)}")

            results = response.get("results", {})
            if results.get("anthropic", {}).get("count", 0) > 0:
                print(f"\n  Anthropic Servers ({results['anthropic']['count']}):")
                for server in results["anthropic"].get("servers", []):
                    print(f"    - {server}")

            if results.get("asor", {}).get("count", 0) > 0:
                print(f"\n  ASOR Agents ({results['asor']['count']}):")
                for agent in results["asor"].get("agents", []):
                    print(f"    - {agent}")

        return 0

    except Exception as e:
        logger.error(f"Federation sync failed: {e}")
        return 1


def cmd_peer_list(args: argparse.Namespace) -> int:
    """
    List all configured peer registries.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)

        enabled_filter = None
        if hasattr(args, "enabled_only") and args.enabled_only:
            enabled_filter = True

        response = client.list_peers(enabled=enabled_filter)

        if args.json:
            masked_response = _mask_sensitive_fields(response)
            print(json.dumps(masked_response, indent=2, default=str))
            return 0

        peers = response if isinstance(response, list) else response.get("peers", [])

        if not peers:
            logger.info("No peer registries configured")
            return 0

        logger.info(f"Found {len(peers)} peer registries:\n")

        for peer in peers:
            status = "enabled" if peer.get("enabled") else "disabled"
            print(f"  Peer ID:   {peer.get('peer_id')}")
            print(f"  Name:      {peer.get('name')}")
            print(f"  Endpoint:  {peer.get('endpoint')}")
            print(f"  Status:    {status}")
            print(f"  Sync Mode: {peer.get('sync_mode', 'all')}")
            print()

        return 0

    except Exception as e:
        logger.error(f"List peers failed: {e}")
        return 1


def cmd_peer_add(args: argparse.Namespace) -> int:
    """
    Add a new peer registry from a JSON config file.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)

        with open(args.config, "r") as f:
            config_data = json.load(f)

        # Override federation_token from CLI arg if provided
        if hasattr(args, "federation_token") and args.federation_token:
            config_data["federation_token"] = args.federation_token

        response = client.add_peer(config=config_data)

        logger.info(f"Peer registry added successfully: {config_data.get('peer_id')}")
        masked_response = _mask_sensitive_fields(response)
        print(json.dumps(masked_response, indent=2, default=str))
        return 0

    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        return 1
    except Exception as e:
        logger.error(f"Add peer failed: {e}")
        return 1


def cmd_peer_get(args: argparse.Namespace) -> int:
    """
    Get details of a specific peer registry.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.get_peer(peer_id=args.peer_id)

        if args.json:
            masked_response = _mask_sensitive_fields(response)
            print(json.dumps(masked_response, indent=2, default=str))
            return 0

        print(f"Peer ID:      {response.get('peer_id')}")
        print(f"Name:         {response.get('name')}")
        print(f"Endpoint:     {response.get('endpoint')}")
        print(f"Enabled:      {response.get('enabled')}")
        print(f"Sync Mode:    {response.get('sync_mode', 'all')}")
        print(f"Created:      {response.get('created_at')}")
        print(f"Updated:      {response.get('updated_at')}")

        # Mask federation token in non-JSON output
        fed_token = response.get("federation_token")
        if fed_token:
            masked_token = f"{fed_token[:3]}..." if len(fed_token) > 3 else "***"
            print(f"Fed Token:    {masked_token}")

        whitelist_servers = response.get("whitelist_servers", [])
        if whitelist_servers:
            print(f"Whitelist:    {', '.join(whitelist_servers)}")

        tag_filter = response.get("tag_filter", [])
        if tag_filter:
            print(f"Tag Filter:   {', '.join(tag_filter)}")

        return 0

    except Exception as e:
        logger.error(f"Get peer failed: {e}")
        return 1


def cmd_peer_update(args: argparse.Namespace) -> int:
    """
    Update an existing peer registry from a JSON config file.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)

        with open(args.config, "r") as f:
            config_data = json.load(f)

        # Override federation_token from CLI arg if provided
        if hasattr(args, "federation_token") and args.federation_token:
            config_data["federation_token"] = args.federation_token

        response = client.update_peer(peer_id=args.peer_id, config=config_data)

        logger.info(f"Peer registry updated successfully: {args.peer_id}")
        masked_response = _mask_sensitive_fields(response)
        print(json.dumps(masked_response, indent=2, default=str))
        return 0

    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        return 1
    except Exception as e:
        logger.error(f"Update peer failed: {e}")
        return 1


def cmd_peer_update_token(args: argparse.Namespace) -> int:
    """
    Update only the federation token for a peer registry.

    This command is useful for:
    - Recovering from token loss (issue #561)
    - Rotating federation tokens without modifying other peer config
    - Fixing authentication issues after peer updates

    Args:
        args: Command arguments with peer_id and federation_token

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)

        response = client.update_peer_token(
            peer_id=args.peer_id, federation_token=args.federation_token
        )

        logger.info(f"Federation token updated successfully for peer: {args.peer_id}")
        print(json.dumps(response, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Update peer token failed: {e}")
        return 1


def cmd_peer_remove(args: argparse.Namespace) -> int:
    """
    Remove a peer registry.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        if not args.force:
            confirm = input(f"Remove peer registry '{args.peer_id}'? (y/N): ")
            if confirm.lower() != "y":
                logger.info("Cancelled")
                return 0

        client = _create_client(args)
        response = client.remove_peer(peer_id=args.peer_id)

        logger.info(f"Peer registry removed: {args.peer_id}")
        masked_response = _mask_sensitive_fields(response)
        print(json.dumps(masked_response, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Remove peer failed: {e}")
        return 1


def cmd_peer_sync(args: argparse.Namespace) -> int:
    """
    Trigger sync from a specific peer registry.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.sync_peer(peer_id=args.peer_id)

        if args.json:
            print(json.dumps(response, indent=2, default=str))
            return 0

        # Check success field from SyncResult model
        success = response.get("success", False)
        status_text = "SUCCESS" if success else "FAILED"

        print(f"\nSync Results for peer '{args.peer_id}':")
        print(f"  Status:           {status_text}")
        print(f"  Servers Synced:   {response.get('servers_synced', 0)}")
        print(f"  Agents Synced:    {response.get('agents_synced', 0)}")
        print(f"  Servers Orphaned: {response.get('servers_orphaned', 0)}")
        print(f"  Agents Orphaned:  {response.get('agents_orphaned', 0)}")

        # SyncResult has 'error_message' (singular), not 'errors' (plural)
        error_msg = response.get("error_message")
        if error_msg:
            print(f"\n  Error:")
            print(f"    {error_msg}")

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Peer sync failed: {e}")
        return 1


def cmd_peer_sync_all(args: argparse.Namespace) -> int:
    """
    Trigger sync from all enabled peer registries.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.sync_all_peers()

        if args.json:
            print(json.dumps(response, indent=2, default=str))
            return 0

        results = response if isinstance(response, list) else response.get("results", [])
        print(f"\nSync All Peers Results:")
        print(f"  Total peers synced: {len(results)}")

        for result in results:
            peer_id = result.get("peer_id", "unknown")
            status = result.get("status", "unknown")
            print(f"\n  {peer_id}: {status}")
            print(f"    Servers: {result.get('servers_synced', 0)}")
            print(f"    Agents:  {result.get('agents_synced', 0)}")

        return 0

    except Exception as e:
        logger.error(f"Sync all peers failed: {e}")
        return 1


def cmd_peer_status(args: argparse.Namespace) -> int:
    """
    Get sync status for a specific peer registry.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.get_peer_status(peer_id=args.peer_id)

        if args.json:
            print(json.dumps(response, indent=2, default=str))
            return 0

        print(f"\nSync Status for peer '{args.peer_id}':")

        # Determine last sync status from history or health
        history = response.get("sync_history", [])
        if history:
            last_entry = history[0]
            last_status = "success" if last_entry.get("success") else "failed"
            last_time = last_entry.get("completed_at") or last_entry.get("started_at")
        else:
            last_status = "never"
            last_time = response.get("last_successful_sync") or response.get("last_sync_attempt")

        print(f"  Last Sync Status:  {last_status}")
        print(f"  Last Sync Time:    {last_time or 'never'}")
        print(f"  Last Generation:   {response.get('current_generation', 0)}")
        print(f"  Servers Synced:    {response.get('total_servers_synced', 0)}")
        print(f"  Agents Synced:     {response.get('total_agents_synced', 0)}")
        print(f"  Is Healthy:        {response.get('is_healthy', False)}")

        if history:
            print(f"\n  Recent Sync History ({len(history)} entries):")
            for entry in history[:5]:
                entry_status = "success" if entry.get("success") else "failed"
                entry_time = entry.get("completed_at") or entry.get("started_at")
                print(f"    {entry_time} - {entry_status}")
                print(
                    f"      Servers: {entry.get('servers_synced', 0)}, "
                    f"Agents: {entry.get('agents_synced', 0)}"
                )

        return 0

    except Exception as e:
        logger.error(f"Get peer status failed: {e}")
        return 1


def cmd_peer_enable(args: argparse.Namespace) -> int:
    """
    Enable a peer registry.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.enable_peer(peer_id=args.peer_id)

        logger.info(f"Peer registry enabled: {args.peer_id}")
        masked_response = _mask_sensitive_fields(response)
        print(json.dumps(masked_response, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Enable peer failed: {e}")
        return 1


def cmd_peer_disable(args: argparse.Namespace) -> int:
    """
    Disable a peer registry.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.disable_peer(peer_id=args.peer_id)

        logger.info(f"Peer registry disabled: {args.peer_id}")
        masked_response = _mask_sensitive_fields(response)
        print(json.dumps(masked_response, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Disable peer failed: {e}")
        return 1


def cmd_peer_connections(args: argparse.Namespace) -> int:
    """
    Get all federation connections across all peers.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.get_peer_connections()

        if args.json:
            print(json.dumps(response, indent=2, default=str))
            return 0

        connections = response if isinstance(response, list) else response.get("connections", [])

        if not connections:
            logger.info("No federation connections found")
            return 0

        logger.info(f"Found {len(connections)} federation connections:\n")
        for conn in connections:
            print(f"  Peer: {conn.get('peer_id')}")
            print(f"  Direction: {conn.get('direction', 'unknown')}")
            print(f"  Status: {conn.get('status', 'unknown')}")
            print()

        return 0

    except Exception as e:
        logger.error(f"Get peer connections failed: {e}")
        return 1


def cmd_peer_shared_resources(args: argparse.Namespace) -> int:
    """
    Get resource sharing summary across all peers.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.get_shared_resources()

        if args.json:
            print(json.dumps(response, indent=2, default=str))
            return 0

        print("\nShared Resources Summary:")
        print(json.dumps(response, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Get shared resources failed: {e}")
        return 1


# ==========================================
# Virtual MCP Server Command Handlers
# ==========================================


def cmd_vs_create(args: argparse.Namespace) -> int:
    """
    Create a virtual MCP server from JSON config.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)

        # Load config from file
        with open(args.config, "r") as f:
            config_data = json.load(f)

        # Build tool mappings
        tool_mappings = []
        for mapping in config_data.get("tool_mappings", []):
            tool_mappings.append(
                ToolMapping(
                    tool_name=mapping["tool_name"],
                    alias=mapping.get("alias"),
                    backend_server_path=mapping["backend_server_path"],
                    backend_version=mapping.get("backend_version"),
                    description_override=mapping.get("description_override"),
                )
            )

        # Build tool scope overrides
        tool_scope_overrides = []
        for override in config_data.get("tool_scope_overrides", []):
            tool_scope_overrides.append(
                ToolScopeOverride(
                    tool_alias=override["tool_alias"],
                    required_scopes=override.get("required_scopes", []),
                )
            )

        request = VirtualServerCreateRequest(
            path=config_data["path"],
            server_name=config_data["server_name"],
            description=config_data.get("description"),
            tool_mappings=tool_mappings,
            required_scopes=config_data.get("required_scopes", []),
            tool_scope_overrides=tool_scope_overrides,
            tags=config_data.get("tags", []),
            supported_transports=config_data.get("supported_transports", ["streamable-http"]),
            is_enabled=config_data.get("is_enabled", True),
        )

        result = client.create_virtual_server(request)

        logger.info(f"Virtual server created: {result.path}")
        print(
            json.dumps(
                {
                    "message": "Virtual server created successfully",
                    "virtual_server": {
                        "path": result.path,
                        "server_name": result.server_name,
                        "description": result.description,
                        "is_enabled": result.is_enabled,
                        "tool_count": len(result.tool_mappings),
                    },
                },
                indent=2,
            )
        )
        return 0

    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        return 1
    except KeyError as e:
        logger.error(f"Missing required field in config: {e}")
        return 1
    except Exception as e:
        logger.error(f"Create virtual server failed: {e}")
        return 1


def cmd_vs_list(args: argparse.Namespace) -> int:
    """
    List virtual MCP servers.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        response = client.list_virtual_servers(
            enabled_only=args.enabled_only if hasattr(args, "enabled_only") else False,
            tag=args.tag if hasattr(args, "tag") else None,
        )

        if args.json:
            print(json.dumps(response.model_dump(), indent=2, default=str))
            return 0

        print(f"\nVirtual MCP Servers ({response.total} total):")
        print("-" * 80)

        for vs in response.virtual_servers:
            status = "enabled" if vs.is_enabled else "disabled"
            tool_count = len(vs.tool_mappings)
            print(f"  {vs.path}")
            print(f"    Name: {vs.server_name}")
            print(f"    Status: {status}")
            print(f"    Tools: {tool_count}")
            if vs.description:
                print(f"    Description: {vs.description[:60]}...")
            if vs.tags:
                print(f"    Tags: {', '.join(vs.tags)}")
            print()

        return 0

    except Exception as e:
        logger.error(f"List virtual servers failed: {e}")
        return 1


def cmd_vs_get(args: argparse.Namespace) -> int:
    """
    Get virtual MCP server details.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        result = client.get_virtual_server(args.path)

        if args.json:
            print(json.dumps(result.model_dump(), indent=2, default=str))
            return 0

        print(f"\nVirtual MCP Server: {result.path}")
        print("-" * 60)
        print(f"  Name: {result.server_name}")
        print(f"  Status: {'enabled' if result.is_enabled else 'disabled'}")
        print(f"  Description: {result.description or 'N/A'}")
        print(f"  Rating: {result.num_stars} stars")
        print(f"  Tags: {', '.join(result.tags) if result.tags else 'None'}")
        print(f"  Transports: {', '.join(result.supported_transports)}")
        print(
            f"  Required Scopes: {', '.join(result.required_scopes) if result.required_scopes else 'None'}"
        )

        print(f"\n  Tool Mappings ({len(result.tool_mappings)}):")
        for mapping in result.tool_mappings:
            alias_info = f" -> {mapping.alias}" if mapping.alias else ""
            version_info = f" @{mapping.backend_version}" if mapping.backend_version else ""
            print(f"    - {mapping.tool_name}{alias_info}")
            print(f"      Backend: {mapping.backend_server_path}{version_info}")

        if result.tool_scope_overrides:
            print(f"\n  Tool Scope Overrides:")
            for override in result.tool_scope_overrides:
                print(f"    - {override.tool_alias}: {', '.join(override.required_scopes)}")

        print(f"\n  Created: {result.created_at or 'N/A'}")
        print(f"  Updated: {result.updated_at or 'N/A'}")
        print(f"  Created By: {result.created_by or 'N/A'}")

        return 0

    except Exception as e:
        logger.error(f"Get virtual server failed: {e}")
        return 1


def cmd_vs_update(args: argparse.Namespace) -> int:
    """
    Update a virtual MCP server from JSON config.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)

        # Load config from file
        with open(args.config, "r") as f:
            config_data = json.load(f)

        # Build tool mappings
        tool_mappings = []
        for mapping in config_data.get("tool_mappings", []):
            tool_mappings.append(
                ToolMapping(
                    tool_name=mapping["tool_name"],
                    alias=mapping.get("alias"),
                    backend_server_path=mapping["backend_server_path"],
                    backend_version=mapping.get("backend_version"),
                    description_override=mapping.get("description_override"),
                )
            )

        # Build tool scope overrides
        tool_scope_overrides = []
        for override in config_data.get("tool_scope_overrides", []):
            tool_scope_overrides.append(
                ToolScopeOverride(
                    tool_alias=override["tool_alias"],
                    required_scopes=override.get("required_scopes", []),
                )
            )

        request = VirtualServerCreateRequest(
            path=config_data["path"],
            server_name=config_data["server_name"],
            description=config_data.get("description"),
            tool_mappings=tool_mappings,
            required_scopes=config_data.get("required_scopes", []),
            tool_scope_overrides=tool_scope_overrides,
            tags=config_data.get("tags", []),
            supported_transports=config_data.get("supported_transports", ["streamable-http"]),
            is_enabled=config_data.get("is_enabled", True),
        )

        result = client.update_virtual_server(args.path, request)

        logger.info(f"Virtual server updated: {result.path}")
        print(
            json.dumps(
                {
                    "message": "Virtual server updated successfully",
                    "virtual_server": {
                        "path": result.path,
                        "server_name": result.server_name,
                        "is_enabled": result.is_enabled,
                    },
                },
                indent=2,
            )
        )
        return 0

    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        return 1
    except Exception as e:
        logger.error(f"Update virtual server failed: {e}")
        return 1


def cmd_vs_delete(args: argparse.Namespace) -> int:
    """
    Delete a virtual MCP server.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        if not args.force:
            confirm = input(f"Delete virtual server '{args.path}'? [y/N]: ")
            if confirm.lower() != "y":
                print("Cancelled")
                return 0

        client = _create_client(args)
        result = client.delete_virtual_server(args.path)

        logger.info(f"Virtual server deleted: {args.path}")
        print(
            json.dumps(
                {
                    "message": result.message,
                    "path": result.path,
                },
                indent=2,
            )
        )
        return 0

    except Exception as e:
        logger.error(f"Delete virtual server failed: {e}")
        return 1


def cmd_vs_toggle(args: argparse.Namespace) -> int:
    """
    Enable or disable a virtual MCP server.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        enable = args.enabled.lower() == "true"
        result = client.toggle_virtual_server(args.path, enable)

        action = "enabled" if result.is_enabled else "disabled"
        logger.info(f"Virtual server {action}: {args.path}")
        print(
            json.dumps(
                {
                    "message": result.message,
                    "path": result.path,
                    "is_enabled": result.is_enabled,
                },
                indent=2,
            )
        )
        return 0

    except Exception as e:
        logger.error(f"Toggle virtual server failed: {e}")
        return 1


def cmd_vs_rate(args: argparse.Namespace) -> int:
    """
    Rate a virtual MCP server.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        if not 1 <= args.rating <= 5:
            logger.error("Rating must be between 1 and 5")
            return 1

        client = _create_client(args)
        result = client.rate_virtual_server(args.path, args.rating)

        logger.info(f"Virtual server rated: {args.path}")
        print(json.dumps(result, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Rate virtual server failed: {e}")
        return 1


def cmd_vs_rating(args: argparse.Namespace) -> int:
    """
    Get rating information for a virtual MCP server.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        client = _create_client(args)
        result = client.get_virtual_server_rating(args.path)

        print(json.dumps(result, indent=2, default=str))
        return 0

    except Exception as e:
        logger.error(f"Get virtual server rating failed: {e}")
        return 1


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="MCP Gateway Registry Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (used if command-line options not provided):
  REGISTRY_URL        Registry base URL
  AWS_REGION          AWS region where Keycloak and SSM are deployed
  KEYCLOAK_URL        Keycloak base URL
  CLIENT_NAME         Keycloak client name (default: registry-admin-bot)
  GET_TOKEN_SCRIPT    Path to get-m2m-token.sh script

Examples:
  # Register a server (using environment variables)
  export REGISTRY_URL=https://registry.us-east-1.mycorp.click
  export AWS_REGION=us-east-1
  export KEYCLOAK_URL=https://kc.us-east-1.mycorp.click
  uv run python registry_management.py register --config server-config.json

  # Register a server (using command-line arguments)
  uv run python registry_management.py \\
    --registry-url https://registry.us-east-1.mycorp.click \\
    --aws-region us-east-1 \\
    --keycloak-url https://kc.us-east-1.mycorp.click \\
    register --config server-config.json

  # Register a server (using token file)
  uv run python registry_management.py \\
    --registry-url https://registry.us-east-1.mycorp.click \\
    --token-file /path/to/token.txt \\
    register --config server-config.json

  # List all servers
  uv run python registry_management.py list

  # Toggle server status
  uv run python registry_management.py toggle --path /cloudflare-docs

  # Add server to groups
  uv run python registry_management.py add-to-groups --server my-server --groups finance,analytics
        """,
    )

    parser.add_argument("--registry-url", help="Registry base URL (overrides REGISTRY_URL env var)")

    parser.add_argument("--aws-region", help="AWS region (overrides AWS_REGION env var)")

    parser.add_argument("--keycloak-url", help="Keycloak base URL (overrides KEYCLOAK_URL env var)")

    parser.add_argument(
        "--token-file", help="Path to file containing JWT token (bypasses token script)"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new server")
    register_parser.add_argument(
        "--config", required=True, help="Path to server configuration JSON file"
    )
    register_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite if server already exists"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List all servers")
    list_parser.add_argument("--json", action="store_true", help="Print raw JSON response")

    # Toggle command
    toggle_parser = subparsers.add_parser("toggle", help="Toggle server status")
    toggle_parser.add_argument("--path", required=True, help="Server path to toggle")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a server")
    remove_parser.add_argument("--path", required=True, help="Server path to remove")
    remove_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    # Healthcheck command
    healthcheck_parser = subparsers.add_parser("healthcheck", help="Health check all servers")

    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Get registry configuration (deployment mode, features)"
    )
    config_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # Add to groups command
    add_groups_parser = subparsers.add_parser("add-to-groups", help="Add server to groups")
    add_groups_parser.add_argument("--server", required=True, help="Server name")
    add_groups_parser.add_argument("--groups", required=True, help="Comma-separated group names")

    # Remove from groups command
    remove_groups_parser = subparsers.add_parser(
        "remove-from-groups", help="Remove server from groups"
    )
    remove_groups_parser.add_argument("--server", required=True, help="Server name")
    remove_groups_parser.add_argument("--groups", required=True, help="Comma-separated group names")

    # Create group command
    create_group_parser = subparsers.add_parser("create-group", help="Create a new group")
    create_group_parser.add_argument("--name", required=True, help="Group name")
    create_group_parser.add_argument("--description", help="Group description")
    create_group_parser.add_argument(
        "--idp", action="store_true", help="Also create in IdP (Keycloak/Entra)"
    )

    # Delete group command
    delete_group_parser = subparsers.add_parser("delete-group", help="Delete a group")
    delete_group_parser.add_argument("--name", required=True, help="Group name")
    delete_group_parser.add_argument(
        "--idp", action="store_true", help="Also delete from IdP (Keycloak/Entra)"
    )
    delete_group_parser.add_argument(
        "--force", action="store_true", help="Force deletion of system groups and skip confirmation"
    )

    # Import group command
    import_group_parser = subparsers.add_parser(
        "import-group", help="Import a complete group definition from JSON file"
    )
    import_group_parser.add_argument(
        "--file", required=True, help="Path to JSON file containing group definition"
    )

    # List groups command
    list_groups_parser = subparsers.add_parser("list-groups", help="List all groups")
    list_groups_parser.add_argument(
        "--no-keycloak", action="store_true", help="Exclude Keycloak information"
    )
    list_groups_parser.add_argument(
        "--no-scopes", action="store_true", help="Exclude scope information"
    )
    list_groups_parser.add_argument("--json", action="store_true", help="Output raw JSON response")

    # Describe group command
    describe_group_parser = subparsers.add_parser(
        "describe-group", help="Show detailed information about a specific group"
    )
    describe_group_parser.add_argument("--name", required=True, help="Group name to describe")
    describe_group_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON response"
    )

    # Server rate command
    server_rate_parser = subparsers.add_parser("server-rate", help="Rate a server (1-5 stars)")
    server_rate_parser.add_argument(
        "--path", required=True, help="Server path (e.g., /cloudflare-docs)"
    )
    server_rate_parser.add_argument(
        "--rating",
        required=True,
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Rating value (1-5 stars)",
    )

    # Server rating command
    server_rating_parser = subparsers.add_parser(
        "server-rating", help="Get rating information for a server"
    )
    server_rating_parser.add_argument(
        "--path", required=True, help="Server path (e.g., /cloudflare-docs)"
    )

    # Server security scan command
    security_scan_parser = subparsers.add_parser(
        "security-scan", help="Get security scan results for a server"
    )
    security_scan_parser.add_argument(
        "--path", required=True, help="Server path (e.g., /cloudflare-docs)"
    )
    security_scan_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # Server rescan command
    rescan_parser = subparsers.add_parser(
        "rescan", help="Trigger manual security scan for a server (admin only)"
    )
    rescan_parser.add_argument("--path", required=True, help="Server path (e.g., /cloudflare-docs)")
    rescan_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # Server credential update command
    server_update_cred_parser = subparsers.add_parser(
        "server-update-credential", help="Update authentication credentials for a server"
    )
    server_update_cred_parser.add_argument(
        "--path", required=True, help="Server path (e.g., /cloudflare-api)"
    )
    server_update_cred_parser.add_argument(
        "--auth-scheme",
        required=True,
        choices=["none", "bearer", "api_key"],
        help="Authentication scheme",
    )
    server_update_cred_parser.add_argument(
        "--credential", help="New credential value (required if auth-scheme is not 'none')"
    )
    server_update_cred_parser.add_argument(
        "--auth-header-name", help="Custom header name (optional, for api_key scheme)"
    )
    server_update_cred_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # Server search command
    server_search_parser = subparsers.add_parser(
        "server-search",
        help="Semantic search across all entity types (servers, tools, agents, skills, virtual servers)",
    )
    server_search_parser.add_argument(
        "--query", required=True, help="Natural language search query (e.g., 'coding assistants')"
    )
    server_search_parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results per entity type (default: 10)",
    )
    server_search_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON with all entity types"
    )

    # Server Version Management Commands

    # List versions command
    list_versions_parser = subparsers.add_parser(
        "list-versions", help="List all versions for a server"
    )
    list_versions_parser.add_argument("--path", required=True, help="Server path (e.g., /context7)")
    list_versions_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # Remove version command
    remove_version_parser = subparsers.add_parser(
        "remove-version", help="Remove a version from a server"
    )
    remove_version_parser.add_argument(
        "--path", required=True, help="Server path (e.g., /context7)"
    )
    remove_version_parser.add_argument("--version", required=True, help="Version to remove")
    remove_version_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # Set default version command
    set_default_version_parser = subparsers.add_parser(
        "set-default-version", help="Set the default version for a server"
    )
    set_default_version_parser.add_argument(
        "--path", required=True, help="Server path (e.g., /context7)"
    )
    set_default_version_parser.add_argument(
        "--version", required=True, help="Version to set as default"
    )
    set_default_version_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # Agent Management Commands

    # Agent register command
    agent_register_parser = subparsers.add_parser("agent-register", help="Register a new A2A agent")
    agent_register_parser.add_argument(
        "--config", required=True, help="Path to agent configuration JSON file"
    )

    # Agent list command
    agent_list_parser = subparsers.add_parser("agent-list", help="List all A2A agents")
    agent_list_parser.add_argument("--query", help="Search query string")
    agent_list_parser.add_argument(
        "--enabled-only", action="store_true", help="Show only enabled agents"
    )
    agent_list_parser.add_argument(
        "--visibility", choices=["public", "private", "internal"], help="Filter by visibility level"
    )

    # Agent get command
    agent_get_parser = subparsers.add_parser("agent-get", help="Get agent details")
    agent_get_parser.add_argument("--path", required=True, help="Agent path (e.g., /code-reviewer)")

    # Agent update command
    agent_update_parser = subparsers.add_parser("agent-update", help="Update an existing agent")
    agent_update_parser.add_argument("--path", required=True, help="Agent path")
    agent_update_parser.add_argument(
        "--config", required=True, help="Path to updated agent configuration JSON file"
    )

    # Agent delete command
    agent_delete_parser = subparsers.add_parser("agent-delete", help="Delete an agent")
    agent_delete_parser.add_argument("--path", required=True, help="Agent path")
    agent_delete_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # Agent toggle command
    agent_toggle_parser = subparsers.add_parser(
        "agent-toggle", help="Toggle agent enabled/disabled status"
    )
    agent_toggle_parser.add_argument("--path", required=True, help="Agent path")
    agent_toggle_parser.add_argument(
        "--enabled",
        required=True,
        type=lambda x: x.lower() == "true",
        help="True to enable, false to disable",
    )

    # Agent discover command
    agent_discover_parser = subparsers.add_parser(
        "agent-discover", help="Discover agents by skills"
    )
    agent_discover_parser.add_argument(
        "--skills", required=True, help="Comma-separated list of required skills"
    )
    agent_discover_parser.add_argument("--tags", help="Comma-separated list of tag filters")
    agent_discover_parser.add_argument(
        "--max-results", type=int, default=10, help="Maximum number of results (default: 10)"
    )

    # Agent search command
    agent_search_parser = subparsers.add_parser("agent-search", help="Semantic search for agents")
    agent_search_parser.add_argument("--query", required=True, help="Natural language search query")
    agent_search_parser.add_argument(
        "--max-results", type=int, default=10, help="Maximum number of results (default: 10)"
    )
    agent_search_parser.add_argument("--json", action="store_true", help="Output results as JSON")

    # Agent rate command
    agent_rate_parser = subparsers.add_parser("agent-rate", help="Rate an agent (1-5 stars)")
    agent_rate_parser.add_argument(
        "--path", required=True, help="Agent path (e.g., /code-reviewer)"
    )
    agent_rate_parser.add_argument(
        "--rating",
        required=True,
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Rating value (1-5 stars)",
    )

    # Agent rating command
    agent_rating_parser = subparsers.add_parser(
        "agent-rating", help="Get rating information for an agent"
    )
    agent_rating_parser.add_argument(
        "--path", required=True, help="Agent path (e.g., /code-reviewer)"
    )

    # Agent security scan command
    agent_security_scan_parser = subparsers.add_parser(
        "agent-security-scan", help="Get security scan results for an agent"
    )
    agent_security_scan_parser.add_argument(
        "--path", required=True, help="Agent path (e.g., /code-reviewer)"
    )

    # Agent rescan command
    agent_rescan_parser = subparsers.add_parser(
        "agent-rescan", help="Trigger manual security scan for an agent (admin only)"
    )
    agent_rescan_parser.add_argument(
        "--path", required=True, help="Agent path (e.g., /code-reviewer)"
    )
    agent_rescan_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # ==========================================
    # Agent Skills Commands
    # ==========================================

    # Skill register command
    skill_register_parser = subparsers.add_parser(
        "skill-register", help="Register a new Agent Skill"
    )
    skill_register_parser.add_argument(
        "--name", required=True, help="Skill name (lowercase alphanumeric with hyphens)"
    )
    skill_register_parser.add_argument("--url", required=True, help="URL to SKILL.md file")
    skill_register_parser.add_argument("--description", help="Skill description")
    skill_register_parser.add_argument("--tags", help="Comma-separated tags")
    skill_register_parser.add_argument(
        "--visibility",
        choices=["public", "private", "group"],
        default="public",
        help="Visibility level (default: public)",
    )

    # Skill list command
    skill_list_parser = subparsers.add_parser("skill-list", help="List all Agent Skills")
    skill_list_parser.add_argument(
        "--include-disabled", action="store_true", help="Include disabled skills"
    )
    skill_list_parser.add_argument("--tag", help="Filter by tag")

    # Skill get command
    skill_get_parser = subparsers.add_parser("skill-get", help="Get skill details")
    skill_get_parser.add_argument(
        "--path", required=True, help="Skill path or name (e.g., pdf-processing)"
    )

    # Skill delete command
    skill_delete_parser = subparsers.add_parser("skill-delete", help="Delete a skill")
    skill_delete_parser.add_argument("--path", required=True, help="Skill path or name")

    # Skill toggle command
    skill_toggle_parser = subparsers.add_parser(
        "skill-toggle", help="Toggle skill enabled/disabled state"
    )
    skill_toggle_parser.add_argument("--path", required=True, help="Skill path or name")
    skill_toggle_parser.add_argument(
        "--enable",
        type=lambda x: x.lower() == "true",
        required=True,
        help="Enable (true) or disable (false)",
    )

    # Skill health command
    skill_health_parser = subparsers.add_parser(
        "skill-health", help="Check skill health (SKILL.md accessibility)"
    )
    skill_health_parser.add_argument("--path", required=True, help="Skill path or name")

    # Skill content command
    skill_content_parser = subparsers.add_parser(
        "skill-content", help="Get SKILL.md content for a skill"
    )
    skill_content_parser.add_argument("--path", required=True, help="Skill path or name")
    skill_content_parser.add_argument("--raw", action="store_true", help="Output raw content only")

    # Skill search command
    skill_search_parser = subparsers.add_parser("skill-search", help="Search for skills")
    skill_search_parser.add_argument("--query", required=True, help="Search query")
    skill_search_parser.add_argument("--tags", help="Comma-separated tags filter")

    # Skill rate command
    skill_rate_parser = subparsers.add_parser("skill-rate", help="Rate a skill (1-5 stars)")
    skill_rate_parser.add_argument("--path", required=True, help="Skill path or name")
    skill_rate_parser.add_argument(
        "--rating", type=int, required=True, choices=[1, 2, 3, 4, 5], help="Rating (1-5 stars)"
    )

    # Skill rating command
    skill_rating_parser = subparsers.add_parser(
        "skill-rating", help="Get rating information for a skill"
    )
    skill_rating_parser.add_argument("--path", required=True, help="Skill path or name")

    # Skill security scan command
    skill_security_scan_parser = subparsers.add_parser(
        "skill-security-scan", help="Get security scan results for a skill"
    )
    skill_security_scan_parser.add_argument("--path", required=True, help="Skill path or name")

    # Skill rescan command
    skill_rescan_parser = subparsers.add_parser(
        "skill-rescan", help="Trigger manual security scan for a skill (admin only)"
    )
    skill_rescan_parser.add_argument("--path", required=True, help="Skill path or name")
    skill_rescan_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Output raw JSON only"
    )

    # Anthropic Registry API Commands

    # Anthropic list servers command
    anthropic_list_parser = subparsers.add_parser(
        "anthropic-list", help="List all servers (Anthropic Registry API v0.1)"
    )
    anthropic_list_parser.add_argument("--limit", type=int, help="Maximum results per page")
    anthropic_list_parser.add_argument(
        "--raw", action="store_true", help="Output raw JSON response"
    )

    # Anthropic list versions command
    anthropic_versions_parser = subparsers.add_parser(
        "anthropic-versions", help="List versions for a server (Anthropic Registry API v0.1)"
    )
    anthropic_versions_parser.add_argument(
        "--server-name",
        required=True,
        help="Server name in reverse-DNS format (e.g., 'io.mcpgateway/example-server')",
    )
    anthropic_versions_parser.add_argument(
        "--raw", action="store_true", help="Output raw JSON response"
    )

    # Anthropic get server command
    anthropic_get_parser = subparsers.add_parser(
        "anthropic-get", help="Get server details (Anthropic Registry API v0.1)"
    )
    anthropic_get_parser.add_argument(
        "--server-name", required=True, help="Server name in reverse-DNS format"
    )
    anthropic_get_parser.add_argument(
        "--version", default="latest", help="Server version (default: latest)"
    )
    anthropic_get_parser.add_argument("--raw", action="store_true", help="Output raw JSON response")

    # User Management Commands (Management API)

    # List users command
    user_list_parser = subparsers.add_parser("user-list", help="List Keycloak users")
    user_list_parser.add_argument("--search", help="Search string to filter users")
    user_list_parser.add_argument(
        "--limit", type=int, default=500, help="Maximum number of results (default: 500)"
    )

    # Create M2M account command
    user_m2m_parser = subparsers.add_parser("user-create-m2m", help="Create M2M service account")
    user_m2m_parser.add_argument("--name", required=True, help="Service account name/client ID")
    user_m2m_parser.add_argument(
        "--groups", required=True, help="Comma-separated list of group names"
    )
    user_m2m_parser.add_argument("--description", help="Account description")

    # Create human user command
    user_human_parser = subparsers.add_parser("user-create-human", help="Create human user account")
    user_human_parser.add_argument("--username", required=True, help="Username")
    user_human_parser.add_argument("--email", required=True, help="Email address")
    user_human_parser.add_argument("--first-name", required=True, help="First name")
    user_human_parser.add_argument("--last-name", required=True, help="Last name")
    user_human_parser.add_argument(
        "--groups", required=True, help="Comma-separated list of group names"
    )
    user_human_parser.add_argument("--password", help="Initial password (optional)")

    # Delete user command
    user_delete_parser = subparsers.add_parser("user-delete", help="Delete a user")
    user_delete_parser.add_argument("--username", required=True, help="Username to delete")
    user_delete_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    # Create IAM group command
    group_create_parser = subparsers.add_parser("group-create", help="Create a new IAM group")
    group_create_parser.add_argument("--name", required=True, help="Group name")
    group_create_parser.add_argument("--description", help="Group description")

    # Delete IAM group command
    group_delete_parser = subparsers.add_parser("group-delete", help="Delete an IAM group")
    group_delete_parser.add_argument("--name", required=True, help="Group name to delete")
    group_delete_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # List IAM groups command
    group_list_parser = subparsers.add_parser("group-list", help="List IAM groups")

    # Federation Management Commands

    # Get federation config command
    federation_get_parser = subparsers.add_parser(
        "federation-get", help="Get federation configuration"
    )
    federation_get_parser.add_argument(
        "--config-id", default="default", help="Configuration ID (default: default)"
    )
    federation_get_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # Save federation config command
    federation_save_parser = subparsers.add_parser(
        "federation-save", help="Save federation configuration from JSON file"
    )
    federation_save_parser.add_argument(
        "--config", required=True, help="Path to federation config JSON file"
    )
    federation_save_parser.add_argument(
        "--config-id", default="default", help="Configuration ID (default: default)"
    )

    # Delete federation config command
    federation_delete_parser = subparsers.add_parser(
        "federation-delete", help="Delete federation configuration"
    )
    federation_delete_parser.add_argument(
        "--config-id", default="default", help="Configuration ID to delete (default: default)"
    )
    federation_delete_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # List federation configs command
    federation_list_parser = subparsers.add_parser(
        "federation-list", help="List all federation configurations"
    )
    federation_list_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # Add Anthropic server command
    federation_add_anthropic_parser = subparsers.add_parser(
        "federation-add-anthropic-server", help="Add Anthropic server to federation config"
    )
    federation_add_anthropic_parser.add_argument(
        "--server-name",
        required=True,
        help="Anthropic server name (e.g., io.github.jgador/websharp)",
    )
    federation_add_anthropic_parser.add_argument(
        "--config-id", default="default", help="Configuration ID (default: default)"
    )

    # Remove Anthropic server command
    federation_remove_anthropic_parser = subparsers.add_parser(
        "federation-remove-anthropic-server", help="Remove Anthropic server from federation config"
    )
    federation_remove_anthropic_parser.add_argument(
        "--server-name", required=True, help="Anthropic server name to remove"
    )
    federation_remove_anthropic_parser.add_argument(
        "--config-id", default="default", help="Configuration ID (default: default)"
    )

    # Add ASOR agent command
    federation_add_asor_parser = subparsers.add_parser(
        "federation-add-asor-agent", help="Add ASOR agent to federation config"
    )
    federation_add_asor_parser.add_argument(
        "--agent-id", required=True, help="ASOR agent ID (e.g., aws_assistant)"
    )
    federation_add_asor_parser.add_argument(
        "--config-id", default="default", help="Configuration ID (default: default)"
    )

    # Remove ASOR agent command
    federation_remove_asor_parser = subparsers.add_parser(
        "federation-remove-asor-agent", help="Remove ASOR agent from federation config"
    )
    federation_remove_asor_parser.add_argument(
        "--agent-id", required=True, help="ASOR agent ID to remove"
    )
    federation_remove_asor_parser.add_argument(
        "--config-id", default="default", help="Configuration ID (default: default)"
    )

    # Federation sync command
    federation_sync_parser = subparsers.add_parser(
        "federation-sync", help="Trigger manual federation sync to import servers/agents"
    )
    federation_sync_parser.add_argument(
        "--config-id", default="default", help="Configuration ID (default: default)"
    )
    federation_sync_parser.add_argument(
        "--source",
        choices=["anthropic", "asor"],
        help="Optional source filter (anthropic or asor). Syncs all enabled sources if not specified.",
    )
    federation_sync_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # ==========================================
    # Peer Registry Management Commands
    # ==========================================

    # List peers command
    peer_list_parser = subparsers.add_parser(
        "peer-list", help="List all configured peer registries"
    )
    peer_list_parser.add_argument(
        "--enabled-only", action="store_true", help="Show only enabled peers"
    )
    peer_list_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # Add peer command
    peer_add_parser = subparsers.add_parser(
        "peer-add", help="Add a new peer registry from JSON config"
    )
    peer_add_parser.add_argument(
        "--config", required=True, help="Path to peer configuration JSON file"
    )
    peer_add_parser.add_argument(
        "--federation-token",
        required=False,
        help="Federation static token from the remote peer registry. "
        "Overrides federation_token in the JSON config file if both are provided.",
    )

    # Get peer command
    peer_get_parser = subparsers.add_parser(
        "peer-get", help="Get details of a specific peer registry"
    )
    peer_get_parser.add_argument("--peer-id", required=True, help="Peer registry identifier")
    peer_get_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # Update peer command
    peer_update_parser = subparsers.add_parser(
        "peer-update", help="Update an existing peer registry"
    )
    peer_update_parser.add_argument("--peer-id", required=True, help="Peer registry identifier")
    peer_update_parser.add_argument(
        "--config", required=True, help="Path to updated peer configuration JSON file"
    )
    peer_update_parser.add_argument(
        "--federation-token",
        required=False,
        help="Federation static token from the remote peer registry. "
        "Overrides federation_token in the JSON config file if both are provided.",
    )

    # Update peer token command
    peer_update_token_parser = subparsers.add_parser(
        "peer-update-token", help="Update only the federation token for a peer registry"
    )
    peer_update_token_parser.add_argument(
        "--peer-id", required=True, help="Peer registry identifier"
    )
    peer_update_token_parser.add_argument(
        "--federation-token",
        required=True,
        help="New federation static token from the remote peer registry. "
        "Use this to recover from token loss (issue #561) or rotate tokens.",
    )

    # Remove peer command
    peer_remove_parser = subparsers.add_parser("peer-remove", help="Remove a peer registry")
    peer_remove_parser.add_argument("--peer-id", required=True, help="Peer registry identifier")
    peer_remove_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    # Sync from specific peer command
    peer_sync_parser = subparsers.add_parser(
        "peer-sync", help="Trigger sync from a specific peer registry"
    )
    peer_sync_parser.add_argument(
        "--peer-id", required=True, help="Peer registry identifier to sync from"
    )
    peer_sync_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # Sync from all peers command
    peer_sync_all_parser = subparsers.add_parser(
        "peer-sync-all", help="Trigger sync from all enabled peer registries"
    )
    peer_sync_all_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # Get peer sync status command
    peer_status_parser = subparsers.add_parser(
        "peer-status", help="Get sync status for a specific peer registry"
    )
    peer_status_parser.add_argument("--peer-id", required=True, help="Peer registry identifier")
    peer_status_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # Enable peer command
    peer_enable_parser = subparsers.add_parser("peer-enable", help="Enable a peer registry")
    peer_enable_parser.add_argument("--peer-id", required=True, help="Peer registry identifier")

    # Disable peer command
    peer_disable_parser = subparsers.add_parser("peer-disable", help="Disable a peer registry")
    peer_disable_parser.add_argument("--peer-id", required=True, help="Peer registry identifier")

    # Get peer connections command
    peer_connections_parser = subparsers.add_parser(
        "peer-connections", help="Get all federation connections across all peers"
    )
    peer_connections_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # Get shared resources command
    peer_shared_resources_parser = subparsers.add_parser(
        "peer-shared-resources", help="Get resource sharing summary across all peers"
    )
    peer_shared_resources_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # ==========================================
    # Virtual MCP Server Commands
    # ==========================================

    # Create virtual server command
    vs_create_parser = subparsers.add_parser(
        "vs-create", help="Create a virtual MCP server from JSON config"
    )
    vs_create_parser.add_argument(
        "--config", required=True, help="Path to virtual server configuration JSON file"
    )

    # List virtual servers command
    vs_list_parser = subparsers.add_parser("vs-list", help="List all virtual MCP servers")
    vs_list_parser.add_argument(
        "--enabled-only", action="store_true", help="Show only enabled virtual servers"
    )
    vs_list_parser.add_argument("--tag", help="Filter by tag")
    vs_list_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # Get virtual server command
    vs_get_parser = subparsers.add_parser("vs-get", help="Get virtual MCP server details")
    vs_get_parser.add_argument(
        "--path", required=True, help="Virtual server path (e.g., /virtual/dev-tools)"
    )
    vs_get_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    # Update virtual server command
    vs_update_parser = subparsers.add_parser(
        "vs-update", help="Update a virtual MCP server from JSON config"
    )
    vs_update_parser.add_argument("--path", required=True, help="Virtual server path to update")
    vs_update_parser.add_argument(
        "--config", required=True, help="Path to updated configuration JSON file"
    )

    # Delete virtual server command
    vs_delete_parser = subparsers.add_parser("vs-delete", help="Delete a virtual MCP server")
    vs_delete_parser.add_argument("--path", required=True, help="Virtual server path to delete")
    vs_delete_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    # Toggle virtual server command
    vs_toggle_parser = subparsers.add_parser(
        "vs-toggle", help="Enable or disable a virtual MCP server"
    )
    vs_toggle_parser.add_argument("--path", required=True, help="Virtual server path")
    vs_toggle_parser.add_argument(
        "--enabled",
        required=True,
        choices=["true", "false"],
        help="Enable (true) or disable (false)",
    )

    # Rate virtual server command
    vs_rate_parser = subparsers.add_parser("vs-rate", help="Rate a virtual MCP server (1-5 stars)")
    vs_rate_parser.add_argument("--path", required=True, help="Virtual server path")
    vs_rate_parser.add_argument(
        "--rating", required=True, type=int, choices=[1, 2, 3, 4, 5], help="Rating (1-5 stars)"
    )

    # Get virtual server rating command
    vs_rating_parser = subparsers.add_parser(
        "vs-rating", help="Get rating information for a virtual MCP server"
    )
    vs_rating_parser.add_argument("--path", required=True, help="Virtual server path")

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Dispatch to command handler
    if not args.command:
        parser.print_help()
        return 1

    command_handlers = {
        "register": cmd_register,
        "list": cmd_list,
        "toggle": cmd_toggle,
        "remove": cmd_remove,
        "healthcheck": cmd_healthcheck,
        "config": cmd_config,
        "add-to-groups": cmd_add_to_groups,
        "remove-from-groups": cmd_remove_from_groups,
        "create-group": cmd_create_group,
        "delete-group": cmd_delete_group,
        "import-group": cmd_import_group,
        "list-groups": cmd_list_groups,
        "describe-group": cmd_describe_group,
        "server-rate": cmd_server_rate,
        "server-rating": cmd_server_rating,
        "security-scan": cmd_security_scan,
        "rescan": cmd_rescan,
        "server-update-credential": cmd_server_update_credential,
        "server-search": cmd_server_search,
        "list-versions": cmd_list_versions,
        "remove-version": cmd_remove_version,
        "set-default-version": cmd_set_default_version,
        "agent-register": cmd_agent_register,
        "agent-list": cmd_agent_list,
        "agent-get": cmd_agent_get,
        "agent-update": cmd_agent_update,
        "agent-delete": cmd_agent_delete,
        "agent-toggle": cmd_agent_toggle,
        "agent-discover": cmd_agent_discover,
        "agent-search": cmd_agent_search,
        "agent-rate": cmd_agent_rate,
        "agent-rating": cmd_agent_rating,
        "agent-security-scan": cmd_agent_security_scan,
        "agent-rescan": cmd_agent_rescan,
        # Skill commands
        "skill-register": cmd_skill_register,
        "skill-list": cmd_skill_list,
        "skill-get": cmd_skill_get,
        "skill-delete": cmd_skill_delete,
        "skill-toggle": cmd_skill_toggle,
        "skill-health": cmd_skill_health,
        "skill-content": cmd_skill_content,
        "skill-search": cmd_skill_search,
        "skill-rate": cmd_skill_rate,
        "skill-rating": cmd_skill_rating,
        "skill-security-scan": cmd_skill_security_scan,
        "skill-rescan": cmd_skill_rescan,
        "anthropic-list": cmd_anthropic_list_servers,
        "anthropic-versions": cmd_anthropic_list_versions,
        "anthropic-get": cmd_anthropic_get_server,
        "user-list": cmd_user_list,
        "user-create-m2m": cmd_user_create_m2m,
        "user-create-human": cmd_user_create_human,
        "user-delete": cmd_user_delete,
        "group-create": cmd_group_create,
        "group-delete": cmd_group_delete,
        "group-list": cmd_group_list,
        "federation-get": cmd_federation_get,
        "federation-save": cmd_federation_save,
        "federation-delete": cmd_federation_delete,
        "federation-list": cmd_federation_list,
        "federation-add-anthropic-server": cmd_federation_add_anthropic_server,
        "federation-remove-anthropic-server": cmd_federation_remove_anthropic_server,
        "federation-add-asor-agent": cmd_federation_add_asor_agent,
        "federation-remove-asor-agent": cmd_federation_remove_asor_agent,
        "federation-sync": cmd_federation_sync,
        "peer-list": cmd_peer_list,
        "peer-add": cmd_peer_add,
        "peer-get": cmd_peer_get,
        "peer-update": cmd_peer_update,
        "peer-update-token": cmd_peer_update_token,
        "peer-remove": cmd_peer_remove,
        "peer-sync": cmd_peer_sync,
        "peer-sync-all": cmd_peer_sync_all,
        "peer-status": cmd_peer_status,
        "peer-enable": cmd_peer_enable,
        "peer-disable": cmd_peer_disable,
        "peer-connections": cmd_peer_connections,
        "peer-shared-resources": cmd_peer_shared_resources,
        # Virtual server commands
        "vs-create": cmd_vs_create,
        "vs-list": cmd_vs_list,
        "vs-get": cmd_vs_get,
        "vs-update": cmd_vs_update,
        "vs-delete": cmd_vs_delete,
        "vs-toggle": cmd_vs_toggle,
        "vs-rate": cmd_vs_rate,
        "vs-rating": cmd_vs_rating,
    }

    handler = command_handlers.get(args.command)
    if not handler:
        logger.error(f"Unknown command: {args.command}")
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
