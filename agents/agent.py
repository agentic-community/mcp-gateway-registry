#!/usr/bin/env python3
"""
Interactive LangGraph Agent with Registry Tool Discovery

This agent discovers and invokes MCP tools using semantic search on the registry.
It supports multi-turn conversation and maintains conversation history.

Authentication:
    The agent requires a JWT token for authenticating with the MCP Registry.
    The token can be obtained from different sources depending on your setup.

Usage Examples:
    # Using token from api/.token file (simple cat):
    python agents/agent.py \
        --mcp-registry-url https://mcpgateway.ddns.net/mcpgw/mcp \
        --jwt-token "$(cat api/.token)" \
        --provider bedrock \
        --prompt "What time is it in New York?"

    # Using token from .oauth-tokens/ingress.json (requires jq):
    python agents/agent.py \
        --mcp-registry-url https://mcpgateway.ddns.net/mcpgw/mcp \
        --jwt-token "$(jq -r '.access_token' .oauth-tokens/ingress.json)" \
        --provider bedrock \
        --prompt "What time is it in New York?"

    # Interactive mode for multi-turn conversations:
    python agents/agent.py \
        --mcp-registry-url https://mcpgateway.ddns.net/mcpgw/mcp \
        --jwt-token "$(cat api/.token)" \
        --provider bedrock \
        --interactive

    # With verbose logging for debugging:
    python agents/agent.py \
        --mcp-registry-url https://mcpgateway.ddns.net/mcpgw/mcp \
        --jwt-token "$(cat api/.token)" \
        --provider bedrock \
        --prompt "What time is it in New York?" \
        --verbose

Available Tools:
    - calculator: For mathematical calculations
    - search_registry_tools: Discover MCP tools via semantic search
    - invoke_mcp_tool: Invoke discovered tools on MCP servers

Environment Variables:
    - ANTHROPIC_API_KEY: Required when using --provider anthropic
"""

import asyncio
import argparse
import json
import logging
import ast
import operator as _operator
import os
import re
import sys
import threading
import time
import yaml
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)
from urllib.parse import (
    urlparse,
    urljoin,
)

import httpx
import mcp
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

from registry_client import (
    RegistryClient,
    _format_tool_result,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Get logger
logger = logging.getLogger(__name__)

# Global registry client instance (initialized in main)
registry_client: Optional[RegistryClient] = None


class ProgressSpinner:
    """Simple progress spinner for showing activity during operations."""

    SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _spin(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            char = self.SPINNER_CHARS[idx % len(self.SPINNER_CHARS)]
            sys.stdout.write(f"\r{char}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)

    def start(self) -> "ProgressSpinner":
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def stop(
        self,
        final_message: str = None,
    ) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        # Clear the spinner character
        sys.stdout.write("\r  \r")
        sys.stdout.flush()
        if final_message:
            print(f"  {final_message}")


def print_step(
    step: str,
    icon: str = "->",
) -> None:
    """Print a step indicator."""
    print(f"  {icon} {step}")


def load_server_config(config_file: str = "server_config.yml") -> Dict[str, Any]:
    """
    Load server configuration from YAML file.

    Args:
        config_file: Path to the configuration file

    Returns:
        Dict containing server configurations
    """
    try:
        # Try to find config file in the same directory as this script
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        if not os.path.exists(config_path):
            # Try current working directory
            config_path = config_file
            if not os.path.exists(config_path):
                logger.warning(
                    f"Server config file not found: {config_file}. Using default configuration."
                )
                return {"servers": {}}

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded server config from: {config_path}")
            return config or {"servers": {}}
    except Exception as e:
        logger.warning(f"Failed to load server config: {e}. Using default configuration.")
        return {"servers": {}}


def resolve_env_vars(value: str, server_name: str = None) -> str:
    """
    Resolve environment variable references in a string.
    Supports ${VAR_NAME} syntax.

    Args:
        value: String that may contain environment variable references
        server_name: Name of the server (for error context)

    Returns:
        String with environment variables resolved

    Raises:
        ValueError: If a required environment variable is not found
    """
    import re

    missing_vars = []

    def replace_env_var(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            missing_vars.append(var_name)
            return match.group(0)  # Return original if not found
        return env_value

    # Find all ${VAR_NAME} patterns and replace them
    pattern = r"\$\{([^}]+)\}"
    resolved_value = re.sub(pattern, replace_env_var, value)

    # If any environment variables were missing, raise an error
    if missing_vars:
        server_context = f" for server '{server_name}'" if server_name else ""
        missing_list = "', '".join(missing_vars)
        raise ValueError(
            f"Missing required environment variable(s): '{missing_list}'{server_context}. "
            f"Please set these environment variables and try again."
        )

    return resolved_value


def get_server_headers(server_name: str, config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get server-specific headers from configuration with environment variable resolution.

    Args:
        server_name: Name of the server (e.g., 'sre-gateway', 'atlassian')
        config: Loaded server configuration

    Returns:
        Dictionary of headers for the server

    Raises:
        ValueError: If required environment variables for the server are missing
    """
    servers = config.get("servers", {})
    server_config = servers.get(server_name, {})
    raw_headers = server_config.get("headers", {})

    if not raw_headers:
        logger.debug(f"No custom headers configured for server '{server_name}'")
        return {}

    # Resolve environment variables in header values
    resolved_headers = {}
    try:
        for header_name, header_value in raw_headers.items():
            resolved_value = resolve_env_vars(header_value, server_name)
            if resolved_value != header_value:
                logger.debug(f"Resolved header {header_name} for server {server_name}")
            resolved_headers[header_name] = resolved_value

        logger.info(f"Applied {len(resolved_headers)} custom headers for server '{server_name}'")
        return resolved_headers

    except ValueError as e:
        # Re-raise with additional context about which server failed
        logger.error(f"Failed to configure headers for server '{server_name}': {e}")
        raise


def enable_verbose_logging():
    """Enable verbose debug logging for HTTP libraries and main logger."""
    # Set main logger to DEBUG level
    logger.setLevel(logging.DEBUG)

    # Enable debug logging for httpx to see request/response details
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.DEBUG)
    httpx_logger.propagate = True

    # Enable debug logging for httpcore (underlying HTTP library)
    httpcore_logger = logging.getLogger("httpcore")
    httpcore_logger.setLevel(logging.DEBUG)
    httpcore_logger.propagate = True

    # Enable debug logging for mcp client libraries
    mcp_logger = logging.getLogger("mcp")
    mcp_logger.setLevel(logging.DEBUG)
    mcp_logger.propagate = True

    logger.info("Verbose logging enabled for httpx, httpcore, mcp libraries, and main logger")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the Interactive LangGraph Agent.

    Returns:
        argparse.Namespace: The parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Interactive LangGraph Agent with Registry Tool Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using token from api/.token file:
    python agents/agent.py --jwt-token "$(cat api/.token)" --prompt "What time is it?"

    # Using token from .oauth-tokens/ingress.json:
    python agents/agent.py --jwt-token "$(jq -r '.access_token' .oauth-tokens/ingress.json)" --prompt "What time is it?"

    # Interactive mode:
    python agents/agent.py --jwt-token "$(cat api/.token)" --interactive
""",
    )

    # Server connection arguments
    parser.add_argument(
        "--mcp-registry-url",
        type=str,
        default="https://mcpgateway.ddns.net/mcpgw/mcp",
        help="URL of the MCP Registry (default: https://mcpgateway.ddns.net/mcpgw/mcp)",
    )

    # Authentication - JWT token required
    parser.add_argument(
        "--jwt-token",
        type=str,
        required=True,
        help="JWT token for authentication (required)",
    )

    # Model and provider arguments
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "bedrock"],
        default="bedrock",
        help="Model provider to use (default: bedrock)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        help="Model ID to use",
    )

    # Prompt arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Initial prompt to send to the agent",
    )

    # Interactive mode argument
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enable interactive mode for multi-turn conversations",
    )

    # Verbose logging argument
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose HTTP debugging output",
    )

    args = parser.parse_args()

    # Enable verbose logging if requested
    if args.verbose:
        enable_verbose_logging()

    return args


_SAFE_OPERATORS: dict = {
    ast.Add: _operator.add,
    ast.Sub: _operator.sub,
    ast.Mult: _operator.mul,
    ast.Div: _operator.truediv,
    ast.Pow: _operator.pow,
    ast.FloorDiv: _operator.floordiv,
    ast.Mod: _operator.mod,
}

_SAFE_UNARY_OPERATORS: dict = {
    ast.UAdd: _operator.pos,
    ast.USub: _operator.neg,
}


def _safe_eval_arithmetic(expression: str) -> Union[int, float]:
    """Safely evaluate an arithmetic expression using AST node whitelisting.

    Only numeric literals and basic arithmetic operators are permitted.
    Function calls, attribute access, names, and all other non-arithmetic
    constructs raise ValueError immediately.

    Args:
        expression: A pre-validated arithmetic expression string.

    Returns:
        The numeric result of the expression.

    Raises:
        ValueError: If the expression contains unsupported operations.
        ZeroDivisionError: If the expression divides by zero.
    """

    def _eval_node(node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op_func = _SAFE_OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

            # Special handling for exponentiation to prevent DoS
            if isinstance(node.op, ast.Pow):
                left_val = _eval_node(node.left)
                right_val = _eval_node(node.right)
                if abs(right_val) > 100:
                    raise ValueError("Exponent too large (max 100)")
                return op_func(left_val, right_val)

            return op_func(_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op_func = _SAFE_UNARY_OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_func(_eval_node(node.operand))
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.

    This tool can perform basic arithmetic operations like addition, subtraction,
    multiplication, division, and exponentiation.

    Args:
        expression (str): The mathematical expression to evaluate (e.g., "2 + 2", "5 * 10", "(3 + 4) / 2")

    Returns:
        str: The result of the evaluation as a string

    Example:
        calculator("2 + 2") -> "4"
        calculator("5 * 10") -> "50"
        calculator("(3 + 4) / 2") -> "3.5"
    """
    # Security check: only allow basic arithmetic operations and numbers
    # Remove all whitespace
    expression = expression.replace(" ", "")

    # Guard against excessively long expressions (DoS via large exponents)
    if len(expression) > 200:
        return "Error: Expression too long (max 200 characters)."

    # Check if the expression contains only allowed characters
    if not re.match(r"^[0-9+\-*/().^ ]+$", expression):
        return "Error: Only basic arithmetic operations (+, -, *, /, ^, (), .) are allowed."

    try:
        # Replace ^ with ** for exponentiation
        expression = expression.replace("^", "**")

        # Safely evaluate using AST node whitelisting (no arbitrary code execution)
        result = _safe_eval_arithmetic(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
async def search_registry_tools(
    query: str,
    max_results: int = 10,
) -> str:
    """
    Search for MCP tools using semantic search on the registry.

    Use this tool to discover available MCP tools that can help accomplish a task.
    The search uses natural language understanding to find the most relevant tools.

    Args:
        query (str): Natural language description of the capability you need
            (e.g., "get current time", "search jira issues", "manage files")
        max_results (int): Maximum number of results to return (default: 10)

    Returns:
        str: JSON string containing matching tools with their details including:
            - tool_name: Name of the tool
            - server_path: Path to invoke the tool on
            - server_name: Human-readable server name
            - description: What the tool does
            - relevance_score: How well it matches your query (0-1)
            - supported_transports: Transport protocols supported
            - auth_provider: Authentication provider if needed
            - tool_schema: Input parameters for the tool

    Example:
        search_registry_tools("get the current time in different timezones")
        search_registry_tools("search for jira issues", max_results=5)
    """
    global registry_client

    if registry_client is None:
        return json.dumps(
            {"error": "Registry client not initialized. Check authentication configuration."}
        )

    try:
        logger.info(f"Searching registry for tools: '{query}' (max_results={max_results})")

        # Search for tools using semantic search
        search_response = await registry_client.search_tools(
            query=query,
            max_results=max_results,
            entity_types=["mcp_server", "tool"],
        )

        results = []

        # Process tool results first (most specific)
        # The search API now returns inputSchema directly, no need for get_server_info
        for tool_result in search_response.tools:
            formatted = _format_tool_result(tool_result)
            results.append(formatted)

        # Also include matching tools from server results
        for server_result in search_response.servers:
            for matching_tool in server_result.matching_tools:
                # Check if this tool is already in results
                existing = [
                    r
                    for r in results
                    if r["tool_name"] == matching_tool.tool_name
                    and r["server_path"] == server_result.path
                ]
                if existing:
                    continue

                tool_data = {
                    "tool_name": matching_tool.tool_name,
                    "server_path": server_result.path,
                    "server_name": server_result.server_name,
                    "description": matching_tool.description or "No description available",
                    "relevance_score": matching_tool.relevance_score,
                    "supported_transports": ["streamable_http"],
                }
                # Note: inputSchema is available in the tools[] array, not matching_tools

                results.append(tool_data)

        # Sort by relevance score
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Limit to max_results
        results = results[:max_results]

        logger.info(f"Found {len(results)} matching tools for query: '{query}'")

        return json.dumps(
            {
                "query": query,
                "tools": results,
                "total_found": len(results),
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Error searching registry: {e}", exc_info=True)
        return json.dumps({"error": f"Search failed: {str(e)}"})


@tool
async def invoke_mcp_tool(
    mcp_registry_url: str,
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any],
    supported_transports: List[str] = None,
    auth_provider: str = None,
) -> str:
    """
    Invoke a tool on an MCP server using the MCP Registry URL and server name.

    Args:
        mcp_registry_url: The URL of the MCP Registry
        server_name: The name of the MCP server to connect to
        tool_name: The name of the tool to invoke
        arguments: Dictionary containing the arguments for the tool
        supported_transports: Transport protocols (["streamable_http"] or ["sse"])
        auth_provider: Authentication provider (e.g., "atlassian")

    Returns:
        The result of the tool invocation as a string
    """
    # Build server URL from registry URL and server name
    parsed_url = urlparse(mcp_registry_url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

    # Remove leading slash from server_name if present
    server_name_clean = server_name.lstrip("/")
    server_url = urljoin(base_url + "/", server_name_clean)

    # Build headers with authentication
    auth_token = agent_settings.auth_token
    region = agent_settings.region

    headers = {
        "X-Authorization": f"Bearer {auth_token}",
        "X-Region": region,
        "Authorization": f"Bearer {auth_token}",
    }

    # Get server-specific headers from configuration
    server_headers = get_server_headers(server_name_clean, server_config)
    headers.update(server_headers)

    # Handle egress authentication if auth_provider is specified
    if auth_provider:
        headers = _add_egress_auth(headers, auth_provider, server_name_clean)

    # Determine transport (default to streamable_http)
    use_sse = (
        supported_transports
        and "sse" in supported_transports
        and "streamable_http" not in supported_transports
    )

    if use_sse:
        server_url = server_url.rstrip("/") + "/sse"

    logger.info(f"Invoking {tool_name} on {server_name_clean}")

    # Try invocation, retry with /mcp suffix on failure
    try:
        if use_sse:
            return await _invoke_via_sse(server_url, headers, tool_name, arguments)
        else:
            return await _invoke_via_http(server_url, headers, tool_name, arguments)
    except Exception as e:
        # Always retry with /mcp suffix on first failure
        mcp_url = server_url.rstrip("/") + "/mcp"
        logger.info(f"First attempt failed, retrying with /mcp suffix: {mcp_url}")
        try:
            if use_sse:
                return await _invoke_via_sse(mcp_url, headers, tool_name, arguments)
            else:
                return await _invoke_via_http(mcp_url, headers, tool_name, arguments)
        except Exception as retry_e:
            logger.error(f"Error invoking MCP tool (retry): {retry_e}")
            return f"Error invoking MCP tool: {str(retry_e)}"


def _add_egress_auth(
    headers: Dict[str, str],
    auth_provider: str,
    server_name: str,
) -> Dict[str, str]:
    """Add egress authentication headers if available."""
    oauth_tokens_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".oauth-tokens"
    )
    server_lower = server_name.lower()
    provider_lower = auth_provider.lower()

    # Try provider-server specific file, then provider-only file
    egress_files = [
        os.path.join(oauth_tokens_dir, f"{provider_lower}-{server_lower}-egress.json"),
        os.path.join(oauth_tokens_dir, f"{provider_lower}-egress.json"),
    ]

    for egress_file in egress_files:
        if os.path.exists(egress_file):
            with open(egress_file, "r") as f:
                egress_data = json.load(f)

            egress_token = egress_data.get("access_token")
            if egress_token:
                headers["Authorization"] = f"Bearer {egress_token}"
                logger.info(f"Using egress auth for {auth_provider}")

            # Provider-specific headers
            if provider_lower == "atlassian":
                cloud_id = egress_data.get("cloud_id")
                if cloud_id:
                    headers["X-Atlassian-Cloud-Id"] = cloud_id

            break

    return headers


async def _invoke_via_sse(
    server_url: str,
    headers: Dict[str, str],
    tool_name: str,
    arguments: Dict[str, Any],
) -> str:
    """Invoke tool via SSE transport."""
    async with sse_client(server_url, headers=headers) as (read, write):
        async with mcp.ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return _format_tool_response(result)


async def _invoke_via_http(
    server_url: str,
    headers: Dict[str, str],
    tool_name: str,
    arguments: Dict[str, Any],
) -> str:
    """Invoke tool via streamable HTTP transport."""
    async with httpx.AsyncClient(headers=headers) as http_client:
        async with streamable_http_client(
            url=server_url,
            http_client=http_client,
        ) as (read, write, _):
            async with mcp.ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return _format_tool_response(result)


def _format_tool_response(result: Any) -> str:
    """Format MCP tool result as string."""
    response_parts = []
    for r in result.content:
        if hasattr(r, "text"):
            response_parts.append(r.text)
    return "\n".join(response_parts).strip()


# Get current UTC time (using timezone.utc to avoid deprecation warning)
current_utc_time = str(datetime.now(timezone.utc))


# Global agent settings to store authentication details
class AgentSettings:
    """Stores authentication details for MCP tool invocation."""

    def __init__(self):
        self.auth_token: Optional[str] = None
        self.region: str = "us-east-1"


agent_settings = AgentSettings()

# Global server configuration
server_config = {}


def load_system_prompt():
    """
    Load the system prompt template from the system_prompt.txt file.

    Returns:
        str: The system prompt template
    """
    import os

    try:
        # Get the directory where this Python file is located
        current_dir = os.path.dirname(__file__)
        system_prompt_path = os.path.join(current_dir, "system_prompt.txt")
        with open(system_prompt_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading system prompt: {e}")
        # Provide a minimal fallback prompt in case the file can't be loaded
        return """
        <instructions>
        You are a highly capable AI assistant designed to solve problems for users.
        Current UTC time: {current_utc_time}
        MCP Registry URL: {mcp_registry_url}
        </instructions>
        """


def print_agent_response(
    response_dict: Dict[str, Any],
    verbose: bool = False,
) -> None:
    """
    Print the agent's final response.

    Args:
        response_dict: Dictionary containing the agent response with 'messages' key
        verbose: Whether to show detailed message flow
    """
    if not response_dict or "messages" not in response_dict:
        return

    messages = response_dict["messages"]

    # In verbose mode, show the message flow
    if verbose:
        _print_verbose_messages(messages)

    # Find and print the final AI response
    for message in reversed(messages):
        message_type = type(message).__name__

        if "AIMessage" in message_type:
            content = getattr(message, "content", None)
            if content:
                print("\n" + str(content), flush=True)
            break


def _print_verbose_messages(messages: List[Any]) -> None:
    """Print detailed message flow for debugging."""
    colors = {
        "SYSTEM": "\033[1;33m",
        "HUMAN": "\033[1;32m",
        "AI": "\033[1;36m",
        "TOOL": "\033[1;35m",
        "RESET": "\033[0m",
    }

    print(f"\n{colors['AI']}=== Message Flow ({len(messages)} messages) ==={colors['RESET']}\n")

    for i, message in enumerate(messages, 1):
        msg_type = type(message).__name__
        color = colors.get(
            "AI" if "AI" in msg_type else "TOOL" if "Tool" in msg_type else "HUMAN", colors["RESET"]
        )

        content = getattr(message, "content", str(message))
        preview = content[:100] + "..." if len(str(content)) > 100 else content

        print(f"{color}[{i}] {msg_type}: {preview}{colors['RESET']}")

        # Show tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                print(f"     -> Tool: {tc.get('name', 'unknown')}")


class InteractiveAgent:
    """Interactive agent that maintains conversation history."""

    def __init__(
        self,
        agent,
        system_prompt: str,
        verbose: bool = False,
    ):
        self.agent = agent
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.conversation_history: List[Dict[str, str]] = []

    async def process_message(
        self,
        user_input: str,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Process a user message and return the agent's response."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})

        spinner = None
        if show_progress:
            spinner = ProgressSpinner().start()

        try:
            response = await self.agent.ainvoke({"messages": messages})
        finally:
            if spinner:
                spinner.stop()

        # Update history
        self.conversation_history.append({"role": "user", "content": user_input})

        if response and "messages" in response:
            for message in reversed(response["messages"]):
                if "AIMessage" in type(message).__name__:
                    ai_content = getattr(message, "content", str(message))
                    self.conversation_history.append({"role": "assistant", "content": ai_content})
                    break

        return response

    async def run_interactive_session(self) -> None:
        """Run an interactive conversation session."""
        print("\n" + "=" * 60)
        print("Interactive Agent Session")
        print("=" * 60)
        print("Commands: 'exit' to quit, 'clear' to reset, 'history' to view")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nGoodbye!")
                    break

                if user_input.lower() in ["clear", "reset"]:
                    self.conversation_history = []
                    print("History cleared.")
                    continue

                if user_input.lower() == "history":
                    self._print_history()
                    continue

                if not user_input:
                    continue

                response = await self.process_message(user_input)
                print("\nAgent:", end="")
                print_agent_response(response, self.verbose)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\nError: {str(e)}")
                if self.verbose:
                    import traceback

                    traceback.print_exc()

    def _print_history(self) -> None:
        """Print conversation history."""
        if not self.conversation_history:
            print("No history yet.")
            return

        print("\nConversation History:")
        print("-" * 40)
        for i, msg in enumerate(self.conversation_history, 1):
            role = "You" if msg["role"] == "user" else "Agent"
            preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            print(f"{i}. {role}: {preview}")


async def main():
    """Main function - parses args, sets up model, and runs agent."""
    args = parse_arguments()

    # Set up authentication
    agent_settings.auth_token = args.jwt_token

    # Load server configuration
    global server_config
    server_config = load_server_config()

    # Show startup info
    print_step(f"Registry: {args.mcp_registry_url}")
    print_step(f"Provider: {args.provider}")
    print_step(f"Model: {args.model}")

    # Initialize model
    model = _create_model(args.provider, args.model)
    if not model:
        return

    try:
        # Initialize registry client
        global registry_client
        parsed_url = urlparse(args.mcp_registry_url)
        registry_base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        registry_client = RegistryClient(
            registry_url=registry_base_url,
            jwt_token=args.jwt_token,
        )

        # Create the agent
        all_tools = [calculator, search_registry_tools, invoke_mcp_tool]
        agent = create_react_agent(model, all_tools)

        # Load system prompt
        system_prompt = load_system_prompt().format(
            current_utc_time=current_utc_time,
            mcp_registry_url=args.mcp_registry_url,
        )

        interactive_agent = InteractiveAgent(agent, system_prompt, args.verbose)

        # Process initial prompt if provided
        if args.prompt:
            print_step("Processing prompt...")
            response = await interactive_agent.process_message(args.prompt)

            if not args.interactive:
                print_agent_response(response, args.verbose)
                return
            else:
                print("\nAgent:", end="")
                print_agent_response(response, args.verbose)

        # Run interactive session or show usage
        if args.interactive:
            await interactive_agent.run_interactive_session()
        elif not args.prompt:
            print("\nNo prompt provided. Use --prompt or --interactive")
            print("\nExamples:")
            print('  python agent.py --prompt "What time is it?"')
            print("  python agent.py --interactive")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()


def _create_model(
    provider: str,
    model_id: str,
):
    """Create the LLM model based on provider."""
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found")
            return None

        return ChatAnthropic(
            model=model_id,
            api_key=api_key,
            temperature=0,
            max_tokens=8192,
        )

    # Default to Bedrock
    aws_region = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1"))
    return ChatBedrock(
        model_id=model_id,
        region_name=aws_region,
        temperature=0,
        max_tokens=8192,
    )


if __name__ == "__main__":
    asyncio.run(main())
