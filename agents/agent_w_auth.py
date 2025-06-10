#!/usr/bin/env python3
"""
LangGraph MCP Client with Multi-method Authentication

This script demonstrates using LangGraph with the MultiServerMCPClient adapter to connect to an
MCP-compatible server with multiple authentication methods and query information using a Bedrock-hosted Claude model.

Supported authentication methods:
1. Cognito M2M authentication (machine-to-machine)
2. Session cookie authentication (from CLI auth or OAuth2 login)
3. GitHub OAuth2 authentication (via CLI auth tool)

The script accepts command line arguments for:
- Server host and port
- Model ID to use
- User message to process
- Authentication method and parameters

Configuration can be provided via command line arguments or environment variables.
Command line arguments take precedence over environment variables.

Environment Variables:
- COGNITO_CLIENT_ID: Cognito App Client ID
- COGNITO_CLIENT_SECRET: Cognito App Client Secret
- COGNITO_USER_POOL_ID: Cognito User Pool ID
- AWS_REGION: AWS region for Cognito
- GITHUB_CLIENT_ID: GitHub OAuth2 App Client ID
- GITHUB_CLIENT_SECRET: GitHub OAuth2 App Client Secret

Usage Examples:

1. Cognito M2M Authentication:
    python agent_w_auth.py --mcp-registry-url URL --model model_id --message "your question" \
        --client-id CLIENT_ID --client-secret CLIENT_SECRET --user-pool-id USER_POOL_ID --region REGION

2. Session Cookie Authentication:
    python agent_w_auth.py --use-session-cookie --message "your question"

3. GitHub OAuth2 Authentication:
    python agent_w_auth.py --use-github-auth --message "your question"

Example with environment variables (create a .env file):
    COGNITO_CLIENT_ID=your_client_id
    COGNITO_CLIENT_SECRET=your_client_secret
    COGNITO_USER_POOL_ID=your_user_pool_id
    AWS_REGION=us-east-1
    GITHUB_CLIENT_ID=your_github_client_id
    GITHUB_CLIENT_SECRET=your_github_client_secret
    
    python agent_w_auth.py --use-github-auth --message "current time in new delhi"
"""

import asyncio
import argparse
import re
import sys
import os
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse, urljoin
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_core.tools import tool
import mcp
from mcp import ClientSession
from mcp.client.sse import sse_client

# Import dotenv for loading environment variables
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Environment file loading disabled.")

# Add the auth_server directory to the path to import cognito_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'auth_server'))
from cognito_utils import generate_token

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)

# Get logger
logger = logging.getLogger(__name__)

def load_env_config() -> Dict[str, Optional[str]]:
    """
    Load configuration from .env file if available.
    
    Returns:
        Dict[str, Optional[str]]: Dictionary containing environment variables
    """
    env_config = {
        'client_id': None,
        'client_secret': None,
        'region': None,
        'user_pool_id': None,
        'domain': None,
        'github_client_id': None,
        'github_client_secret': None,
        'secret_key': None
    }
    
    if DOTENV_AVAILABLE:
        # Try to load from .env file in the current directory
        env_file = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loading environment variables from {env_file}")
        else:
            # Try to load from .env file in the parent directory
            env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
            if os.path.exists(env_file):
                load_dotenv(env_file)
                logger.info(f"Loading environment variables from {env_file}")
            else:
                # Try to load from current working directory
                load_dotenv()
                logger.info("Loading environment variables from current directory")
        
        # Get values from environment - Cognito
        env_config['client_id'] = os.getenv('COGNITO_CLIENT_ID')
        env_config['client_secret'] = os.getenv('COGNITO_CLIENT_SECRET')
        env_config['region'] = os.getenv('AWS_REGION')
        env_config['user_pool_id'] = os.getenv('COGNITO_USER_POOL_ID')
        env_config['domain'] = os.getenv('COGNITO_DOMAIN')
        
        # Get values from environment - GitHub OAuth2
        env_config['github_client_id'] = os.getenv('GITHUB_CLIENT_ID')
        env_config['github_client_secret'] = os.getenv('GITHUB_CLIENT_SECRET')
        env_config['secret_key'] = os.getenv('SECRET_KEY')
    
    return env_config

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the LangGraph MCP client with Cognito authentication.
    Command line arguments take precedence over environment variables.
    
    Returns:
        argparse.Namespace: The parsed command line arguments
    """
    # Load environment configuration first
    env_config = load_env_config()
    
    parser = argparse.ArgumentParser(description='LangGraph MCP Client with Cognito Authentication')
    
    # Server connection arguments
    parser.add_argument('--mcp-registry-url', type=str, default='https://mcpgateway.ddns.net/mcpgw/sse',
                        help='Hostname of the MCP Registry')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='us.anthropic.claude-3-5-haiku-20241022-v1:0',
                        help='Model ID to use with Bedrock')
    
    # Message arguments
    parser.add_argument('--message', type=str, default='what is the current time in Clarksburg, MD',
                        help='Message to send to the agent')
    
    # Authentication method arguments
    parser.add_argument('--use-session-cookie', action='store_true',
                        help='Use session cookie authentication instead of M2M')
    parser.add_argument('--session-cookie-file', type=str, default='~/.mcp/session_cookie',
                        help='Path to session cookie file (default: ~/.mcp/session_cookie)')
    parser.add_argument('--use-github-auth', action='store_true',
                        help='Use GitHub OAuth2 authentication (requires CLI auth setup)')
    parser.add_argument('--github-client-id', type=str, default=env_config['github_client_id'],
                        help='GitHub OAuth2 Client ID (can be set via GITHUB_CLIENT_ID env var)')
    parser.add_argument('--github-client-secret', type=str, default=env_config['github_client_secret'],
                        help='GitHub OAuth2 Client Secret (can be set via GITHUB_CLIENT_SECRET env var)')
    parser.add_argument('--secret-key', type=str, default=env_config['secret_key'],
                        help='Secret key for session cookie validation (can be set via SECRET_KEY env var)')
    
    # Cognito authentication arguments - now optional if available in environment
    parser.add_argument('--client-id', type=str, default=env_config['client_id'],
                        help='Cognito App Client ID (can be set via COGNITO_CLIENT_ID env var)')
    parser.add_argument('--client-secret', type=str, default=env_config['client_secret'],
                        help='Cognito App Client Secret (can be set via COGNITO_CLIENT_SECRET env var)')
    parser.add_argument('--user-pool-id', type=str, default=env_config['user_pool_id'],
                        help='Cognito User Pool ID (can be set via COGNITO_USER_POOL_ID env var)')
    parser.add_argument('--region', type=str, default=env_config['region'],
                        help='AWS region for Cognito (can be set via AWS_REGION env var)')
    parser.add_argument('--domain', type=str, default=env_config['domain'],
                        help='Cognito custom domain (can be set via COGNITO_DOMAIN env var)')
    parser.add_argument('--scopes', type=str, nargs='*', default=None,
                        help='Optional scopes for the token request')
    
    args = parser.parse_args()
    
    # Validate authentication parameters based on method
    auth_methods = sum([args.use_session_cookie, args.use_github_auth])
    if auth_methods > 1:
        parser.error("Please specify only one authentication method: --use-session-cookie, --use-github-auth, or neither (for M2M)")
    
    if args.use_session_cookie:
        # For session cookie auth, we just need the cookie file
        cookie_path = os.path.expanduser(args.session_cookie_file)
        if not os.path.exists(cookie_path):
            parser.error(f"Session cookie file not found: {cookie_path}\n"
                        f"Run 'python auth_server/cli_auth.py' to authenticate first")
    elif args.use_github_auth:
        # For GitHub auth, validate GitHub parameters
        missing_params = []
        if not args.github_client_id:
            missing_params.append('--github-client-id (or GITHUB_CLIENT_ID env var)')
        if not args.github_client_secret:
            missing_params.append('--github-client-secret (or GITHUB_CLIENT_SECRET env var)')
        if not args.secret_key:
            missing_params.append('--secret-key (or SECRET_KEY env var)')
        
        if missing_params:
            parser.error(f"Missing required parameters for GitHub authentication: {', '.join(missing_params)}")
    else:
        # For M2M auth, validate Cognito parameters
        missing_params = []
        if not args.client_id:
            missing_params.append('--client-id (or COGNITO_CLIENT_ID env var)')
        if not args.client_secret:
            missing_params.append('--client-secret (or COGNITO_CLIENT_SECRET env var)')
        if not args.user_pool_id:
            missing_params.append('--user-pool-id (or COGNITO_USER_POOL_ID env var)')
        if not args.region:
            missing_params.append('--region (or AWS_REGION env var)')
        
        if missing_params:
            parser.error(f"Missing required parameters for M2M authentication: {', '.join(missing_params)}")
    
    return args

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
    
    # Check if the expression contains only allowed characters
    if not re.match(r'^[0-9+\-*/().^ ]+$', expression):
        return "Error: Only basic arithmetic operations (+, -, *, /, ^, (), .) are allowed."
    
    try:
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        
        # Evaluate the expression
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
async def invoke_mcp_tool(mcp_registry_url: str, server_name: str, tool_name: str, arguments: Dict[str, Any],
                         auth_token: str = None, user_pool_id: str = None, client_id: str = None, region: str = None,
                         auth_method: str = "m2m", session_cookie: str = None) -> str:
    """
    Invoke a tool on an MCP server using the MCP Registry URL and server name with authentication.
    
    This tool creates an MCP SSE client and calls the specified tool with the provided arguments.
    Supports both M2M (JWT) and session cookie authentication.
    
    Args:
        mcp_registry_url (str): The URL of the MCP Registry
        server_name (str): The name of the MCP server to connect to
        tool_name (str): The name of the tool to invoke
        arguments (Dict[str, Any]): Dictionary containing the arguments for the tool
        auth_token (str): Bearer token for authentication (for M2M)
        user_pool_id (str): Cognito User Pool ID for X-User-Pool-Id header
        client_id (str): Cognito Client ID for X-Client-Id header
        region (str): AWS region for X-Region header
        auth_method (str): Authentication method ("m2m" or "session_cookie")
        session_cookie (str): Session cookie value (for session auth)
    
    Returns:
        str: The result of the tool invocation as a string
    
    Example:
        invoke_mcp_tool("registry url", "currenttime", "current_time_by_timezone", {"tz_name": "America/New_York"}, "auth_token", "user_pool_id", "client_id", "region")
    """
    # Construct the MCP server URL from the registry URL and server name using standard URL parsing
    parsed_url = urlparse(mcp_registry_url)
    
    # Extract the scheme and netloc (hostname:port) from the parsed URL
    scheme = parsed_url.scheme
    netloc = parsed_url.netloc
    
    # Construct the base URL with scheme and netloc
    base_url = f"{scheme}://{netloc}"
    
    # Create the server URL by joining the base URL with the server name and sse path
    server_url = urljoin(base_url, f"{server_name}/sse")
    print(f"Server URL: {server_url}")
    
    # Prepare headers based on authentication method
    headers = {
        'X-User-Pool-Id': user_pool_id or '',
        'X-Client-Id': client_id or '',
        'X-Region': region or 'us-east-1'
    }
    
    if auth_method == "session_cookie" and session_cookie:
        headers['Cookie'] = f'mcp_gateway_session={session_cookie}'
        redacted_headers = {
            'Cookie': f'mcp_gateway_session={redact_sensitive_value(session_cookie)}',
            'X-User-Pool-Id': redact_sensitive_value(user_pool_id) if user_pool_id else '',
            'X-Client-Id': redact_sensitive_value(client_id) if client_id else '',
            'X-Region': region or 'us-east-1'
        }
    else:
        headers['Authorization'] = f'Bearer {auth_token}'
        redacted_headers = {
            'Authorization': f'Bearer {redact_sensitive_value(auth_token)}',
            'X-User-Pool-Id': redact_sensitive_value(user_pool_id) if user_pool_id else '',
            'X-Client-Id': redact_sensitive_value(client_id) if client_id else '',
            'X-Region': region or 'us-east-1'
        }
    
    try:
        # Create an MCP SSE client and call the tool with authentication headers
        #print(f"Connecting to MCP server: {server_url}, headers: {redacted_headers}")
        logger.info(f"Connecting to MCP server: {server_url}, headers: {redacted_headers}")
        async with mcp.client.sse.sse_client(server_url, headers=headers) as (read, write):
            async with mcp.ClientSession(read, write, sampling_callback=None) as session:
                # Initialize the connection
                await session.initialize()
                
                # Call the specified tool with the provided arguments
                result = await session.call_tool(tool_name, arguments=arguments)
                
                # Format the result as a string
                response = ""
                for r in result.content:
                    response += r.text + "\n"
                
                return response.strip()
    except Exception as e:
        return f"Error invoking MCP tool: {str(e)}"

from datetime import datetime, UTC
current_utc_time = str(datetime.now(UTC))

def redact_sensitive_value(value: str, show_chars: int = 4) -> str:
    """Redact sensitive values, showing only the first few characters"""
    if not value or len(value) <= show_chars:
        return "*" * len(value) if value else ""
    return value[:show_chars] + "*" * (len(value) - show_chars)

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

def print_agent_response(response_dict: Dict[str, Any]) -> None:
    """
    Parse and print all messages in the response with color coding

    Args:
        response_dict: Dictionary containing the agent response with 'messages' key
    """
    # Define ANSI color codes for different message types
    COLORS = {
        "SYSTEM": "\033[1;33m",  # Yellow
        "HUMAN": "\033[1;32m",   # Green
        "AI": "\033[1;36m",      # Cyan
        "TOOL": "\033[1;35m",    # Magenta
        "UNKNOWN": "\033[1;37m", # White
        "RESET": "\033[0m"       # Reset to default
    }
    if 'messages' not in response_dict:
        logger.warning("No messages found in response")
        return
    
    messages = response_dict['messages']
    blue = "\033[1;34m"  # Blue
    reset = COLORS["RESET"]
    logger.info(f"\n{blue}=== Found {len(messages)} messages ==={reset}\n")
    
    for i, message in enumerate(messages, 1):
        # Determine message type based on class name or type
        message_type = type(message).__name__
        
        if "SystemMessage" in message_type:
            msg_type = "SYSTEM"
        elif "HumanMessage" in message_type:
            msg_type = "HUMAN"
        elif "AIMessage" in message_type:
            msg_type = "AI"
        elif "ToolMessage" in message_type:
            msg_type = "TOOL"
        else:
            # Fallback to string matching if type name doesn't match expected patterns
            message_str = str(message)
            if "SystemMessage" in message_str:
                msg_type = "SYSTEM"
            elif "HumanMessage" in message_str:
                msg_type = "HUMAN"
            elif "AIMessage" in message_str:
                msg_type = "AI"
            elif "ToolMessage" in message_str:
                msg_type = "TOOL"
            else:
                msg_type = "UNKNOWN"
        
        # Get message content
        content = message.content if hasattr(message, 'content') else str(message)
        
        # Check for tool calls
        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                tool_args = tool_call.get('args', {})
                tool_calls.append(f"Tool: {tool_name}, Args: {tool_args}")
        
        # Get the color for this message type
        color = COLORS.get(msg_type, COLORS["UNKNOWN"])
        reset = COLORS["RESET"]
        
        # Log message with enhanced formatting and color coding - entire message in color
        logger.info(f"\n{color}{'=' * 20} MESSAGE #{i} - TYPE: {msg_type} {'=' * 20}")
        logger.info(f"{'-' * 80}")
        logger.info(f"CONTENT: {content}")
        
        # Log any tool calls
        if tool_calls:
            logger.info(f"\nTOOL CALLS:")
            for tc in tool_calls:
                logger.info(f"  {tc}")
        logger.info(f"{'=' * 20} END OF {msg_type} MESSAGE #{i} {'=' * 20}{reset}")
        logger.info("")

def perform_github_device_flow(github_client_id: str, github_client_secret: str, secret_key: str) -> str:
    """
    Perform GitHub device flow authentication.
    
    Args:
        github_client_id: GitHub OAuth2 App Client ID
        github_client_secret: GitHub OAuth2 App Client Secret  
        secret_key: Secret key for session cookie validation
        
    Returns:
        str: Session cookie value
        
    Raises:
        Exception: If authentication fails
    """
    import requests
    import time
    from itsdangerous import URLSafeTimedSerializer
    
    logger.info("Starting GitHub device flow authentication...")
    
    try:
        # Step 1: Request device and user codes
        device_response = requests.post(
            'https://github.com/login/device/code',
            data={
                'client_id': github_client_id,
                'scope': 'read:user user:email'
            },
            headers={'Accept': 'application/json'}
        )
        device_response.raise_for_status()
        device_data = device_response.json()
        
        device_code = device_data['device_code']
        user_code = device_data['user_code']
        verification_uri = device_data['verification_uri']
        interval = device_data.get('interval', 5)
        
        # Step 2: Display user code and instructions
        print(f"\nTo authenticate with GitHub:")
        print(f"1. Go to: {verification_uri}")
        print(f"2. Enter code: {user_code}")
        print("3. Complete the authorization in your browser")
        print("Waiting for authorization...")
        
        # Step 3: Poll for token
        access_token = None
        while True:
            token_response = requests.post(
                'https://github.com/login/oauth/access_token',
                data={
                    'client_id': github_client_id,
                    'device_code': device_code,
                    'grant_type': 'urn:ietf:params:oauth:grant-type:device_code'
                },
                headers={'Accept': 'application/json'}
            )
            token_response.raise_for_status()
            token_data = token_response.json()
            
            if 'access_token' in token_data:
                access_token = token_data['access_token']
                break
            elif token_data.get('error') == 'authorization_pending':
                time.sleep(interval)
                continue
            elif token_data.get('error') == 'slow_down':
                interval += 5
                time.sleep(interval)
                continue
            elif token_data.get('error') == 'expired_token':
                raise Exception("Device code expired. Please try again.")
            elif token_data.get('error') == 'access_denied':
                raise Exception("Authorization denied by user.")
            else:
                raise Exception(f"Unknown error: {token_data.get('error', 'Unknown')}")
        
        # Step 4: Get user information
        user_response = requests.get(
            'https://api.github.com/user',
            headers={
                'Authorization': f'token {access_token}',
                'Accept': 'application/json'
            }
        )
        user_response.raise_for_status()
        user_info = user_response.json()
        
        # Step 5: Create session cookie
        session_data = {
            "username": user_info.get('login'),
            "email": user_info.get('email'),
            "name": user_info.get('name'),
            "groups": [],
            "provider": "github",
            "auth_method": "device_flow"
        }
        
        signer = URLSafeTimedSerializer(secret_key)
        session_cookie = signer.dumps(session_data)
        
        logger.info(f"Successfully authenticated GitHub user: {user_info.get('login')}")
        return session_cookie
        
    except requests.RequestException as e:
        raise Exception(f"GitHub API error: {e}")
    except Exception as e:
        raise Exception(f"Device flow authentication failed: {e}")

async def main():
    """
    Main function that:
    1. Parses command line arguments
    2. Generates Cognito M2M authentication token OR loads session cookie
    3. Sets up the LangChain MCP client and Bedrock model with authentication
    4. Creates a LangGraph agent with available tools
    5. Invokes the agent with the provided message
    6. Displays the response
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Display configuration
    server_url = args.mcp_registry_url
    logger.info(f"Connecting to MCP server: {server_url}")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Message: {args.message}")
    # Determine authentication method
    if args.use_session_cookie:
        auth_method_name = "Session Cookie"
        auth_method = "session_cookie"
    elif args.use_github_auth:
        auth_method_name = "GitHub OAuth2"
        auth_method = "github_oauth2"
    else:
        auth_method_name = "M2M Token"
        auth_method = "m2m"
    
    logger.info(f"Authentication method: {auth_method_name}")
    
    # Initialize authentication variables
    access_token = None
    session_cookie = None
    
    if args.use_session_cookie:
        # Load session cookie from file
        try:
            cookie_path = os.path.expanduser(args.session_cookie_file)
            with open(cookie_path, 'r') as f:
                session_cookie = f.read().strip()
            logger.info(f"Successfully loaded session cookie from {cookie_path}")
        except Exception as e:
            logger.error(f"Failed to load session cookie: {e}")
            return
    elif args.use_github_auth:
        # Perform GitHub OAuth2 authentication
        try:
            session_cookie = perform_github_device_flow(
                args.github_client_id,
                args.github_client_secret,
                args.secret_key
            )
            logger.info("Successfully completed GitHub OAuth2 authentication")
        except Exception as e:
            logger.error(f"Failed to complete GitHub OAuth2 authentication: {e}")
            return
    else:
        # Generate Cognito M2M authentication token
        logger.info(f"Cognito User Pool ID: {redact_sensitive_value(args.user_pool_id)}")
        logger.info(f"Cognito Client ID: {redact_sensitive_value(args.client_id)}")
        logger.info(f"AWS Region: {args.region}")
        
        try:
            logger.info("Generating Cognito M2M authentication token...")
            token_data = generate_token(
                client_id=args.client_id,
                client_secret=args.client_secret,
                user_pool_id=args.user_pool_id,
                region=args.region,
                scopes=args.scopes,
                domain=args.domain
            )
            access_token = token_data.get('access_token')
            if not access_token:
                raise ValueError("No access token received from Cognito")
            logger.info("Successfully generated authentication token")
        except Exception as e:
            logger.error(f"Failed to generate authentication token: {e}")
            return
    
    # Initialize the model
    model = ChatBedrockConverse(model_id=args.model, region_name='us-east-1')
    
    try:
        # Prepare headers for MCP client authentication based on method
        if args.use_session_cookie or args.use_github_auth:
            auth_headers = {
                'Cookie': f'mcp_gateway_session={session_cookie}',
                'X-User-Pool-Id': args.user_pool_id or '',
                'X-Client-Id': args.client_id or '',
                'X-Region': args.region or 'us-east-1'
            }
        else:
            auth_headers = {
                'Authorization': f'Bearer {access_token}',
                'X-User-Pool-Id': args.user_pool_id,
                'X-Client-Id': args.client_id,
                'X-Region': args.region
            }
        
        # Log redacted headers
        redacted_headers = {}
        for k, v in auth_headers.items():
            if k in ['Authorization', 'Cookie', 'X-User-Pool-Id', 'X-Client-Id']:
                redacted_headers[k] = redact_sensitive_value(v) if v else ''
            else:
                redacted_headers[k] = v
        logger.info(f"Using authentication headers: {redacted_headers}")
        
        # Initialize MCP client with the server configuration and authentication headers
        client = MultiServerMCPClient(
            {
                "default_server": {
                    "url": server_url,
                    "transport": "sse",
                    "headers": auth_headers
                }
            }
        )
        logger.info("Connected to MCP server successfully with authentication")

        # Get available tools from MCP and display them
        mcp_tools = await client.get_tools()
        logger.info(f"Available MCP tools: {[tool.name for tool in mcp_tools]}")
        
        # Add the calculator and invoke_mcp_tool to the tools array
        # The invoke_mcp_tool function already supports authentication parameters
        all_tools = [calculator, invoke_mcp_tool] + mcp_tools
        logger.info(f"All available tools: {[tool.name if hasattr(tool, 'name') else tool.__name__ for tool in all_tools]}")
        
        # Create the agent with the model and all tools
        agent = create_react_agent(
            model,
            all_tools
        )
        
        # Load and format the system prompt with the current time and MCP registry URL
        system_prompt_template = load_system_prompt()
        
        # Prepare authentication parameters for system prompt
        if args.use_session_cookie or args.use_github_auth:
            system_prompt = system_prompt_template.format(
                current_utc_time=current_utc_time,
                mcp_registry_url=args.mcp_registry_url,
                auth_token='',  # Not used for session cookie auth
                user_pool_id=args.user_pool_id or '',
                client_id=args.client_id or '',
                region=args.region or 'us-east-1',
                auth_method=auth_method,
                session_cookie=session_cookie
            )
        else:
            system_prompt = system_prompt_template.format(
                current_utc_time=current_utc_time,
                mcp_registry_url=args.mcp_registry_url,
                auth_token=access_token,
                user_pool_id=args.user_pool_id,
                client_id=args.client_id,
                region=args.region,
                auth_method=auth_method,
                session_cookie=''  # Not used for M2M auth
            )
        
        # Format the message with system message first
        formatted_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.message}
        ]
        
        logger.info("\nInvoking agent...\n" + "-"*40)
        
        # Invoke the agent with the formatted messages
        response = await agent.ainvoke({"messages": formatted_messages})
        
        logger.info("\nResponse:" + "\n" + "-"*40)
        #print(response)
        print_agent_response(response)
        
        # Process and display the response
        if response and "messages" in response and response["messages"]:
            # Get the last message from the response
            last_message = response["messages"][-1]
            
            if isinstance(last_message, dict) and "content" in last_message:
                # Display the content of the response
                print(last_message["content"])
            else:
                print(str(last_message.content))
        else:
            print("No valid response received")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())