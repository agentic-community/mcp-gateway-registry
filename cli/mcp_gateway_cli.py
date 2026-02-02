#!/usr/bin/env python3
"""MCP Gateway CLI - Authentication and management tool."""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import msal
import requests
import typer
from rich.console import Console
from rich.status import Status
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="mcp-gateway",
    help="MCP Gateway CLI - Authenticate and interact with MCP Gateway",
)
console = Console()

# Sub-apps for command groups
mcp_app = typer.Typer(help="MCP server management commands")
agent_app = typer.Typer(help="A2A agent management commands")

# Register sub-apps
app.add_typer(mcp_app, name="mcp")
app.add_typer(agent_app, name="agent")

# Config locations (in cli/ folder of project)
CONFIG_DIR = Path(__file__).parent
SESSION_FILE = CONFIG_DIR / ".mcp_session"
CONFIG_FILE = CONFIG_DIR / ".mcp_config.json"


def _load_config() -> dict:
    """Load configuration from file and environment.

    Returns:
        Dictionary with configuration values from environment and config file.
    """
    config = {
        "tenant_id": os.environ.get("ENTRA_TENANT_ID", ""),
        "client_id": os.environ.get("ENTRA_CLIENT_ID", ""),
        "auth_server_url": os.environ.get("MCP_AUTH_SERVER_URL", "http://localhost:8888"),
        "registry_url": os.environ.get("MCP_REGISTRY_URL", "http://localhost:7860"),
    }

    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            file_config = json.load(f)
            config.update({k: v for k, v in file_config.items() if v})

    return config


def _save_config(
    config: dict,
) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary to save.
    """
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    logger.debug(f"Configuration saved to {CONFIG_FILE}")


def _get_session() -> Optional[str]:
    """Load stored session cookie.

    Returns:
        Session cookie string if exists, None otherwise.
    """
    if SESSION_FILE.exists():
        return SESSION_FILE.read_text().strip()
    return None


def _save_session(
    session_cookie: str,
) -> None:
    """Save session cookie to file with secure permissions.

    Args:
        session_cookie: Session cookie string to save.
    """
    SESSION_FILE.write_text(session_cookie)
    SESSION_FILE.chmod(0o600)
    logger.debug(f"Session saved to {SESSION_FILE}")


def _clear_session() -> bool:
    """Clear stored session.

    Returns:
        True if session was cleared, False if no session existed.
    """
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()
        return True
    return False


def _make_request(
    method: str,
    endpoint: str,
    params: Optional[dict] = None,
) -> requests.Response:
    """Make authenticated request to registry API.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path (e.g., /api/servers)
        params: Optional query parameters

    Returns:
        Response object from the request

    Raises:
        typer.Exit: If not logged in, session expired, or connection fails
    """
    config = _load_config()
    session = _get_session()

    if not session:
        console.print("[yellow]Not logged in.[/yellow] Run: mcp-gateway login")
        raise typer.Exit(1)

    url = f"{config['registry_url']}{endpoint}"
    try:
        resp = requests.request(
            method,
            url,
            params=params,
            cookies={"mcp_gateway_session": session},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error:[/red] Failed to connect: {e}")
        raise typer.Exit(1)

    if resp.status_code == 401:
        console.print("[yellow]Session expired.[/yellow] Run: mcp-gateway login")
        raise typer.Exit(1)

    return resp


@app.command()
def login(
    tenant_id: Optional[str] = typer.Option(
        None,
        "--tenant-id",
        "-t",
        envvar="ENTRA_TENANT_ID",
        help="Azure AD tenant ID",
    ),
    client_id: Optional[str] = typer.Option(
        None,
        "--client-id",
        "-c",
        envvar="ENTRA_CLIENT_ID",
        help="App registration client ID",
    ),
    auth_server: Optional[str] = typer.Option(
        None,
        "--auth-server",
        envvar="MCP_AUTH_SERVER_URL",
        help="MCP Auth server URL",
    ),
) -> None:
    """Authenticate with MCP Gateway using device code flow."""
    config = _load_config()
    tenant_id = tenant_id or config.get("tenant_id")
    client_id = client_id or config.get("client_id")
    auth_server = auth_server or config.get("auth_server_url")

    if not tenant_id or not client_id:
        console.print("[red]Error:[/red] ENTRA_TENANT_ID and ENTRA_CLIENT_ID are required")
        console.print("Set via environment variables or --tenant-id/--client-id options")
        raise typer.Exit(1)

    logger.debug(f"Using tenant_id: {tenant_id}")
    logger.debug(f"Using auth_server: {auth_server}")

    # Step 1: Initiate device code flow
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    msal_app = msal.PublicClientApplication(client_id, authority=authority)

    # Note: MSAL automatically includes 'openid', 'profile', 'offline_access'
    # We only need to request 'email' explicitly
    flow = msal_app.initiate_device_flow(scopes=["email"])
    if "error" in flow:
        console.print(f"[red]Error:[/red] {flow.get('error_description', flow.get('error'))}")
        raise typer.Exit(1)

    # Step 2: Show instructions
    console.print("\n[bold]To sign in:[/bold]")
    console.print(f"  1. Open: [cyan]{flow['verification_uri']}[/cyan]")
    console.print(f"  2. Enter code: [bold yellow]{flow['user_code']}[/bold yellow]\n")

    # Step 3: Wait for authentication
    with Status("[bold blue]Waiting for authentication...", console=console):
        result = msal_app.acquire_token_by_device_flow(flow)

    if "error" in result:
        console.print(
            f"[red]Authentication failed:[/red] "
            f"{result.get('error_description', result.get('error'))}"
        )
        raise typer.Exit(1)

    id_token = result.get("id_token")
    if not id_token:
        console.print("[red]Error:[/red] No ID token received")
        raise typer.Exit(1)

    # Step 4: Exchange for session cookie
    with Status("[bold blue]Exchanging token...", console=console):
        try:
            resp = requests.post(
                f"{auth_server}/oauth2/exchange-token",
                json={"id_token": id_token},
                timeout=10,
            )
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error:[/red] Failed to connect to auth server: {e}")
            raise typer.Exit(1)

    if resp.status_code != 200:
        console.print(f"[red]Error:[/red] Token exchange failed: {resp.text}")
        raise typer.Exit(1)

    data = resp.json()
    session_cookie = data.get("session_cookie")
    if not session_cookie:
        console.print("[red]Error:[/red] No session cookie received")
        raise typer.Exit(1)

    # Step 5: Save session
    _save_session(session_cookie)

    username = data.get("username", "Unknown")
    console.print(f"[green]Logged in as:[/green] {username}")
    console.print("[dim]Session saved to cli/.mcp_session[/dim]")


@app.command()
def whoami(
    registry_url: Optional[str] = typer.Option(
        None,
        "--registry-url",
        "-r",
        envvar="MCP_REGISTRY_URL",
        help="MCP Registry URL",
    ),
) -> None:
    """Show current authenticated user."""
    config = _load_config()
    registry_url = registry_url or config.get("registry_url")
    session = _get_session()

    if not session:
        console.print("[yellow]Not logged in.[/yellow] Run: mcp-gateway login")
        raise typer.Exit(1)

    try:
        resp = requests.get(
            f"{registry_url}/api/auth/me",
            cookies={"mcp_gateway_session": session},
            timeout=10,
        )
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error:[/red] Failed to connect to registry: {e}")
        raise typer.Exit(1)

    if resp.status_code == 401:
        console.print("[yellow]Session expired.[/yellow] Run: mcp-gateway login")
        raise typer.Exit(1)

    if resp.status_code != 200:
        console.print(f"[red]Error:[/red] {resp.text}")
        raise typer.Exit(1)

    user = resp.json()
    console.print(f"[bold]Username:[/bold] {user.get('username', 'Unknown')}")
    console.print(f"[bold]Groups:[/bold] {', '.join(user.get('groups', []))}")
    console.print(f"[bold]Scopes:[/bold] {', '.join(user.get('scopes', []))}")


@app.command()
def logout() -> None:
    """Clear stored session."""
    if _clear_session():
        console.print("[green]Logged out.[/green]")
    else:
        console.print("[dim]No active session.[/dim]")


@app.command()
def config(
    set_value: Optional[str] = typer.Option(
        None,
        "--set",
        help="Set config: key=value",
    ),
) -> None:
    """View or modify CLI configuration.

    Without options, shows current configuration.
    Use --set key=value to set a configuration value.
    """
    if set_value:
        key, _, value = set_value.partition("=")
        if not value:
            console.print("[red]Error:[/red] Use --set key=value")
            raise typer.Exit(1)

        existing = {}
        if CONFIG_FILE.exists():
            existing = json.loads(CONFIG_FILE.read_text())
        existing[key] = value
        _save_config(existing)
        console.print(f"[green]Set {key}[/green]")
        return

    # Show config (default behavior)
    cfg = _load_config()
    console.print("[bold]Current configuration:[/bold]")
    for key, value in cfg.items():
        display = value if value else "[dim]not set[/dim]"
        console.print(f"  {key}: {display}")


@mcp_app.command("list")
def mcp_list(
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Search filter",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json",
    ),
) -> None:
    """List available MCP servers."""
    params = {}
    if query:
        params["query"] = query

    resp = _make_request("GET", "/api/servers", params=params)

    if resp.status_code != 200:
        console.print(f"[red]Error:[/red] {resp.text}")
        raise typer.Exit(1)

    data = resp.json()
    servers = data.get("servers", [])

    if output_format == "json":
        console.print_json(json.dumps(servers, indent=2))
        return

    if not servers:
        console.print("[dim]No servers found.[/dim]")
        return

    table = Table(title="MCP Servers")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Tools", justify="right")
    table.add_column("Status", style="green")

    for server in servers:
        health_status = server.get("health_status", "unknown")
        status_style = "green" if health_status == "healthy" else "red"
        table.add_row(
            server.get("display_name", ""),
            server.get("path", ""),
            str(server.get("num_tools", 0)),
            f"[{status_style}]{health_status}[/{status_style}]",
        )

    console.print(table)


@agent_app.command("list")
def agent_list(
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Search filter",
    ),
    enabled_only: bool = typer.Option(
        False,
        "--enabled",
        "-e",
        help="Show only enabled agents",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json",
    ),
) -> None:
    """List available A2A agents."""
    params = {}
    if query:
        params["query"] = query
    if enabled_only:
        params["enabled_only"] = "true"

    resp = _make_request("GET", "/api/agents", params=params)

    if resp.status_code != 200:
        console.print(f"[red]Error:[/red] {resp.text}")
        raise typer.Exit(1)

    data = resp.json()
    agents = data.get("agents", [])

    if output_format == "json":
        console.print_json(json.dumps(agents, indent=2))
        return

    if not agents:
        console.print("[dim]No agents found.[/dim]")
        return

    table = Table(title="A2A Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Skills", justify="right")
    table.add_column("Trust", style="yellow")
    table.add_column("Enabled")

    for agent in agents:
        enabled_icon = "[green]Yes[/green]" if agent.get("isEnabled") else "[red]No[/red]"
        table.add_row(
            agent.get("name", ""),
            agent.get("path", ""),
            str(agent.get("numSkills", 0)),
            agent.get("trustLevel", "unknown"),
            enabled_icon,
        )

    console.print(table)


if __name__ == "__main__":
    app()
