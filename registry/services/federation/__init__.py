"""
Federation services for integrating with external registries.

Supports federation with:
- Anthropic MCP Registry
- Workday ASOR (Agent Service Operating Registry)
- AWS Agent Registry
"""

from .agentcore_client import AgentCoreFederationClient
from .anthropic_client import AnthropicFederationClient
from .asor_client import AsorFederationClient
from .base_client import BaseFederationClient

__all__ = [
    "AgentCoreFederationClient",
    "AnthropicFederationClient",
    "AsorFederationClient",
    "BaseFederationClient",
]
