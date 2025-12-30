"""Compatibility wrapper for the agent service.

This module preserves the historical import path (`registry.services.agent_service`)
while keeping the implementation in `registry.services.agents.service`.
"""

from .agents.service import (
    AgentService,
    agent_service,
)

__all__ = [
    "AgentService",
    "agent_service",
]

