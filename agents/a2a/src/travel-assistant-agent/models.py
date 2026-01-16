"""Data models for Travel Assistant Agent."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentSkill(BaseModel):
    """Skill/capability of an agent."""

    id: str = Field(..., description="Skill identifier")
    name: str = Field(..., description="Skill name")
    description: Optional[str] = Field(None, description="Skill description")
    tags: List[str] = Field(default_factory=list, description="Skill tags")
    examples: Optional[List[str]] = Field(None, description="Usage examples")
    input_modes: Optional[List[str]] = Field(None, description="Supported input modes")
    output_modes: Optional[List[str]] = Field(None, description="Supported output modes")
    security: Optional[Dict[str, Any]] = Field(None, description="Security requirements")


class DiscoveredAgent(BaseModel):
    """Agent discovered from registry."""

    model_config = {"populate_by_name": True, "extra": "ignore"}

    name: str = Field(..., description="Agent name")
    description: str = Field(default="", description="Agent description")
    path: str = Field(..., description="Registry path")
    url: Optional[str] = Field(None, description="Agent endpoint URL for invocation")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    skills: List[AgentSkill] = Field(default_factory=list, description="Agent skills")
    is_enabled: bool = Field(False, description="Whether agent is enabled")
    trust_level: str = Field("unverified", description="Trust level")
    visibility: str = Field("public", description="Agent visibility")
    relevance_score: Optional[float] = Field(None, description="Relevance score from search")

    @property
    def agent_name(self) -> str:
        """Alias for name for backward compatibility."""
        return self.name

    @property
    def skill_names(self) -> List[str]:
        """Get list of skill names."""
        return [skill.name for skill in self.skills]
