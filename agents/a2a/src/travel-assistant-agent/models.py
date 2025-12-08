"""Data models for Travel Assistant Agent."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DiscoveredAgent(BaseModel):
    """Agent discovered from registry."""
    
    model_config = {"populate_by_name": True}

    agent_name: str = Field(..., description="Agent name")
    description: str = Field(default="", description="Agent description")
    path: str = Field(..., description="Registry path")
    url: Optional[str] = Field(None, description="Agent endpoint URL for invocation")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    skills: List[str] = Field(default_factory=list, description="Skill names")
    is_enabled: bool = Field(False, description="Whether agent is enabled")
    trust_level: str = Field("unverified", description="Trust level")
    visibility: str = Field("public", description="Agent visibility")
    relevance_score: Optional[float] = Field(None, description="Relevance score from search")
    match_context: Optional[str] = Field(None, description="Context of the match")
