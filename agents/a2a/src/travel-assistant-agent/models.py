"""Data models for Travel Assistant Agent."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DiscoveredAgent(BaseModel):
    """Agent discovered from registry."""

    name: str = Field(..., description="Agent name")
    description: str = Field(default="", description="Agent description")
    path: str = Field(..., description="Registry path")
    url: str = Field(..., description="Agent endpoint URL")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    skills: List[str] = Field(default_factory=list, description="Skill names")
    num_skills: int = Field(0, ge=0, description="Number of skills")
    num_stars: int = Field(0, ge=0, description="Community rating")
    is_enabled: bool = Field(False, description="Whether agent is enabled")
    provider: Optional[str] = Field(None, description="Agent provider/author")
    streaming: bool = Field(False, description="Supports streaming responses")
    trust_level: str = Field("unverified", description="Trust level")
    score: Optional[float] = Field(None, description="Relevance score from search")
