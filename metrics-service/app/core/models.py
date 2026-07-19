from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import StrEnum


class MetricType(StrEnum):
    AUTH_REQUEST = "auth_request"
    TOOL_DISCOVERY = "tool_discovery"
    TOOL_EXECUTION = "tool_execution"
    REGISTRY_OPERATION = "registry_operation"
    HEALTH_CHECK = "health_check"
    PROTOCOL_LATENCY = "protocol_latency"
    CUSTOM = "custom"


class Metric(BaseModel):
    type: MetricType
    timestamp: datetime | None = Field(default_factory=datetime.utcnow)
    value: float
    duration_ms: float | None = None
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricRequest(BaseModel):
    service: str = Field(..., max_length=50)
    version: str | None = Field(None, max_length=20)
    instance_id: str | None = Field(None, max_length=50)
    metrics: List[Metric]


class MetricResponse(BaseModel):
    status: str
    accepted: int
    rejected: int
    errors: List[str] = []
    request_id: str


class ErrorResponse(BaseModel):
    status: str
    error: str
    message: str
    request_id: str
