"""ANS (Agent Name Service) Pydantic models for the registry."""

import logging
from datetime import datetime

from pydantic import (
    BaseModel,
    Field,
)


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class ANSCertificateInfo(BaseModel):
    """Certificate information from ANS."""

    serial_number: str | None = Field(
        default=None,
        description="Certificate serial number",
    )
    not_before: str | None = Field(
        default=None,
        description="Certificate validity start (ISO 8601)",
    )
    not_after: str | None = Field(
        default=None,
        description="Certificate validity end (ISO 8601)",
    )
    subject_dn: str | None = Field(
        default=None,
        description="Certificate subject distinguished name",
    )
    issuer_dn: str | None = Field(
        default=None,
        description="Certificate issuer distinguished name",
    )


class ANSFunctionInfo(BaseModel):
    """Function (skill) information from an ANS endpoint."""

    id: str = Field(description="Function identifier")
    name: str = Field(description="Function display name")
    tags: list[str] | None = Field(default=None, description="Function tags")


class ANSEndpointInfo(BaseModel):
    """Endpoint information from ANS."""

    type: str = Field(
        description="Endpoint type (e.g., http)",
    )
    url: str = Field(
        description="Endpoint URL",
    )
    protocol: str | None = Field(
        default=None,
        description="Protocol (A2A, MCP, HTTP-API)",
    )
    transports: list[str] = Field(
        default_factory=list,
        description="Transport types (e.g., STREAMABLE-HTTP, JSON-RPC)",
    )
    functions: list[ANSFunctionInfo] = Field(
        default_factory=list,
        description="Functions available on this endpoint",
    )


class ANSMetadata(BaseModel):
    """ANS verification metadata stored on agents and servers."""

    ans_agent_id: str = Field(
        description="ANS Agent ID (e.g., ans://v1.0.0.myagent.example.com)",
    )
    linked_at: datetime = Field(
        description="When the ANS ID was linked",
    )
    last_verified: datetime = Field(
        description="When ANS status was last verified",
    )
    status: str = Field(
        default="pending",
        description="Verification status: verified, expired, revoked, not_found, pending",
    )
    domain: str | None = Field(
        default=None,
        description="Verified domain from ANS",
    )
    organization: str | None = Field(
        default=None,
        description="Organization name from ANS",
    )
    ans_name: str | None = Field(
        default=None,
        description="Full ANS name (e.g., ans://v1.0.0.myagent.example.com)",
    )
    ans_display_name: str | None = Field(
        default=None,
        description="Display name as registered in ANS",
    )
    ans_description: str | None = Field(
        default=None,
        description="Description as registered in ANS",
    )
    ans_version: str | None = Field(
        default=None,
        description="Agent version registered in ANS",
    )
    registered_with_ans_at: str | None = Field(
        default=None,
        description="When the agent was registered with ANS (ISO 8601)",
    )
    certificate: ANSCertificateInfo | None = Field(
        default=None,
        description="Certificate details from ANS",
    )
    endpoints: list[ANSEndpointInfo] = Field(
        default_factory=list,
        description="Endpoints registered in ANS",
    )
    links: list[dict[str, str]] = Field(
        default_factory=list,
        description="HATEOAS links from ANS API (self, server-certificates, identity-certificates)",
    )
    raw_ans_response: dict | None = Field(
        default=None,
        description="Full raw JSON response from the ANS API",
    )


class LinkANSRequest(BaseModel):
    """Request to link an ANS Agent ID."""

    ans_agent_id: str = Field(
        description="ANS Agent ID to link",
        min_length=5,
        pattern=r"^ans://v[\d.]+\.[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?$",
    )


class LinkANSResponse(BaseModel):
    """Response after linking an ANS Agent ID."""

    success: bool = Field(
        description="Whether linking succeeded",
    )
    message: str = Field(
        description="Status message",
    )
    ans_metadata: ANSMetadata | None = Field(
        default=None,
        description="ANS metadata if successful",
    )


class ANSSyncStats(BaseModel):
    """Statistics from an ANS sync operation."""

    total: int = Field(
        default=0,
        description="Total assets with ANS links checked",
    )
    updated: int = Field(
        default=0,
        description="Assets whose status was updated",
    )
    errors: int = Field(
        default=0,
        description="Assets that failed verification",
    )
    duration_seconds: float = Field(
        default=0.0,
        description="Total sync duration in seconds",
    )


class ANSIntegrationMetrics(BaseModel):
    """ANS integration metrics for admin dashboard."""

    total_linked: int = Field(
        default=0,
        description="Total assets with ANS links",
    )
    by_status: dict[str, int] = Field(
        default_factory=dict,
        description="Count of assets by ANS status",
    )
    by_asset_type: dict[str, int] = Field(
        default_factory=dict,
        description="Count of linked assets by type (agent, server)",
    )
    last_sync_at: datetime | None = Field(
        default=None,
        description="When the last sync completed",
    )
    last_sync_stats: ANSSyncStats | None = Field(
        default=None,
        description="Stats from the last sync",
    )
