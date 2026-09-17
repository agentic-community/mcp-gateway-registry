"""
Skill security schema models for skill-scanner integration.

This module defines Pydantic models for skill security scan results, configurations,
and related data structures used throughout the skill security scanning workflow.
"""

from pydantic import BaseModel, Field


class SkillSecurityScanFinding(BaseModel):
    """Individual security finding from skill scanner."""

    file_path: str | None = Field(None, description="File where finding was detected")
    line_number: int | None = Field(None, description="Line number of finding")
    severity: str = Field(..., description="Severity level: CRITICAL, HIGH, MEDIUM, LOW")
    threat_names: list[str] = Field(
        default_factory=list, description="List of detected threat names"
    )
    threat_summary: str = Field(default="", description="Summary of threat found")
    analyzer: str = Field(
        ...,
        description="Analyzer that detected the finding: static, behavioral, llm, meta, virustotal, ai-defense",
    )
    is_safe: bool = Field(..., description="Whether the component is considered safe")


class SkillSecurityScanResult(BaseModel):
    """Complete security scan result for a skill."""

    skill_path: str = Field(..., description="Path of the scanned skill")
    skill_md_url: str | None = Field(None, description="URL to SKILL.md")
    scan_timestamp: str = Field(..., description="ISO timestamp of the scan")
    is_safe: bool = Field(..., description="Overall safety assessment")
    critical_issues: int = Field(default=0, description="Count of critical severity issues")
    high_severity: int = Field(default=0, description="Count of high severity issues")
    medium_severity: int = Field(default=0, description="Count of medium severity issues")
    low_severity: int = Field(default=0, description="Count of low severity issues")
    analyzers_used: list[str] = Field(
        default_factory=list, description="List of analyzers used in scan"
    )
    raw_output: dict = Field(default_factory=dict, description="Full scanner output")
    output_file: str | None = Field(None, description="Path to detailed JSON output file")
    scan_failed: bool = Field(default=False, description="Whether the scan failed to complete")
    error_message: str | None = Field(None, description="Error message if scan failed")


class SkillSecurityScanConfig(BaseModel):
    """Configuration for skill security scanning."""

    enabled: bool = Field(default=True, description="Enable/disable skill security scanning")
    scan_on_registration: bool = Field(default=True, description="Scan skills during registration")
    block_unsafe_skills: bool = Field(
        default=True, description="Disable skills that fail security scan"
    )
    analyzers: str = Field(default="static", description="Comma-separated list of analyzers to use")
    scan_timeout_seconds: int = Field(
        default=120, description="Timeout for security scans in seconds"
    )
    llm_api_key: str | None = Field(None, description="API key for LLM-based analysis")
    virustotal_api_key: str | None = Field(None, description="API key for VirusTotal integration")
    ai_defense_api_key: str | None = Field(None, description="API key for Cisco AI Defense")
    add_security_pending_tag: bool = Field(
        default=True, description="Add 'security-pending' tag to unsafe skills"
    )
    block_on_scan_failure: bool = Field(
        default=True,
        description=(
            "Also disable skills whose scan could not complete. On by default to "
            "preserve earlier behaviour; turn it off because a failed scan is no "
            "verdict at all, so blocking on it takes down a skill the scanner "
            "never assessed"
        ),
    )


class SkillSecurityStatus(BaseModel):
    """Security status summary for a skill."""

    skill_path: str = Field(..., description="Skill path (e.g., /pdf-processing)")
    skill_name: str = Field(..., description="Display name of the skill")
    is_safe: bool = Field(..., description="Whether the skill passed security scan")
    last_scan_timestamp: str | None = Field(None, description="ISO timestamp of last scan")
    critical_issues: int = Field(default=0, description="Count of critical issues")
    high_severity: int = Field(default=0, description="Count of high severity issues")
    scan_status: str = Field(default="pending", description="Status: pending, completed, failed")
    is_disabled_for_security: bool = Field(
        default=False, description="Whether skill is disabled due to security issues"
    )
