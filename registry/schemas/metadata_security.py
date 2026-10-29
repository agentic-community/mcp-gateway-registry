"""Security verdicts for registrant-supplied metadata (issue #1811).

A verdict is produced by the registry when metadata is written, and is stored
beside the record as ``metadata_scan``, never inside ``metadata``. Registrants
cannot set it: every repository write that carries ``metadata`` recomputes it,
and every write that does not drops any supplied value.

Two shapes live here:

- ``MetadataScanVerdict``: what is stored. It records which metadata keys a
  YARA rule matched, but never the matched text.
- ``MetadataSecurityView``: what a read sees. It is derived from the stored
  verdict and the record's current metadata, so a verdict that no longer
  matches the metadata reads as ``stale`` instead of ``clean``.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# A path into the metadata document, one segment per level. Servers and agents
# use one-segment paths (top-level keys); a skill's free-form keys live under
# ``extra``, so they are attributed as ["extra", "<key>"].
MetadataKeyPath = list[str]

ScanStatus = Literal["clean", "flagged", "failed"]

# "unscanned": no verdict stored (records written before this feature).
# "stale": a verdict exists but was computed for different metadata.
# "disabled": metadata scanning is turned off in configuration.
EffectiveScanStatus = Literal["clean", "flagged", "failed", "unscanned", "stale", "disabled"]


class MetadataScanFinding(BaseModel):
    """One YARA rule that matched, and the metadata keys it was attributed to."""

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(..., description="YARA rule name, e.g. 'prompt_injection'")
    threat_type: str = Field(..., description="The rule's threat_type meta value")
    severity: str = Field(..., description="Severity mcp-scanner assigns to this threat type")
    keys: list[MetadataKeyPath] = Field(
        default_factory=list,
        description="Metadata key paths the match was attributed to",
    )


class MetadataScanVerdict(BaseModel):
    """Stored result of scanning one record's metadata."""

    model_config = ConfigDict(extra="forbid")

    status: ScanStatus
    flagged_keys: list[MetadataKeyPath] = Field(default_factory=list)
    findings: list[MetadataScanFinding] = Field(default_factory=list)
    scanned_at: str = Field(..., description="ISO 8601 UTC time of the scan")
    metadata_digest: str = Field(
        ...,
        description="sha256 of the canonical metadata the verdict was computed for",
    )
    ruleset: str = Field(
        ...,
        description="Scanner distribution, version and rule-file digest used",
    )
    error: str | None = Field(
        default=None,
        description="Fixed error code when status is 'failed'; never scanner output",
    )


class MetadataSecurityView(BaseModel):
    """A record's metadata security state as seen at read time.

    Returned to operators on the REST read endpoints as ``metadata_security``.
    The mcpgw tools use ``status`` and ``flagged_keys`` to decide which keys,
    if any, may reach a model.
    """

    status: EffectiveScanStatus
    flagged_keys: list[MetadataKeyPath] = Field(default_factory=list)
    findings: list[MetadataScanFinding] = Field(default_factory=list)
    scanned_at: str | None = None
    ruleset: str | None = None
