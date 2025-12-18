from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from ..models.egress_allowlist import (
    EgressAllowlistEntryRecord,
)
from ..models.egress_allowlist import (
    EgressAllowlistEntryKind,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _normalize_hostname(
    hostname: str,
) -> str:
    normalized = hostname.strip().lower().rstrip(".")
    if not normalized:
        raise ValueError("proxy_pass_url must include a hostname")
    if any(ch in normalized for ch in ("\r", "\n", " ", "\t")):
        raise ValueError("proxy_pass_url hostname must not contain whitespace")
    return normalized


def _is_ip_literal(
    hostname: str,
) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _matches_domain_suffix(
    *,
    hostname: str,
    suffix: str,
) -> bool:
    normalized_suffix = suffix.strip().lower().lstrip(".").rstrip(".")
    if not normalized_suffix:
        return False
    return hostname == normalized_suffix or hostname.endswith(f".{normalized_suffix}")


@dataclass(frozen=True)
class AllowlistDecision:
    allowed: bool
    reason: str
    matched_entry: Optional[EgressAllowlistEntryRecord] = None


def normalize_allowlist_entry_value(
    *,
    kind: EgressAllowlistEntryKind,
    value: str,
) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("value must be a non-empty string")
    if any(ch in stripped for ch in ("\r", "\n")):
        raise ValueError("value must not contain newline characters")

    if kind == "hostname":
        return _normalize_hostname(stripped)

    if kind == "domain-suffix":
        normalized = stripped.strip().lower().lstrip(".").rstrip(".")
        if not normalized:
            raise ValueError("domain-suffix value must be a non-empty string")
        if "*" in normalized:
            raise ValueError("domain-suffix value must not include '*'")
        if any(ch in normalized for ch in (" ", "\t")):
            raise ValueError("domain-suffix value must not contain whitespace")
        return normalized

    if kind == "ip-cidr":
        try:
            network = ipaddress.ip_network(stripped, strict=False)
        except ValueError as exc:
            raise ValueError("ip-cidr value must be a valid CIDR") from exc
        return str(network)

    raise ValueError(f"Unknown allowlist kind: {kind}")


def check_proxy_pass_url(
    *,
    proxy_pass_url: str,
    entries: list[EgressAllowlistEntryRecord],
    now: Optional[datetime] = None,
) -> AllowlistDecision:
    if not proxy_pass_url or not proxy_pass_url.strip():
        return AllowlistDecision(
            allowed=False,
            reason="proxy_pass_url is required",
        )

    parsed = urlparse(proxy_pass_url.strip())
    if parsed.scheme not in {"http", "https"}:
        return AllowlistDecision(
            allowed=False,
            reason="proxy_pass_url scheme must be http or https",
        )

    if parsed.username is not None or parsed.password is not None:
        return AllowlistDecision(
            allowed=False,
            reason="proxy_pass_url must not include userinfo",
        )

    if parsed.hostname is None:
        return AllowlistDecision(
            allowed=False,
            reason="proxy_pass_url must include a hostname",
        )

    ts = now or _utc_now()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    hostname = _normalize_hostname(parsed.hostname)

    if _is_ip_literal(hostname):
        ip = ipaddress.ip_address(hostname)
        for entry in entries:
            if entry.expires_at is not None and entry.expires_at <= ts:
                continue
            if entry.kind != "ip-cidr":
                continue
            try:
                network = ipaddress.ip_network(entry.value, strict=False)
            except ValueError:
                continue
            if ip in network:
                return AllowlistDecision(
                    allowed=True,
                    reason="Matched allowlist ip-cidr entry",
                    matched_entry=entry,
                )

        return AllowlistDecision(
            allowed=False,
            reason="No allowlist match for IP literal hostname",
        )

    for entry in entries:
        if entry.expires_at is not None and entry.expires_at <= ts:
            continue

        if entry.kind == "hostname":
            try:
                normalized_entry = _normalize_hostname(entry.value)
            except ValueError:
                continue

            if normalized_entry == hostname:
                return AllowlistDecision(
                    allowed=True,
                    reason="Matched allowlist hostname entry",
                    matched_entry=entry,
                )

        if entry.kind == "domain-suffix":
            if _matches_domain_suffix(hostname=hostname, suffix=entry.value):
                return AllowlistDecision(
                    allowed=True,
                    reason="Matched allowlist domain-suffix entry",
                    matched_entry=entry,
                )

    return AllowlistDecision(
        allowed=False,
        reason="No allowlist match for hostname",
    )
