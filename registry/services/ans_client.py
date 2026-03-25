# registry/services/ans_client.py

import asyncio
import logging
from datetime import (
    datetime,
    timedelta,
    timezone,
)

import httpx

from registry.core.config import settings
from registry.schemas.ans_models import (
    ANSCertificateInfo,
    ANSEndpointInfo,
    ANSFunctionInfo,
    ANSMetadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

ANS_STATUS_VERIFIED: str = "verified"
ANS_STATUS_EXPIRED: str = "expired"
ANS_STATUS_REVOKED: str = "revoked"
ANS_STATUS_NOT_FOUND: str = "not_found"
ANS_STATUS_PENDING: str = "pending"
ANS_STATUS_ERROR: str = "error"

MAX_RETRIES: int = 3
RETRY_BASE_DELAY_SECONDS: float = 1.0
CIRCUIT_BREAKER_THRESHOLD: int = 5
CIRCUIT_BREAKER_RESET_SECONDS: int = 3600

# Circuit breaker state (module-level)
_consecutive_failures: int = 0
_circuit_open_until: datetime | None = None


def _build_auth_header() -> dict[str, str]:
    """Build the ANS API authentication header."""
    return {
        "Authorization": f"sso-key {settings.ans_api_key}:{settings.ans_api_secret}",
        "Accept": "application/json",
    }


def _check_circuit_breaker() -> bool:
    """Check if circuit breaker is open (ANS API assumed down).

    Returns:
        True if requests should proceed, False if circuit is open
    """
    global _circuit_open_until
    if _circuit_open_until is None:
        return True
    if datetime.now(timezone.utc) > _circuit_open_until:
        _circuit_open_until = None
        logger.info("ANS circuit breaker reset -- resuming API calls")
        return True
    return False


def _record_failure() -> None:
    """Record an ANS API failure for circuit breaker."""
    global _consecutive_failures, _circuit_open_until
    _consecutive_failures += 1
    if _consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
        _circuit_open_until = datetime.now(timezone.utc) + timedelta(
            seconds=CIRCUIT_BREAKER_RESET_SECONDS
        )
        logger.warning(
            f"ANS circuit breaker OPEN after {_consecutive_failures} failures. "
            f"Pausing API calls for {CIRCUIT_BREAKER_RESET_SECONDS} seconds."
        )


def _record_success() -> None:
    """Record an ANS API success -- reset circuit breaker."""
    global _consecutive_failures, _circuit_open_until
    _consecutive_failures = 0
    _circuit_open_until = None


def _determine_status(
    ans_data: dict,
) -> str:
    """Determine verification status from ANS response data.

    Args:
        ans_data: Raw ANS API response

    Returns:
        Status string: verified, expired, revoked
    """
    if ans_data.get("revoked", False):
        return ANS_STATUS_REVOKED

    agent_status = ans_data.get("agentStatus", "")
    if agent_status == "REVOKED":
        return ANS_STATUS_REVOKED

    cert = ans_data.get("certificate", {})
    not_after = cert.get("not_after") or cert.get("notAfter")
    if not_after:
        try:
            expiry = datetime.fromisoformat(not_after.replace("Z", "+00:00"))
            if expiry < datetime.now(expiry.tzinfo):
                return ANS_STATUS_EXPIRED
        except (ValueError, TypeError):
            logger.warning(f"Could not parse certificate expiry: {not_after}")

    return ANS_STATUS_VERIFIED


def _extract_metadata(
    ans_agent_id: str,
    ans_data: dict,
) -> ANSMetadata:
    """Extract ANS metadata from API response.

    Args:
        ans_agent_id: The ANS Agent ID that was queried
        ans_data: Raw ANS API response

    Returns:
        Structured ANS metadata
    """
    now = datetime.now(timezone.utc)
    status = _determine_status(ans_data)

    cert_data = ans_data.get("certificate", {})
    certificate = ANSCertificateInfo(
        serial_number=cert_data.get("serial_number") or cert_data.get("serialNumber"),
        not_before=cert_data.get("not_before") or cert_data.get("notBefore"),
        not_after=cert_data.get("not_after") or cert_data.get("notAfter"),
        subject_dn=cert_data.get("subject_dn") or cert_data.get("subjectDn"),
        issuer_dn=cert_data.get("issuer_dn") or cert_data.get("issuerDn"),
    )

    endpoints = []
    for ep in ans_data.get("endpoints", []):
        functions = []
        for fn in ep.get("functions", []):
            if fn and fn.get("id"):
                functions.append(
                    ANSFunctionInfo(
                        id=fn.get("id", ""),
                        name=fn.get("name", ""),
                        tags=fn.get("tags"),
                    )
                )
        endpoints.append(
            ANSEndpointInfo(
                type=ep.get("type", "http"),
                url=ep.get("agentUrl") or ep.get("url", ""),
                protocol=ep.get("protocol"),
                transports=ep.get("transports", []),
                functions=functions,
            )
        )

    links = ans_data.get("links", [])

    return ANSMetadata(
        ans_agent_id=ans_agent_id,
        linked_at=now,
        last_verified=now,
        status=status,
        domain=ans_data.get("agentHost") or ans_data.get("domain"),
        organization=ans_data.get("organization"),
        ans_name=ans_data.get("ansName") or ans_data.get("name"),
        ans_display_name=ans_data.get("agentDisplayName"),
        ans_description=ans_data.get("agentDescription"),
        ans_version=ans_data.get("version"),
        registered_with_ans_at=ans_data.get("registrationTimestamp"),
        certificate=certificate,
        endpoints=endpoints,
        links=links,
        raw_ans_response=ans_data,
    )


async def _resolve_ans_id(
    ans_agent_id: str,
) -> str | None:
    """Resolve an ANS agent identifier to a UUID.

    If the input is already a UUID, return it as-is.
    If the input is an ans:// URI, search the ANS API to find the UUID.

    Args:
        ans_agent_id: ANS Agent ID (UUID or ans:// URI)

    Returns:
        UUID string if found, None if ans:// URI could not be resolved
    """
    if not ans_agent_id.startswith("ans://"):
        return ans_agent_id

    headers = _build_auth_header()
    search_url = f"{settings.ans_api_endpoint}/v1/agents"

    try:
        async with httpx.AsyncClient(timeout=settings.ans_api_timeout_seconds) as client:
            response = await client.get(
                search_url,
                headers=headers,
                params={"limit": 100, "offset": 0},
            )
            response.raise_for_status()
            data = response.json()

            for agent in data.get("agents", []):
                if agent.get("ansName") == ans_agent_id:
                    agent_uuid = agent.get("agentId")
                    logger.info(
                        f"Resolved ANS name '{ans_agent_id}' to UUID '{agent_uuid}'"
                    )
                    return agent_uuid

            # Search remaining pages if needed
            total_count = data.get("totalCount", 0)
            offset = 100
            while offset < total_count:
                response = await client.get(
                    search_url,
                    headers=headers,
                    params={"limit": 100, "offset": offset},
                )
                response.raise_for_status()
                page_data = response.json()

                for agent in page_data.get("agents", []):
                    if agent.get("ansName") == ans_agent_id:
                        agent_uuid = agent.get("agentId")
                        logger.info(
                            f"Resolved ANS name '{ans_agent_id}' to UUID '{agent_uuid}'"
                        )
                        return agent_uuid

                offset += 100

    except Exception as e:
        logger.error(f"Failed to resolve ANS name '{ans_agent_id}': {e}")

    logger.warning(f"Could not resolve ANS name to UUID: {ans_agent_id}")
    return None


async def verify_ans_agent(
    ans_agent_id: str,
) -> ANSMetadata | None:
    """Verify an ANS Agent ID by calling the GoDaddy ANS API.

    Includes retry with exponential backoff and circuit breaker.

    Args:
        ans_agent_id: ANS Agent ID (e.g., ans://v1.0.0.myagent.example.com)

    Returns:
        ANSMetadata if found, None if not found

    Raises:
        httpx.HTTPStatusError: For non-404 HTTP errors after retries
        httpx.TimeoutException: If ANS API times out after retries
        RuntimeError: If circuit breaker is open
    """
    if not _check_circuit_breaker():
        raise RuntimeError("ANS API circuit breaker is open -- API assumed unavailable")

    # If ans_agent_id is an ans:// URI, resolve it to a UUID first
    resolved_id = await _resolve_ans_id(ans_agent_id)
    if resolved_id is None:
        logger.info(f"ANS name not found in registry: {ans_agent_id}")
        return None

    headers = _build_auth_header()
    url = f"{settings.ans_api_endpoint}/v1/agents/{resolved_id}"

    logger.info(f"Verifying ANS Agent ID: {resolved_id} (input: {ans_agent_id})")

    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=settings.ans_api_timeout_seconds) as client:
                response = await client.get(url, headers=headers)

                if response.status_code == 404:
                    logger.info(f"ANS Agent ID not found: {ans_agent_id}")
                    _record_success()
                    return None

                response.raise_for_status()
                ans_data = response.json()

            logger.info(f"ANS verification successful for: {ans_agent_id}")
            _record_success()
            return _extract_metadata(ans_agent_id, ans_data)

        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
                logger.warning(
                    f"ANS API attempt {attempt + 1}/{MAX_RETRIES} failed for "
                    f"{ans_agent_id}: {e}. Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"ANS API failed after {MAX_RETRIES} attempts for {ans_agent_id}: {e}"
                )

    _record_failure()
    raise last_exception
