"""Registration gate service for admission control.

Calls a configurable external endpoint to approve or deny
registration and update requests before they are persisted.

Security: Credential fields are always stripped from payloads.
Sensitive headers (authorization, cookie, csrf) are excluded.
"""

import asyncio
import logging
import time

import httpx

from registry.core.config import settings
from registry.schemas.registration_gate_models import (
    RegistrationGateAuthType,
    RegistrationGateRequest,
    RegistrationGateResponse,
    RegistrationGateResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

ALLOWED_STATUS_CODE: int = 200
DENIED_STATUS_CODE: int = 403
INITIAL_BACKOFF_SECONDS: float = 0.5
GATE_ERROR_MAX_LENGTH: int = 500

SENSITIVE_FIELD_SUBSTRINGS: list[str] = [
    "credential",
    "secret",
    "token",
    "password",
    "api_key",
]

SENSITIVE_FIELD_NAMES: set[str] = {
    "auth_credential",
    "auth_credential_encrypted",
    "auth_header_name",
}

SENSITIVE_HEADERS: set[str] = {
    "cookie",
    "authorization",
    "x-csrf-token",
}


def _sanitize_payload(
    payload: dict,
) -> dict:
    """Remove credential and sensitive fields from a registration payload.

    Args:
        payload: Raw registration payload dict.

    Returns:
        A new dict with sensitive fields removed.
    """
    sanitized = {}
    for key, value in payload.items():
        if key in SENSITIVE_FIELD_NAMES:
            continue
        key_lower = key.lower()
        if any(sub in key_lower for sub in SENSITIVE_FIELD_SUBSTRINGS):
            continue
        sanitized[key] = value
    return sanitized


def _build_auth_headers() -> dict[str, str]:
    """Build authentication headers based on gate auth configuration.

    Returns:
        Dictionary of auth headers to include in the gate request.
    """
    auth_type = settings.registration_gate_auth_type.lower()
    credential = settings.registration_gate_auth_credential

    if auth_type == RegistrationGateAuthType.BEARER and credential:
        return {"Authorization": f"Bearer {credential}"}

    if auth_type == RegistrationGateAuthType.API_KEY and credential:
        header_name = settings.registration_gate_auth_header_name
        return {header_name: credential}

    return {}


def _extract_request_headers(
    raw_headers: list[tuple[bytes, bytes]],
) -> dict[str, str]:
    """Extract request headers as a string dict, filtering sensitive headers.

    Args:
        raw_headers: Raw ASGI header tuples from the request scope.

    Returns:
        Dictionary of header name to header value strings.
    """
    result = {}
    for name_bytes, value_bytes in raw_headers:
        name = name_bytes.decode("latin-1").lower()
        if name not in SENSITIVE_HEADERS:
            result[name] = value_bytes.decode("latin-1")
    return result


def _is_gate_configured() -> bool:
    """Check if the registration gate is enabled and properly configured.

    Returns:
        True if the gate should be invoked, False otherwise.
    """
    if not settings.registration_gate_enabled:
        return False

    if not settings.registration_gate_url:
        logger.warning(
            "Registration gate is enabled but no URL is configured. "
            "Treating as disabled."
        )
        return False

    return True


def _truncate_error(
    message: str,
) -> str:
    """Truncate gate error message to safe length.

    Args:
        message: Raw error message from gate.

    Returns:
        Truncated message (max GATE_ERROR_MAX_LENGTH chars).
    """
    if len(message) > GATE_ERROR_MAX_LENGTH:
        return message[:GATE_ERROR_MAX_LENGTH] + "..."
    return message


async def _call_gate_endpoint(
    gate_request: RegistrationGateRequest,
) -> RegistrationGateResult:
    """Call the gate endpoint with retry logic.

    Args:
        gate_request: The payload to send to the gate endpoint.

    Returns:
        RegistrationGateResult with the gate decision.
    """
    url = settings.registration_gate_url
    timeout = settings.registration_gate_timeout_seconds
    max_retries = settings.registration_gate_max_retries

    headers = {"Content-Type": "application/json"}
    headers.update(_build_auth_headers())

    payload_json = gate_request.model_dump_json()
    total_attempts = 1 + max_retries
    last_error = ""

    for attempt in range(1, total_attempts + 1):
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    content=payload_json,
                    headers=headers,
                )

            elapsed = time.time() - start_time
            logger.info(
                f"Registration gate response: status={response.status_code}, "
                f"attempt={attempt}/{total_attempts}, elapsed={elapsed:.2f}s"
            )

            if response.status_code == ALLOWED_STATUS_CODE:
                return RegistrationGateResult(
                    allowed=True,
                    error_message=None,
                    gate_status_code=response.status_code,
                    attempts=attempt,
                )

            if response.status_code == DENIED_STATUS_CODE:
                error_message = "Registration denied by policy"
                try:
                    gate_response = RegistrationGateResponse(
                        **response.json()
                    )
                    if gate_response.error:
                        error_message = _truncate_error(gate_response.error)
                except Exception:
                    raw_text = response.text[:GATE_ERROR_MAX_LENGTH]
                    error_message = raw_text or error_message

                return RegistrationGateResult(
                    allowed=False,
                    error_message=error_message,
                    gate_status_code=response.status_code,
                    attempts=attempt,
                )

            last_error = (
                f"Unexpected status code {response.status_code} "
                f"from gate endpoint"
            )
            logger.warning(
                f"Registration gate returned unexpected status "
                f"{response.status_code} on attempt {attempt}/{total_attempts}"
            )

        except httpx.TimeoutException:
            elapsed = time.time() - start_time
            last_error = f"Gate endpoint timed out after {elapsed:.2f}s"
            logger.warning(
                f"Registration gate timeout on attempt "
                f"{attempt}/{total_attempts}: {last_error}"
            )

        except httpx.RequestError as e:
            last_error = f"Connection error: {e}"
            logger.warning(
                f"Registration gate connection error on attempt "
                f"{attempt}/{total_attempts}: {last_error}"
            )

        if attempt < total_attempts:
            backoff = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.info(
                f"Retrying gate call in {backoff:.1f}s "
                f"(attempt {attempt + 1}/{total_attempts})"
            )
            await asyncio.sleep(backoff)

    logger.error(
        f"Registration gate exhausted all {total_attempts} attempts. "
        f"Last error: {last_error}. Blocking registration (fail-closed)."
    )
    return RegistrationGateResult(
        allowed=False,
        error_message=(
            "Registration gate is unavailable. "
            "Registration blocked (fail-closed policy)."
        ),
        gate_status_code=None,
        attempts=total_attempts,
    )


async def check_registration_gate(
    asset_type: str,
    operation: str,
    source_api: str,
    registration_payload: dict,
    raw_headers: list[tuple[bytes, bytes]],
) -> RegistrationGateResult:
    """Check the registration gate for a registration or update request.

    This is the main public function called by registration and update
    endpoints. Returns immediately with allowed=True if the gate is
    not configured.

    Args:
        asset_type: Type of asset ("agent", "server", or "skill").
        operation: "register" or "update".
        source_api: API path that triggered the request.
        registration_payload: Full request as a dict.
        raw_headers: Raw ASGI headers from the HTTP request scope.

    Returns:
        RegistrationGateResult indicating whether to proceed or block.
    """
    if not _is_gate_configured():
        return RegistrationGateResult(
            allowed=True,
            error_message=None,
            gate_status_code=None,
            attempts=0,
        )

    sanitized_payload = _sanitize_payload(registration_payload)
    request_headers = _extract_request_headers(raw_headers)

    gate_request = RegistrationGateRequest(
        asset_type=asset_type,
        operation=operation,
        source_api=source_api,
        registration_payload=sanitized_payload,
        request_headers=request_headers,
    )

    logger.info(
        f"Calling registration gate for {operation} of {asset_type} "
        f"from {source_api}"
    )

    return await _call_gate_endpoint(gate_request)


async def verify_gate_connectivity() -> None:
    """Verify connectivity to the gate endpoint at startup.

    Called during application startup when gate is enabled.
    Logs warnings if the gate is unreachable or uses HTTP.
    Does NOT block startup.
    """
    if not _is_gate_configured():
        return

    url = settings.registration_gate_url
    auth_type = settings.registration_gate_auth_type

    logger.info(
        f"Registration gate enabled: url={url}, auth_type={auth_type}"
    )

    if url.startswith("http://"):
        logger.warning(
            "Registration gate URL uses HTTP. "
            "HTTPS is strongly recommended for production."
        )

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.head(url)
        logger.info(
            f"Registration gate connectivity check: "
            f"status={response.status_code} (reachable)"
        )
    except Exception as e:
        logger.warning(
            f"Registration gate connectivity check failed: {e}. "
            f"The gate endpoint may be unreachable. "
            f"Registrations will be blocked until the gate is available."
        )
