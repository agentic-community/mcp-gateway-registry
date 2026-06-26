"""Registration webhook notification service.

Fires an async POST to a configurable URL when a server, agent, or skill
is registered (added), updated, deleted (removed), or finishes a security
scan. The call is fire-and-forget: failures are logged at WARNING but never
propagated to the caller.

When ``registration_webhook_signing_secret`` is configured, each payload is
signed with HMAC-SHA256 over the exact bytes transmitted and the signature is
sent in the ``X-Registry-Signature`` header (Issue #1330).
"""

import hashlib
import hmac
import json
import logging
from datetime import (
    UTC,
    datetime,
)

import httpx

from registry.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Signature header name for HMAC-signed webhook payloads (Issue #1330).
SIGNATURE_HEADER: str = "X-Registry-Signature"


def _build_auth_headers() -> dict[str, str]:
    """Build authentication headers for the webhook request.

    If auth header is 'Authorization', auto-prepends 'Bearer ' to the token.
    For any other header name, the token is sent as-is.

    Returns:
        Dict with a single auth header entry, or empty dict if no token.
    """
    if not settings.registration_webhook_auth_token:
        return {}

    header_name = settings.registration_webhook_auth_header
    token = settings.registration_webhook_auth_token

    if header_name.lower() == "authorization":
        token = f"Bearer {token}"

    return {header_name: token}


def _sign_body(
    body_bytes: bytes,
) -> str | None:
    """Compute the HMAC-SHA256 signature for a webhook body.

    Args:
        body_bytes: The exact bytes that will be transmitted as the request body.

    Returns:
        A string of the form 'sha256=<hex>' when a signing secret is configured,
        otherwise None (no signature header is sent).
    """
    secret = settings.registration_webhook_signing_secret
    if not secret:
        return None

    digest = hmac.new(
        secret.encode("utf-8"),
        body_bytes,
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


def _record_send_metric(
    event_type: str,
    outcome: str,
) -> None:
    """Record a webhook delivery outcome on the webhook_send_total counter.

    Never raises: metric failures must not affect the fire-and-forget webhook.

    Args:
        event_type: The webhook event_type value.
        outcome: One of 'success', 'timeout', 'error', 'skipped_no_url'.
    """
    try:
        from registry.observability.meters import webhook_send_total

        webhook_send_total.labels(event_type=event_type, outcome=outcome).inc()
    except Exception as e:
        logger.debug(f"webhook_send_total metric inc failed: {type(e).__name__}: {e}")


async def send_registration_webhook(
    event_type: str,
    registration_type: str,
    card_data: dict,
    performed_by: str | None = None,
    extra_fields: dict | None = None,
) -> None:
    """Send a webhook notification for a registration lifecycle event.

    This is fire-and-forget: failures are logged but never raised.

    The card payload is run through sanitize_payload (the same sanitizer the
    registration gate uses) before transmission. This strips top-level
    credential fields and masks local_runtime.env values + args, which are
    considered sensitive when sent to an external endpoint.

    When a signing secret is configured, the exact transmitted body is signed
    with HMAC-SHA256 and the signature is added in the X-Registry-Signature
    header. The body is serialized once so the signature matches the bytes sent.

    Args:
        event_type: One of "registration" (add), "update" (partial/full edit),
            "deletion" (remove), or "scan_complete" (security scan finished).
            Consumers filtering by event type must allow unknown types or skip
            them without failing.
        registration_type: One of "server", "agent", or "skill".
        card_data: The full card JSON as a dictionary.
        performed_by: Username of the operator who performed the action.
        extra_fields: Optional event-specific fields merged into the top-level
            envelope (e.g. {"scan": {...}}). These are NOT passed through the
            card sanitizer, so callers must only pass non-sensitive data.
    """
    # Local import to avoid a circular dep at module load time.
    from .registration_gate_service import sanitize_payload

    webhook_url = settings.registration_webhook_url
    if not webhook_url:
        _record_send_metric(event_type, "skipped_no_url")
        return

    if not webhook_url.startswith(("http://", "https://")):
        logger.error(f"Invalid webhook URL scheme: {webhook_url}")
        _record_send_metric(event_type, "error")
        return

    if webhook_url.startswith("http://"):
        logger.warning(
            "Registration webhook URL uses HTTP (not HTTPS). "
            "Credential data may be transmitted insecurely."
        )

    payload = {
        "event_type": event_type,
        "registration_type": registration_type,
        "timestamp": datetime.now(UTC).isoformat(),
        "performed_by": performed_by,
        "card": sanitize_payload(card_data) if isinstance(card_data, dict) else card_data,
    }
    if extra_fields:
        payload.update(extra_fields)

    # Serialize once so the HMAC signature matches the exact bytes transmitted.
    body_bytes = json.dumps(payload).encode("utf-8")

    headers = _build_auth_headers()
    headers["Content-Type"] = "application/json"
    signature = _sign_body(body_bytes)
    if signature:
        headers[SIGNATURE_HEADER] = signature
        logger.debug(f"Webhook signature attached: event={event_type}, sig={signature[:19]}...")

    timeout = settings.registration_webhook_timeout_seconds

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                webhook_url,
                content=body_bytes,
                headers=headers,
            )
            logger.info(
                f"Registration webhook sent: event={event_type}, "
                f"type={registration_type}, "
                f"status={response.status_code}, url={webhook_url}"
            )
            _record_send_metric(event_type, "success")
    except httpx.TimeoutException:
        logger.warning(
            f"Registration webhook timed out after {timeout}s: "
            f"event={event_type}, type={registration_type}, url={webhook_url}"
        )
        _record_send_metric(event_type, "timeout")
    except Exception as e:
        logger.warning(
            f"Registration webhook failed: event={event_type}, "
            f"type={registration_type}, url={webhook_url}, error={e}"
        )
        _record_send_metric(event_type, "error")
