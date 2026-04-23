"""Registration webhook notification service.

Fires an async POST to a configurable URL when a server, agent, or skill
is registered (added) or deleted (removed). The call is fire-and-forget:
failures are logged at WARNING but never propagated to the caller.
"""

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


async def send_registration_webhook(
    event_type: str,
    registration_type: str,
    card_data: dict,
    performed_by: str | None = None,
) -> None:
    """Send a webhook notification for a successful registration or deletion.

    This is fire-and-forget: failures are logged but never raised.

    Args:
        event_type: One of "registration" (add) or "deletion" (remove).
        registration_type: One of "server", "agent", or "skill".
        card_data: The full card JSON as a dictionary.
        performed_by: Username of the operator who performed the action.
    """
    webhook_url = settings.registration_webhook_url
    if not webhook_url:
        return

    if not webhook_url.startswith(("http://", "https://")):
        logger.error(f"Invalid webhook URL scheme: {webhook_url}")
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
        "card": card_data,
    }

    headers = _build_auth_headers()
    headers["Content-Type"] = "application/json"
    timeout = settings.registration_webhook_timeout_seconds

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers=headers,
            )
            logger.info(
                f"Registration webhook sent: event={event_type}, "
                f"type={registration_type}, "
                f"status={response.status_code}, url={webhook_url}"
            )
    except httpx.TimeoutException:
        logger.warning(
            f"Registration webhook timed out after {timeout}s: "
            f"event={event_type}, type={registration_type}, url={webhook_url}"
        )
    except Exception as e:
        logger.warning(
            f"Registration webhook failed: event={event_type}, "
            f"type={registration_type}, url={webhook_url}, error={e}"
        )
