from __future__ import annotations

from typing import Optional

from ..auth.dependency import (
    EnforceAIManagementContext,
)
from ..config import (
    EnforceAISettings,
)
from ..crypto.keyring import (
    GatewayKeyring,
    load_gateway_keyring_cached,
)
from ..db.data_layer import (
    EnforceAIStores,
)
from ..errors import (
    DependencyUnavailableError,
)
from ..management.service import (
    ManagementService,
)
from ..secrets.pepper import (
    load_api_key_pepper,
)


def _load_gateway_keyring(
    *,
    settings: EnforceAISettings,
) -> Optional[GatewayKeyring]:
    if (
        settings.gateway_private_key_path is None
        or settings.gateway_public_keys_dir is None
        or settings.gateway_active_kid is None
    ):
        return None

    try:
        return load_gateway_keyring_cached(
            private_key_path=settings.gateway_private_key_path,
            public_keys_dir=settings.gateway_public_keys_dir,
            active_kid=settings.gateway_active_kid,
        )
    except Exception as exc:  # noqa: BLE001 - map to 503
        raise DependencyUnavailableError(
            "Gateway keyring unavailable",
            public_message="Enforcement misconfigured",
        ) from exc


def _build_management_service(
    *,
    settings: EnforceAISettings,
    stores: EnforceAIStores,
    context: EnforceAIManagementContext,
) -> ManagementService:
    pepper: Optional[bytes] = None
    if settings.api_key_pepper_path is not None:
        try:
            pepper = load_api_key_pepper(settings.api_key_pepper_path)
        except ValueError as exc:
            raise DependencyUnavailableError(
                "API key pepper unavailable",
                public_message="Enforcement misconfigured",
            ) from exc

    keyring = _load_gateway_keyring(settings=settings)

    return ManagementService(
        agent_store=stores.agent_store,
        api_key_store=stores.api_key_store,
        revocation_store=stores.revocation_store,
        scope_catalog=context.catalog,
        api_key_pepper=pepper,
        gateway_keyring=keyring,
        gateway_issuer=settings.gateway_issuer,
    )
