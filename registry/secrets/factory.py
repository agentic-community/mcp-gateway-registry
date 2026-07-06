"""SecretStore factory.

Selects the concrete SecretStore implementation from
``settings.secret_store_backend`` (secrets-manager | openbao), mirroring the
``registry.repositories.factory`` singleton + lazy import pattern.
Connection/credential validation for the backends happens here so a
misconfigured deployment fails fast.
"""

import logging
import os

from ..core.config import settings
from .interfaces import SecretStoreBase

logger = logging.getLogger(__name__)

_secret_store: SecretStoreBase | None = None


def _build_secrets_manager() -> SecretStoreBase:
    import boto3

    from .secrets_manager.store import SecretsManagerStore

    # Prefer the explicit AWS_SECRETS_REGION, but fall back to the standard AWS
    # region env vars (AWS_REGION / AWS_DEFAULT_REGION) that are always present on
    # ECS/EKS and in the AWS SDK's own resolution. This means an operator does not
    # have to set a dedicated AWS_SECRETS_REGION just to use the secrets-manager
    # backend when the task already runs in a known region.
    region = (
        settings.aws_secrets_region
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or None
    )
    if not region:
        raise ValueError(
            "SECRET_STORE_BACKEND=secrets-manager requires a region: set "
            "AWS_SECRETS_REGION (or AWS_REGION / AWS_DEFAULT_REGION)."
        )
    client = boto3.client("secretsmanager", region_name=region)
    return SecretsManagerStore(
        client=client,
        prefix=settings.secrets_manager_path_prefix,
        kms_key_id=settings.secrets_manager_kms_key_id or None,
    )


def _build_openbao() -> SecretStoreBase:
    import hvac

    from .openbao.store import OpenBaoStore

    if not settings.openbao_addr:
        raise ValueError("SECRET_STORE_BACKEND=openbao requires OPENBAO_ADDR.")

    client_kwargs = {"url": settings.openbao_addr}
    if settings.openbao_namespace:
        client_kwargs["namespace"] = settings.openbao_namespace
    client = hvac.Client(**client_kwargs)

    method = settings.openbao_auth_method

    # Build a (re-)login closure for the configured auth method. The store calls
    # it once at construction and again whenever a request fails because the
    # client's Vault token expired -- OpenBao role tokens are short-lived (often
    # 1h) and hvac does NOT auto-renew, so a long-lived registry process would
    # otherwise start failing every read with "permission denied" once its first
    # login token lapsed. ``token`` auth is static (no server-side login) so its
    # closure is a no-op: nothing to refresh.
    def _login() -> None:
        if method == "token":
            import os

            token = os.environ.get("OPENBAO_TOKEN") or os.environ.get("VAULT_TOKEN")
            if not token:
                raise ValueError(
                    "OPENBAO_AUTH_METHOD=token requires OPENBAO_TOKEN (or VAULT_TOKEN)."
                )
            client.token = token
        elif method == "kubernetes":
            role = settings.openbao_role
            if not role:
                raise ValueError("OPENBAO_AUTH_METHOD=kubernetes requires OPENBAO_ROLE.")
            # Re-read the projected SA token each login: it is rotated in place by
            # the kubelet, so a cached copy could itself be stale on re-auth.
            with open("/var/run/secrets/kubernetes.io/serviceaccount/token") as fh:
                jwt = fh.read()
            client.auth.kubernetes.login(role=role, jwt=jwt)
        elif method == "approle":
            import os

            role_id = os.environ.get("OPENBAO_ROLE_ID")
            secret_id = os.environ.get("OPENBAO_SECRET_ID")
            if not role_id or not secret_id:
                raise ValueError(
                    "OPENBAO_AUTH_METHOD=approle requires OPENBAO_ROLE_ID and OPENBAO_SECRET_ID."
                )
            client.auth.approle.login(role_id=role_id, secret_id=secret_id)
        else:
            raise ValueError(
                f"Invalid OPENBAO_AUTH_METHOD={method!r}. Accepted: token, kubernetes, approle."
            )

    _login()  # initial authentication

    return OpenBaoStore(
        client=client,
        mount_point=settings.openbao_kv_mount,
        prefix=settings.secrets_manager_path_prefix,
        reauthenticate=_login,
    )


def get_secret_store() -> SecretStoreBase:
    """Get the SecretStore singleton for the configured backend."""
    global _secret_store
    if _secret_store is not None:
        return _secret_store

    backend = settings.secret_store_backend
    logger.info("Creating SecretStore with backend: %s", backend)

    if backend == "secrets-manager":
        _secret_store = _build_secrets_manager()
    elif backend == "openbao":
        _secret_store = _build_openbao()
    else:  # validator guarantees this is unreachable
        raise ValueError(f"Unsupported SECRET_STORE_BACKEND={backend!r}.")

    return _secret_store


def reset_secret_store() -> None:
    """Reset the SecretStore singleton. USE ONLY IN TESTS."""
    global _secret_store
    _secret_store = None
