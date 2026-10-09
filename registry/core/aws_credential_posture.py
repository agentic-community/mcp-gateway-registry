"""Report at startup when the registry holds AWS credentials it never uses.

A Docker Compose deployment on an EC2 host reaches the instance metadata service
from every container on the bridge network, so boto3 falls through its provider
chain to ``iam-role`` and picks up the instance role's temporary credentials. That
happens whether or not the deployment calls a single AWS API.

Every AWS client the registry builds sits behind a feature flag: Cognito as the
auth provider, Secrets Manager as the secret store, and AgentCore federation. Run
Keycloak with the default secret store and federation off, and nothing here calls
AWS. The instance profile then adds reachable credentials for no benefit, which is
pure blast radius if the container is ever compromised.

This module names that situation in the log. It deliberately does NOT read the
attached IAM policy to grade it: enumerating a role's permissions needs
``iam:ListAttachedRolePolicies``, ``iam:GetPolicy``, ``iam:GetPolicyVersion``,
``iam:ListRolePolicies`` and ``iam:GetRolePolicy`` granted to the role under audit,
so the check would widen the very role it is auditing, and those reads are what an
attacker holding a stolen credential otherwise has to guess at.

See issue #1861 and the IAM section of docs/installation.md.
"""

import asyncio
import logging

from registry.core.config import settings

logger = logging.getLogger(__name__)

# botocore's provider name for the EC2 instance metadata service. Everything else
# in the chain (env vars, a shared credentials file, the ECS/EKS container
# endpoint) delivers credentials out of band, where this warning does not apply.
IMDS_PROVIDER_NAME: str = "iam-role"

# Where an operator can read what to do about the warning.
IAM_DOCS_REFERENCE: str = "docs/installation.md (IAM for Docker Compose on EC2)"

# Setting values that route the registry at an AWS service.
COGNITO_AUTH_PROVIDER: str = "cognito"
SECRETS_MANAGER_BACKEND: str = "secrets-manager"  # nosec B105 - backend name, not a password

# boto3 defaults to a 60s connect timeout, a 60s read timeout and legacy retries,
# so an unreachable STS endpoint (no egress, a misconfigured proxy) would stall
# FastAPI lifespan for minutes. This is a log line, so it gets seconds.
STS_TIMEOUT_SECONDS: int = 2
STS_MAX_ATTEMPTS: int = 1

# Hard ceiling on the whole check, covering provider resolution as well as the STS
# call. Startup must not wait on a diagnostic. Issue #1129 is the precedent.
POSTURE_CHECK_DEADLINE_SECONDS: float = 10.0


def _resolved_credential_provider() -> str | None:
    """Return the name of the botocore provider that supplied credentials.

    Resolving the chain reads the environment, any shared credentials file and, as
    a last resort, the instance metadata service. It makes no authenticated call.

    Returns:
        The provider name, for example ``iam-role`` or ``env``. ``None`` when no
        credentials resolve at all, or when boto3 is not installed.
    """
    try:
        import botocore.session
    except ImportError:
        logger.debug("botocore is not installed, skipping the AWS credential check")
        return None

    try:
        credentials = botocore.session.get_session().get_credentials()
    except Exception:
        # A broken or partial AWS config must not stop the registry from starting.
        logger.debug("Could not resolve AWS credentials", exc_info=True)
        return None

    if credentials is None:
        return None

    return getattr(credentials, "method", None)


def _caller_identity_arn() -> str | None:
    """Return the ARN the resolved credentials belong to.

    ``sts:GetCallerIdentity`` needs no IAM permission, so this adds no requirement
    to the role. It is only used to name the identity in the log line.

    Returns:
        The assumed-role ARN, or ``None`` if the call fails for any reason.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        return None

    # Short timeouts and a single attempt. The caller is on the startup path.
    config = Config(
        connect_timeout=STS_TIMEOUT_SECONDS,
        read_timeout=STS_TIMEOUT_SECONDS,
        retries={"max_attempts": STS_MAX_ATTEMPTS},
    )

    try:
        return boto3.client("sts", config=config).get_caller_identity().get("Arn")
    except Exception:
        # No network, no endpoint, a proxy in the way: none of it is fatal here.
        logger.debug("sts:GetCallerIdentity failed", exc_info=True)
        return None


def _aws_backed_features(
    aws_federation_enabled: bool,
) -> list[str]:
    """List the enabled features that make AWS API calls.

    Args:
        aws_federation_enabled: Whether AgentCore registry federation is on, read
            from the federation config rather than from a setting.

    Returns:
        Human-readable names of the enabled features. Empty when the registry
        calls no AWS API.
    """
    features: list[str] = []

    if settings.auth_provider == COGNITO_AUTH_PROVIDER:
        features.append("Cognito as the auth provider (AUTH_PROVIDER=cognito)")

    if settings.secret_store_backend == SECRETS_MANAGER_BACKEND:
        features.append(
            "Secrets Manager as the secret store (SECRET_STORE_BACKEND=secrets-manager)"
        )

    if aws_federation_enabled:
        features.append("AgentCore registry federation")

    return features


async def _report_credential_posture(
    aws_federation_enabled: bool,
) -> None:
    """Work out the credential position and log one line about it.

    Args:
        aws_federation_enabled: Whether AgentCore registry federation is on.
    """
    provider = await asyncio.to_thread(_resolved_credential_provider)
    if provider is None:
        logger.info("No AWS credentials resolve in this container. Nothing to report.")
        return

    features = _aws_backed_features(aws_federation_enabled)
    arn = await asyncio.to_thread(_caller_identity_arn)
    identity = arn or "an identity that sts:GetCallerIdentity did not return"

    if provider != IMDS_PROVIDER_NAME:
        logger.info(
            f"AWS credentials come from the '{provider}' provider as {identity}. "
            f"AWS-backed features enabled: {', '.join(features) or 'none'}."
        )
        return

    if features:
        logger.info(
            f"AWS credentials come from the EC2 instance metadata service as "
            f"{identity}. AWS-backed features enabled: {', '.join(features)}. "
            f"Scope the instance role to these features only, see "
            f"{IAM_DOCS_REFERENCE}."
        )
        return

    logger.warning(
        f"This registry calls no AWS API, yet it can read credentials for "
        f"{identity} from the EC2 instance metadata service. Every container "
        f"on the Docker bridge network can read them too, so the instance "
        f"profile adds risk with no benefit here. Detach the instance profile, "
        f"or scope it to the features you turn on. See {IAM_DOCS_REFERENCE}."
    )


async def log_aws_credential_posture(
    aws_federation_enabled: bool,
) -> None:
    """Log which AWS identity the registry runs as, and warn if it is unused.

    Never raises, and never waits longer than ``POSTURE_CHECK_DEADLINE_SECONDS``.
    A diagnostic must not be able to stop or slow the service, so every failure
    becomes a debug line and startup continues.

    Args:
        aws_federation_enabled: Whether AgentCore registry federation is on.
    """
    try:
        await asyncio.wait_for(
            _report_credential_posture(aws_federation_enabled),
            timeout=POSTURE_CHECK_DEADLINE_SECONDS,
        )
    except TimeoutError:
        # asyncio cannot cancel a thread started by to_thread, so a stuck AWS call
        # runs on to its own timeout. Startup does not wait for it.
        logger.debug(
            f"The AWS credential posture check passed "
            f"{POSTURE_CHECK_DEADLINE_SECONDS}s and was abandoned"
        )
    except Exception:
        logger.debug("The AWS credential posture check did not complete", exc_info=True)
