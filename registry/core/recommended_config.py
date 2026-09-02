"""Registry of recommended-but-optional configuration and its evaluation.

Some settings are optional (the app starts and runs without them) yet strongly
recommended for a secure/robust deployment -- e.g. the egress credential vault
runs without ``EGRESS_CREDENTIAL_ENCRYPTION_KEY`` but then stores third-party
credentials in plaintext at rest. Leaving such a setting unset is easy to do
silently, so we nudge operators through three surfaces that all read from this
one source of truth:

- a startup ``WARNING`` log (see ``log_recommended_config_warnings``),
- an OpenTelemetry ``ObservableGauge`` (see ``registry.observability.meters``),
- the System Config UI badge (via ``GET /api/config/full`` ``recommendations``).

Add a new nudge by appending one :class:`RecommendedConfig` descriptor here; all
three surfaces pick it up automatically. Each descriptor is gated by ``applies``
so a deployment that does not use the feature is never nagged about it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommendedConfig:
    """One recommended-but-optional configuration setting.

    Args:
        id: Stable identifier, also used as a metric label. kebab/snake, no PII.
        setting: The ``Settings`` attribute name this recommendation concerns
            (lets the UI cross-reference the config field).
        component: Coarse area label for grouping/alerting (e.g. ``"egress"``).
        severity: Why it matters (e.g. ``"security"``, ``"reliability"``).
        message: Operator-facing nudge. Must not contain secrets.
        applies: Returns True when this recommendation is relevant to the current
            deployment (e.g. the feature it protects is enabled). Non-applicable
            recommendations are omitted entirely, so unused features are not nagged.
        is_configured: Returns True when the setting is satisfied.
    """

    id: str
    setting: str
    component: str
    severity: str
    message: str
    applies: Callable[[Any], bool]
    is_configured: Callable[[Any], bool]


def _egress_encryption_key_set(settings: Any) -> bool:
    """True when the egress credential encryption root key is set (non-blank)."""
    value = getattr(settings, "egress_credential_encryption_key", "") or ""
    return bool(value.strip())


# The single source of truth. Keep label cardinality bounded: this is a fixed,
# code-defined list, never derived from user input.
_RECOMMENDED: tuple[RecommendedConfig, ...] = (
    RecommendedConfig(
        id="egress_credential_encryption",
        setting="egress_credential_encryption_key",
        component="egress",
        severity="security",
        message=(
            "Egress credential vault is enabled but EGRESS_CREDENTIAL_ENCRYPTION_KEY "
            "is unset: per-user third-party credentials are stored in PLAINTEXT at "
            "rest. Set the key to encrypt them (generate: "
            'python3 -c "import secrets; print(secrets.token_urlsafe(32))").'
        ),
        applies=lambda s: bool(getattr(s, "egress_auth_enabled", False)),
        is_configured=_egress_encryption_key_set,
    ),
)


def evaluate_recommendations(
    settings: Any,
) -> list[dict[str, Any]]:
    """Evaluate every recommendation that APPLIES to the current deployment.

    Non-applicable recommendations are skipped so unused features are not nagged.
    Fails safe: if ``applies``/``is_configured`` raises, the recommendation is
    reported as NOT configured (a nudge) rather than silently dropped.

    Args:
        settings: The application ``Settings`` object.

    Returns:
        One dict per applicable recommendation with keys ``id``, ``setting``,
        ``component``, ``severity``, ``message``, and ``configured`` (bool).
        Never raises.
    """
    results: list[dict[str, Any]] = []
    for rec in _RECOMMENDED:
        try:
            if not rec.applies(settings):
                continue
        except Exception as exc:  # defensive: a broken gate must not break callers
            logger.debug("recommended-config applies() failed for %s: %s", rec.id, exc)
            continue
        try:
            configured = bool(rec.is_configured(settings))
        except Exception as exc:
            logger.debug("recommended-config is_configured() failed for %s: %s", rec.id, exc)
            configured = False  # fail closed -> treat as a nudge
        results.append(
            {
                "id": rec.id,
                "setting": rec.setting,
                "component": rec.component,
                "severity": rec.severity,
                "message": rec.message,
                "configured": configured,
            }
        )
    return results


def log_recommended_config_warnings(
    settings: Any,
) -> None:
    """Emit a startup ``WARNING`` for each applicable recommendation left unset.

    Called once at startup. Configured recommendations and non-applicable ones
    produce no output. Never raises.
    """
    for rec in evaluate_recommendations(settings):
        if not rec["configured"]:
            logger.warning(
                "Recommended configuration not set [%s/%s severity=%s]: %s",
                rec["component"],
                rec["id"],
                rec["severity"],
                rec["message"],
            )
