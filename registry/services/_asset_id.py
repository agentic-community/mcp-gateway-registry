# registry/services/_asset_id.py
import logging
import re
from uuid import uuid4

logger = logging.getLogger(__name__)

MAX_ID_LENGTH: int = 512

# Safe-charset allowlist for a caller-supplied id (deny-by-default). Covers the
# id shapes #1276 targets -- UUID (hex + '-'), ARN
# (arn:aws:iam::123456789012:role/foo), URN (urn:example:agent:1), and peer
# registry ids -- while rejecting whitespace, quotes, angle brackets,
# backslash, and shell/regex metacharacters ($ { } | & ; ` * etc.). The id is
# only ever used as a parameterized value today, but we constrain the input at
# the source so a future sink (log/label/URL/path) can't be steered by a hostile
# id. Anchored full-match, so a single bad character rejects the whole value.
_SAFE_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9._:/@#=+-]+$")


class InvalidAssetIdError(ValueError):
    """Raised when a supplied id is empty, too long, or has disallowed characters."""


class CallerSuppliedIdDisabledError(InvalidAssetIdError):
    """Raised when a caller supplies an id while the feature flag is disabled.

    Subclasses InvalidAssetIdError so any route that already maps that to a 422
    handles this case with no extra branch. Fail-closed: caller-supplied ids are
    OFF by default and must be explicitly enabled per deployment.
    """


def check_caller_supplied_id_allowed(
    supplied_id: str | None,
    feature_enabled: bool,
) -> None:
    """Reject a caller-supplied id when the feature flag is disabled.

    Called by the PUBLIC registration routes only (server/agent/skill). The
    federation sync path builds cards through the service layer directly and is
    intentionally NOT gated here: peer ids are governed by the peer allowlist,
    which is the trust boundary for federation. Omitting the id (the default for
    every existing caller) is always allowed; only a non-None supplied id is
    gated.
    """
    if supplied_id is not None and not feature_enabled:
        raise CallerSuppliedIdDisabledError(
            "caller-supplied asset id is disabled on this registry; omit 'id' to "
            "auto-generate one, or set ALLOW_CALLER_SUPPLIED_ASSET_ID=true to enable it"
        )


def validate_asset_id(supplied_id: str) -> str:
    """Validate a supplied (non-None) asset id and return it stripped.

    Single source of truth for the id rules: non-empty after strip, at most
    MAX_ID_LENGTH characters, and every character in the safe allowlist
    (_SAFE_ASSET_ID_RE). Used both by resolve_asset_id (server routes, which
    have no Pydantic model) and by the agent/skill request-model field
    validators, so the three call sites can never drift. Raises
    InvalidAssetIdError (a ValueError subclass, so Pydantic treats it as a
    normal validation error).
    """
    stripped = supplied_id.strip()
    if not stripped:
        # Supplied but blank is a *caller error*, not a request to generate one.
        raise InvalidAssetIdError("id must be a non-empty string when provided")
    if len(stripped) > MAX_ID_LENGTH:
        raise InvalidAssetIdError(f"id must be <= {MAX_ID_LENGTH} characters")
    if not _SAFE_ASSET_ID_RE.match(stripped):
        raise InvalidAssetIdError(
            "id may contain only letters, digits, and the characters . _ - : / @ # = +"
        )
    return stripped


def resolve_asset_id(supplied_id: str | None) -> str:
    """Return the caller-supplied id if valid and non-empty, else a new uuid4 string.

    Generalizes the federation 'use peer id if present, else generate' idiom to
    every registration path. The server route has no Pydantic card model, so this
    function is where server-side id validation actually happens. Raises
    InvalidAssetIdError on a supplied-but-invalid id; the route maps that to 422.
    """
    if supplied_id is None:
        return str(uuid4())  # omitted entirely -> generate, no behavior change
    return validate_asset_id(supplied_id)
