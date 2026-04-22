"""GitHub authentication provider for private repository access.

Provides auth headers for httpx requests to GitHub, supporting:
- Personal Access Token (PAT) -- static, user-scoped
- GitHub App installation token -- ephemeral, org-scoped

Auth headers are only sent to explicitly allowed hosts.
"""

import asyncio
import logging
import time
from datetime import datetime
from urllib.parse import urlparse

import httpx
import jwt

from ..core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Default GitHub hosts that receive auth headers
_DEFAULT_GITHUB_HOSTS: frozenset[str] = frozenset(
    {
        "github.com",
        "raw.githubusercontent.com",
    }
)


class GitHubAuthProvider:
    """Provides auth headers for GitHub API requests.

    Supports two credential tiers with automatic fallback:
    1. GitHub App (installation token, ephemeral, org-scoped)
    2. Personal Access Token (static, user-scoped)

    Auth headers are only sent to explicitly allowed hosts.
    """

    def __init__(self) -> None:
        self._allowed_hosts = self._build_allowed_hosts()
        self._cached_token: str | None = None
        self._token_expires_at: float = 0.0
        self._failure_retry_after: float = 0.0
        self._token_lock = asyncio.Lock()
        self._log_active_tier()

    def _build_allowed_hosts(self) -> frozenset[str]:
        """Build allowed hosts from defaults + github_extra_hosts config."""
        extra_raw = settings.github_extra_hosts
        extra = frozenset(h.strip().lower() for h in extra_raw.split(",") if h.strip())
        return _DEFAULT_GITHUB_HOSTS | extra

    def _is_allowed_host(self, url: str) -> bool:
        """Check if URL hostname is in the allowed GitHub hosts set."""
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        allowed = hostname in self._allowed_hosts
        if not allowed:
            logger.debug("GitHub auth: host '%s' not in allowed hosts, skipping auth", hostname)
        return allowed

    def _log_active_tier(self) -> None:
        """Log which auth tier is active at initialization."""
        if self._has_app_credentials():
            logger.info("GitHub auth: GitHub App credentials configured")
        elif settings.github_pat:
            logger.info("GitHub auth: Personal Access Token configured")
        else:
            logger.info("GitHub auth: No credentials configured (unauthenticated access)")

    def _has_app_credentials(self) -> bool:
        """Check if all GitHub App credentials are present."""
        return bool(
            settings.github_app_id
            and settings.github_app_installation_id
            and settings.github_app_private_key
        )

    async def get_auth_headers(self, url: str) -> dict[str, str]:
        """Return auth headers if url matches an allowed GitHub host.

        Returns empty dict if:
        - URL host is not in the allowed hosts set
        - No credentials are configured
        - Token exchange fails (logged, falls back gracefully)
        """
        if not self._is_allowed_host(url):
            return {}

        # Tier 2: GitHub App
        if self._has_app_credentials():
            token = await self._get_github_app_token()
            if token:
                return {"Authorization": f"Bearer {token}"}
            logger.warning("GitHub App token exchange failed, falling back to PAT")

        # Tier 1: PAT
        if settings.github_pat:
            return {"Authorization": f"Bearer {settings.github_pat}"}

        # Tier 0: Unauthenticated
        return {}

    def _create_jwt(self) -> str:
        """Create signed JWT for GitHub App authentication.

        Uses RS256 algorithm per GitHub's requirements.
        Claims: iat (now - 60s for clock skew), exp (now + 600s), iss (app_id).
        """
        now = int(time.time())
        payload = {
            "iat": now - 60,
            "exp": now + 600,
            "iss": settings.github_app_id,
        }
        # Handle PEM key from env vars where newlines may be literal \n strings
        private_key = settings.github_app_private_key.replace("\\n", "\n")
        return jwt.encode(payload, private_key, algorithm="RS256")

    async def _get_github_app_token(self) -> str | None:
        """Get or refresh cached GitHub App installation token."""
        now = time.time()

        # Fast path: valid cache (no lock needed)
        if self._cached_token and now < self._token_expires_at - 300:
            return self._cached_token

        # Negative cache: avoid hammering GitHub API on sustained misconfiguration
        if not self._cached_token and now < self._failure_retry_after:
            return None

        async with self._token_lock:
            # Double-check after acquiring lock
            if self._cached_token and time.time() < self._token_expires_at - 300:
                return self._cached_token

            try:
                app_jwt = self._create_jwt()
                installation_id = settings.github_app_installation_id
                base_url = settings.github_api_base_url.rstrip("/")
                url = f"{base_url}/app/installations/{installation_id}/access_tokens"

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        headers={
                            "Authorization": f"Bearer {app_jwt}",
                            "Accept": "application/vnd.github+json",
                        },
                        timeout=10,
                    )

                if response.status_code != 201:
                    logger.error(
                        "GitHub App token exchange failed: HTTP %d - %s",
                        response.status_code,
                        response.text[:200],
                    )
                    self._failure_retry_after = time.time() + 60
                    return None

                data = response.json()
                token = data.get("token")
                if not token:
                    logger.error("GitHub App token response missing 'token' field")
                    self._failure_retry_after = time.time() + 60
                    return None

                self._cached_token = token

                # Parse expiry from response, fall back to 1 hour
                expires_at = data.get("expires_at")
                if expires_at:
                    try:
                        expiry_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                        self._token_expires_at = expiry_dt.timestamp()
                    except ValueError:
                        self._token_expires_at = time.time() + 3600
                else:
                    self._token_expires_at = time.time() + 3600

                logger.debug("GitHub App installation token refreshed successfully")
                return self._cached_token

            except (httpx.RequestError, KeyError, ValueError) as e:
                logger.error("GitHub App token exchange error: %s", e)
                self._failure_retry_after = time.time() + 60
                return None


# Module-level singleton -- shared across all consumers
github_auth_provider = GitHubAuthProvider()
