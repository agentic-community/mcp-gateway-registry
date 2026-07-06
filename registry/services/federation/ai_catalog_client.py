"""ARD ai-catalog.json ingestion client (issue #1296, Phase 3).

Crawls an external ``ai-catalog.json`` document and any nested
``application/ai-catalog+json`` children, returning validated
:class:`AICatalogManifest` objects for the ingestion service to map and index.

Deliberately NOT a :class:`BaseFederationClient` subclass: that base is
server-centric (abstract ``fetch_server``/``fetch_all_servers`` and a single
``endpoint``), whereas this client crawls arbitrary catalog URLs with its own
size cap and SSRF guard. It uses a fresh, **auth-less** ``httpx.Client`` so a
peer/federation token is never leaked to a third-party catalog host.

Validation is done by parsing into the ``AICatalogManifest`` Pydantic model
(which mirrors the ARD ai-catalog schema's required fields) — no extra runtime
dependency. Any document that is non-https, resolves to a blocked IP, is too
large, is not JSON, or fails model validation is skipped and logged, never
fatal.
"""

from __future__ import annotations

import json
import logging
import time
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import ValidationError

from ...exceptions import UrlValidationError
from ...schemas.ard_models import AICatalogManifest
from ...utils.url_guard import SKILL_PROFILE, guarded_client
from ..ard_net_guard import assert_fetchable
from ..ard_search_service import ArdValidationError

logger = logging.getLogger(__name__)

MEDIA_TYPE_CATALOG = "application/ai-catalog+json"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per document


class AiCatalogFederationClient:
    """Fetch + validate + recursively crawl ai-catalog.json documents."""

    def __init__(
        self,
        timeout_seconds: int = 15,
        max_depth: int = 3,
        polite_interval_ms: int = 200,
        same_domain_only: bool = True,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_depth = max_depth
        self.polite_interval_ms = polite_interval_ms
        self.same_domain_only = same_domain_only
        # Auth-less, SSRF-pinned client: NEVER send Authorization to a
        # third-party catalog host, and connect only to an IP that the guard
        # validated inside this same request. assert_fetchable() enforces the
        # ARD-specific https-only + same-domain policy up front; the guarded
        # client then pins the resolved public IP at connect time so the fetch
        # cannot re-resolve a rebound hostname to a private/metadata address
        # between validation and connect (SKILL_PROFILE = public-only, no proxy
        # allowlist — the strictest profile, correct for third-party catalogs).
        self.client = guarded_client(
            profile=SKILL_PROFILE,
            timeout=timeout_seconds,
            follow_redirects=False,
        )

    def __del__(self):
        if hasattr(self, "client"):
            self.client.close()

    def fetch_catalog(
        self,
        root_uri: str,
    ) -> list[tuple[AICatalogManifest, str]]:
        """Fetch the root catalog and all nested catalogs up to ``max_depth``.

        Returns a list of ``(manifest, source_uri)`` pairs. Synchronous (httpx);
        the ingestion service runs it inside ``asyncio.to_thread``.
        """
        root_domain = (urlparse(root_uri).hostname or "").lower()
        out: list[tuple[AICatalogManifest, str]] = []
        visited: set[str] = set()
        self._crawl(root_uri, 0, visited, out, root_domain)
        return out

    def _crawl(
        self,
        url: str,
        depth: int,
        visited: set[str],
        out: list[tuple[AICatalogManifest, str]],
        root_domain: str,
    ) -> None:
        """Depth-first crawl with loop/cost guards and per-fetch SSRF checks."""
        if depth > self.max_depth:
            return
        if url in visited:
            logger.debug("ARD ingestion: skipping already-visited catalog URL %s", url)
            return
        visited.add(url)

        # Fail closed per-URL, never per-crawl: a single poisoned entry (e.g. a
        # URL that httpx rejects with InvalidURL, or any unexpected error) must
        # skip that node and let the crawl continue with its siblings, so a
        # hostile catalog cannot DoS the whole ingestion run with one bad link.
        try:
            manifest = self._fetch_one(url, root_domain)
        except Exception as e:
            logger.warning("ARD ingestion: skipping catalog URL %s after error: %s", url, e)
            return
        if manifest is None:
            return
        out.append((manifest, url))

        # Recurse into nested application/ai-catalog+json entries.
        for entry in manifest.entries:
            if entry.type != MEDIA_TYPE_CATALOG or not entry.url:
                continue
            child = urljoin(url, entry.url)
            self._crawl(child, depth + 1, visited, out, root_domain)

    def _fetch_one(
        self,
        url: str,
        root_domain: str,
    ) -> AICatalogManifest | None:
        """Fetch and validate a single catalog document, or return ``None``."""
        allowed = root_domain if self.same_domain_only else None
        try:
            assert_fetchable(url, allowed)
        except ArdValidationError as e:
            logger.warning("ARD ingestion: refusing catalog URL %s: %s", url, e)
            return None

        if self.polite_interval_ms:
            time.sleep(self.polite_interval_ms / 1000.0)

        # Stream and abort early so a hostile host cannot exhaust memory by
        # sending a huge body within the timeout window — the size cap is
        # enforced as bytes arrive, not after the whole body is buffered.
        content = b""
        try:
            with self.client.stream("GET", url, headers={"Accept": "application/json"}) as response:
                response.raise_for_status()
                # follow_redirects=False means a 3xx is returned as-is and does
                # NOT raise_for_status() (which only raises on 4xx/5xx); refuse
                # it explicitly so a hostile host cannot smuggle a fake catalog
                # body inside a redirect response (the guard pins the original
                # host, not the unfollowed Location).
                if 300 <= response.status_code < 400:
                    logger.warning(
                        "ARD ingestion: catalog %s returned redirect status %d, skipping",
                        url,
                        response.status_code,
                    )
                    return None
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > _MAX_BYTES:
                    logger.warning(
                        "ARD ingestion: catalog %s Content-Length %s exceeds %d cap, skipping",
                        url,
                        declared,
                        _MAX_BYTES,
                    )
                    return None
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > _MAX_BYTES:
                        logger.warning(
                            "ARD ingestion: catalog %s exceeds %d byte cap, aborting",
                            url,
                            _MAX_BYTES,
                        )
                        return None
                    chunks.append(chunk)
                content = b"".join(chunks)
        except UrlValidationError as e:
            # The pinned guarded transport re-resolves and re-validates the host
            # at connect time (and on every redirect hop). A hostname that
            # passed assert_fetchable() but rebinds to a private/metadata IP
            # before the connect is blocked here — skip and log, never fatal.
            logger.warning("ARD ingestion: refusing rebound/unsafe catalog URL %s: %s", url, e)
            return None
        except httpx.HTTPError as e:
            logger.warning("ARD ingestion: fetch failed for %s: %s", url, e)
            return None

        try:
            payload = json.loads(content)
        except (ValueError, UnicodeDecodeError) as e:
            logger.warning("ARD ingestion: catalog %s is not valid JSON: %s", url, e)
            return None

        try:
            return AICatalogManifest.model_validate(payload)
        except ValidationError as e:
            logger.warning(
                "ARD ingestion: catalog %s failed schema validation: %s",
                url,
                e.errors()[:3],
            )
            return None
