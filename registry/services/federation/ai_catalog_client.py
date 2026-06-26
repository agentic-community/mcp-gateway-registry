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

from ...schemas.ard_models import AICatalogManifest
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
        # Auth-less client: NEVER send Authorization to a third-party catalog host.
        self.client = httpx.Client(timeout=timeout_seconds, follow_redirects=False)

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

        manifest = self._fetch_one(url, root_domain)
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
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > _MAX_BYTES:
                    logger.warning(
                        "ARD ingestion: catalog %s Content-Length %s exceeds %d cap, skipping",
                        url, declared, _MAX_BYTES,
                    )
                    return None
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > _MAX_BYTES:
                        logger.warning(
                            "ARD ingestion: catalog %s exceeds %d byte cap, aborting",
                            url, _MAX_BYTES,
                        )
                        return None
                    chunks.append(chunk)
                content = b"".join(chunks)
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
