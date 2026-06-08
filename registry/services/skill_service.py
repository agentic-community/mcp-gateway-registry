"""
Service layer for skill management.

Simplified design:
- No in-memory state duplication
- Database as source of truth
- SKILL.md URL validation on registration
"""

import hashlib
import ipaddress
import logging
import re
import socket
from datetime import UTC, datetime
from functools import lru_cache
from typing import (
    Any,
)
from urllib.parse import urlparse

import httpx

from ..core.config import settings
from ..exceptions import (
    SkillUrlValidationError,
)
from ..repositories.factory import (
    get_search_repository,
    get_skill_repository,
)
from ..repositories.interfaces import (
    SearchRepositoryBase,
    SkillRepositoryBase,
)
from ..schemas.skill_models import (
    ContentIntegrity,
    FileHash,
    SkillCard,
    SkillInfo,
    SkillMetadata,
    SkillRegistrationRequest,
    VisibilityEnum,
)
from ..utils.path_utils import normalize_skill_path
from ..utils.url_utils import (
    extract_repository_url,
    translate_skill_url,
)
from .github_auth import github_auth_provider as _github_auth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Constants
URL_VALIDATION_TIMEOUT: int = 10

# Upper bound on GitLab tree-API pages fetched during resource discovery, so a
# server that keeps echoing the x-next-page header cannot loop indefinitely.
MAX_GITLAB_TREE_PAGES: int = 100

# Built-in trusted domains that skip IP validation (SSRF protection allowlist).
# Enterprise GitHub hosts are merged in at runtime from settings.github_extra_hosts
# via _trusted_domains() so GHES instances on private IPs are reachable for
# SKILL.md fetches (matches the host allowlist used by the GitHub auth provider).
_DEFAULT_TRUSTED_DOMAINS: frozenset = frozenset(
    {
        "github.com",
        "gitlab.com",
        "raw.githubusercontent.com",
        "bitbucket.org",
    }
)


@lru_cache(maxsize=1)
def _trusted_domains() -> frozenset[str]:
    """Return SSRF allowlist: built-in defaults plus configured GHES hosts.

    Reads settings.github_extra_hosts (the same setting that authorises auth
    header injection) so a single config knob covers both trust decisions.
    Cached because settings are immutable per-process.
    """
    extra_raw = settings.github_extra_hosts or ""
    extra = frozenset(h.strip().lower() for h in extra_raw.split(",") if h.strip())
    return _DEFAULT_TRUSTED_DOMAINS | extra


def _is_private_ip(
    ip_str: str,
) -> bool:
    """Check if an IP address is private, loopback, or link-local.

    Args:
        ip_str: IP address string to check

    Returns:
        True if the IP is private/loopback/link-local, False otherwise
    """
    try:
        ip = ipaddress.ip_address(ip_str)

        # Check for private, loopback, link-local, or reserved addresses
        if ip.is_private:
            return True
        if ip.is_loopback:
            return True
        if ip.is_link_local:
            return True
        if ip.is_reserved:
            return True

        # Check for cloud metadata endpoint (169.254.169.254)
        if ip_str == "169.254.169.254":
            return True

        return False
    except ValueError:
        # Invalid IP address format
        return True


def _is_safe_url(
    url: str,
) -> bool:
    """Check if a URL is safe to fetch (SSRF protection).

    This function validates that a URL:
    1. Uses http or https scheme
    2. Does not resolve to a private/loopback/link-local IP address
    3. Does not target cloud metadata endpoints

    Trusted domains (github.com, gitlab.com, etc., plus any host configured
    via settings.github_extra_hosts) skip the IP check so GHES instances on
    private networks remain reachable.

    Args:
        url: URL to validate

    Returns:
        True if the URL is safe to fetch, False otherwise
    """
    try:
        parsed = urlparse(url)

        # Check scheme - only allow http and https
        if parsed.scheme not in ("http", "https"):
            logger.warning(f"SSRF protection: Blocked URL with scheme '{parsed.scheme}'")
            return False

        hostname = parsed.hostname
        if not hostname:
            logger.warning("SSRF protection: URL has no hostname")
            return False

        # Check if hostname is in trusted domains allowlist
        hostname_lower = hostname.lower()
        if hostname_lower in _trusted_domains():
            logger.debug(f"SSRF protection: Trusted domain '{hostname_lower}'")
            return True

        # Resolve hostname to IP addresses
        try:
            addr_info = socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                proto=socket.IPPROTO_TCP,
            )
        except socket.gaierror as e:
            logger.warning(f"SSRF protection: Failed to resolve hostname '{hostname}': {e}")
            return False

        # Check all resolved IP addresses
        for family, socktype, proto, canonname, sockaddr in addr_info:
            ip_address = sockaddr[0]
            if _is_private_ip(ip_address):
                logger.warning(
                    f"SSRF protection: Blocked URL resolving to private IP "
                    f"'{ip_address}' for hostname '{hostname}'"
                )
                return False

        return True

    except Exception as e:
        logger.warning(f"SSRF protection: Error validating URL: {e}")
        return False


def _append_page_param(
    url: str,
    page: str,
) -> str:
    """Append or replace the page query parameter on a URL."""
    if re.search(r"(?<![a-z_])page=\d+", url):
        return re.sub(r"(?<![a-z_])page=\d+", f"page={page}", url)
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}page={page}"


def _build_fetch_headers(
    url: str,
    auth_scheme: str = "none",
    auth_credential: str | None = None,
    auth_header_name: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Build fetch URL and auth headers for a SKILL.md request.

    For GitLab URLs with credentials, translates /-/raw/ web URLs to
    API v4 endpoints that accept PRIVATE-TOKEN headers.

    Returns (fetch_url, headers) tuple.
    """
    headers: dict[str, str] = {}
    fetch_url = url

    if auth_scheme in ("none", "global_credentials") or not auth_credential:
        return fetch_url, headers

    if auth_scheme == "bearer":
        header_name = auth_header_name or "Authorization"
        headers[header_name] = f"Bearer {auth_credential}"
    elif auth_scheme == "api_key":
        header_name = auth_header_name or "PRIVATE-TOKEN"
        headers[header_name] = auth_credential

    if headers:
        try:
            from ..utils.gitlab_url_utils import (
                parse_gitlab_url,
                translate_gitlab_to_api_url,
            )

            if parse_gitlab_url(url):
                api_url = translate_gitlab_to_api_url(url)
                if api_url:
                    fetch_url = api_url
        except ImportError:
            pass

    return fetch_url, headers


_RESOURCE_TYPE_MAP: dict[str, str] = {
    "references": "reference",
    "scripts": "script",
    "agents": "agent",
    "assets": "asset",
}

_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".sh": "shell",
    ".bash": "shell",
    ".js": "javascript",
    ".ts": "typescript",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".json": "json",
}


# SKILL.md URL path patterns. Public GitHub blob and raw URLs differ;
# enterprise GitHub instances mirror the public path layout.
_BLOB_PATH_PATTERN = re.compile(r"^/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$")
_RAW_PATH_PATTERN = re.compile(r"^/([^/]+)/([^/]+)/refs/heads/([^/]+)/(.+)$")


def _api_base_for_skill_host(
    hostname: str,
) -> str | None:
    """Map a SKILL.md hostname (raw or blob, public or GHES) to its REST API base.

    Public GitHub maps to https://api.github.com.  GHES maps to the
    configured ``settings.github_api_base_url`` only when the configured
    host matches the SKILL.md host (modulo the optional ``raw.`` prefix).

    Returns None when the hostname is not a recognised GitHub instance.
    """
    hostname_lower = hostname.lower()
    if hostname_lower in ("github.com", "raw.githubusercontent.com"):
        return "https://api.github.com"

    if "github" not in hostname_lower:
        return None

    skill_host = hostname_lower[4:] if hostname_lower.startswith("raw.") else hostname_lower
    configured = settings.github_api_base_url.rstrip("/")
    configured_host = (urlparse(configured).hostname or "").lower()
    if configured_host and configured_host == skill_host:
        return configured

    return None


def _resolve_tree_api(
    skill_md_url: str,
) -> tuple[str, str, str, str] | None:
    """Resolve the GitHub/GHES Trees API endpoint for a SKILL.md URL.

    Parses both blob-style URLs (``github.com/{owner}/{repo}/blob/{ref}/...``)
    and raw-content URLs (``raw.githubusercontent.com/{owner}/{repo}/refs/heads/{ref}/...``),
    plus the corresponding GHES variants.

    Returns:
        (tree_api_url, encoded_project, ref, skill_dir) when the URL is a
        recognised GitHub/GHES SKILL.md URL, else None.

        - ``tree_api_url``: full Trees API URL with ``recursive=1``.
        - ``encoded_project``: ``{owner}/{repo}`` (kept for backward compat
          with the documented contract; GitHub's API uses owner+repo
          separately so callers can ignore this).
        - ``ref``: branch name resolved from the SKILL.md URL.
        - ``skill_dir``: directory portion of the path leading up to
          ``SKILL.md`` (``""`` when SKILL.md is at the repo root).
    """
    # Try GitLab translation first (Meraki: defensive import so removing
    # gitlab_url_utils leaves upstream features fully functional for GitHub).
    try:
        from ..utils.gitlab_url_utils import translate_gitlab_tree_api_url

        gitlab_result = translate_gitlab_tree_api_url(skill_md_url)
        if gitlab_result is not None:
            return gitlab_result
    except ImportError:
        pass

    # Fall through to upstream GitHub / GHES handling.
    parsed = urlparse(skill_md_url)
    hostname = parsed.hostname or ""
    if not hostname:
        return None

    match = _BLOB_PATH_PATTERN.match(parsed.path) or _RAW_PATH_PATTERN.match(parsed.path)
    if not match:
        return None

    owner, repo, ref, file_path = match.groups()
    skill_dir = file_path.rsplit("/", 1)[0] if "/" in file_path else ""

    api_base = _api_base_for_skill_host(hostname)
    if not api_base:
        return None

    tree_url = f"{api_base}/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    return tree_url, f"{owner}/{repo}", ref, skill_dir


# Files that live in the skill folder but should never be classified as
# resources (the SKILL.md itself, plus standard repo metadata).
_SKILL_DIR_EXCLUDED_FILENAMES: frozenset[str] = frozenset(
    {
        "SKILL.MD",
        "README.MD",
        "LICENSE",
        "LICENSE.TXT",
        "LICENSE.MD",
    }
)


def _classify_resource(
    path: str,
    skill_dir: str,
) -> str | None:
    """Classify a file's resource type given its repo-relative path.

    Two-tier classification:
      1. Subfolder convention (Anthropic style): if the immediate parent
         directory is one of references/scripts/agents/assets, use that.
      2. Flat-skill fallback: when the file lives directly under
         ``skill_dir`` (no recognised subfolder), classify by extension:
         ``.py``/``.sh``/``.bash``/``.js``/``.ts`` -> script,
         ``.md``/``.txt``/``.rst``                 -> reference,
         everything else                           -> asset.

    Returns None when the file should be excluded from the manifest
    (e.g. nested test/ directories or build outputs that don't fit the
    convention).
    """
    parts = path.split("/")

    # Subfolder convention -- works regardless of nesting depth.
    if len(parts) >= 2:
        subdir = parts[-2]
        resource_type = _RESOURCE_TYPE_MAP.get(subdir)
        if resource_type:
            return resource_type

    # Flat-skill fallback: only files DIRECTLY under skill_dir qualify.
    # We don't want to vacuum up nested test/, build/, __pycache__/ trees.
    parent = "/".join(parts[:-1])
    if parent != skill_dir:
        return None

    name = parts[-1]
    if "." not in name:
        return None
    ext = "." + name.rsplit(".", 1)[-1].lower()

    if ext in {".py", ".sh", ".bash", ".js", ".ts"}:
        return "script"
    if ext in {".md", ".txt", ".rst"}:
        return "reference"
    return "asset"


def _bucket_for_type(
    manifest: "SkillResourceManifest",
    resource_type: str,
) -> "list[SkillResource] | None":
    """Return the manifest list corresponding to a resource type."""
    attr = "references" if resource_type == "reference" else f"{resource_type}s"
    return getattr(manifest, attr, None)


async def _discover_skill_resources(
    skill_md_url: str,
    auth_scheme: str = "none",
    auth_credential: str | None = None,
    auth_header_name: str | None = None,
) -> "SkillResourceManifest | None":
    """Discover companion resource files in the skill directory.

    Calls the hosting platform's tree/directory listing API (currently
    GitHub / GHES via :func:`_resolve_tree_api`) to find files alongside
    SKILL.md and classifies them into the resource manifest.

    Classification falls back to extension-based heuristics for skills
    that keep all files at the skill root (no references/ scripts/ etc.
    subfolders).  See :func:`_classify_resource`.

    Returns None when the platform is not recognised, the tree fetch
    fails, or no files classify as resources.
    """
    from ..schemas.skill_models import SkillResource, SkillResourceManifest

    tree_info = _resolve_tree_api(skill_md_url)
    if not tree_info:
        logger.debug(
            "Cannot derive tree API URL from %s — skipping resource discovery", skill_md_url
        )
        return None

    tree_url, _encoded_project, _ref, skill_dir = tree_info
    skill_dir_prefix = skill_dir + "/" if skill_dir else ""
    _, headers = _build_fetch_headers(tree_url, auth_scheme, auth_credential, auth_header_name)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Merge global GitHub auth headers (PAT / GitHub App) so private
            # repos and rate-limited public repos work.
            github_headers = await _github_auth.get_auth_headers(tree_url)
            merged_headers = {**github_headers, **headers}
            resp = await client.get(tree_url, headers=merged_headers)
            if resp.status_code >= 400:
                logger.warning(
                    "Resource discovery failed: HTTP %s for %s", resp.status_code, tree_url
                )
                return None
            payload = resp.json()

            # GitLab tree API returns a bare JSON array and paginates via
            # the x-next-page response header. Fetch remaining pages.
            if isinstance(payload, list):
                next_page = resp.headers.get("x-next-page", "")
                pages_fetched = 0
                while next_page and pages_fetched < MAX_GITLAB_TREE_PAGES:
                    page_url = _append_page_param(tree_url, next_page)
                    resp = await client.get(page_url, headers=merged_headers)
                    if resp.status_code >= 400:
                        break
                    page_payload = resp.json()
                    if not isinstance(page_payload, list):
                        break
                    payload.extend(page_payload)
                    pages_fetched += 1
                    next_page = resp.headers.get("x-next-page", "")
                if next_page:
                    logger.warning(
                        "GitLab tree pagination hit %s-page cap for %s; "
                        "results may be incomplete",
                        MAX_GITLAB_TREE_PAGES,
                        tree_url,
                    )
    except Exception as e:
        logger.warning("Resource discovery error for %s: %s", tree_url, e)
        return None

    # GitHub's Trees API returns {"sha": ..., "url": ..., "tree": [...], "truncated": bool}.
    # Extract the file list from the "tree" key; fall back to top-level list
    # so a future hosting-platform provider that returns a bare array still works.
    if isinstance(payload, dict):
        items = payload.get("tree")
        if payload.get("truncated"):
            logger.warning(
                "Trees API response truncated for %s; manifest may be incomplete",
                tree_url,
            )
    else:
        items = payload

    if not isinstance(items, list):
        return None

    manifest = SkillResourceManifest()
    for item in items:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if not path:
            continue

        # Restrict to files inside the skill directory so we don't
        # surface unrelated repo content.
        if skill_dir_prefix and not path.startswith(skill_dir_prefix):
            continue

        # Skip the SKILL.md itself and standard repo metadata files.
        name = path.rsplit("/", 1)[-1]
        if name.upper() in _SKILL_DIR_EXCLUDED_FILENAMES:
            continue

        # Drop hidden files / directories (e.g. __pycache__/, .DS_Store)
        # WITHIN the skill folder. Don't reject parts of the path that
        # appear in skill_dir itself (e.g. ".claude/skills/usage-report"
        # is a perfectly valid skill location).
        relative_path = path[len(skill_dir_prefix) :] if path.startswith(skill_dir_prefix) else path
        if any(part.startswith(".") or part.startswith("__") for part in relative_path.split("/")):
            continue

        resource_type = _classify_resource(path, skill_dir)
        if not resource_type:
            continue

        ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""

        resource = SkillResource(
            path=relative_path,
            type=resource_type,
            size_bytes=item.get("size", 0),
            language=_LANG_BY_EXT.get(ext),
        )

        bucket = _bucket_for_type(manifest, resource_type)
        if bucket is not None:
            bucket.append(resource)

    total = (
        len(manifest.references)
        + len(manifest.scripts)
        + len(manifest.agents)
        + len(manifest.assets)
    )
    if total == 0:
        return None

    logger.info(
        "Discovered %d resources: %d references, %d scripts, %d agents, %d assets",
        total,
        len(manifest.references),
        len(manifest.scripts),
        len(manifest.agents),
        len(manifest.assets),
    )
    return manifest


async def _validate_skill_md_url(
    url: str,
    auth_scheme: str = "none",
    auth_credential: str | None = None,
    auth_header_name: str | None = None,
) -> dict[str, Any]:
    """Validate SKILL.md URL is accessible and get content hash.

    Args:
        url: URL to SKILL.md file
        auth_scheme: Authentication scheme (none, bearer, api_key)
        auth_credential: Plaintext credential for URL validation
        auth_header_name: Custom header name for the credential

    Returns:
        Dict with validation result and content hash

    Raises:
        SkillUrlValidationError: If URL is not accessible or fails SSRF check
    """
    if not _is_safe_url(url):
        raise SkillUrlValidationError(
            url, "URL failed SSRF validation - private/internal addresses are not allowed"
        )

    fetch_url, fetch_headers = _build_fetch_headers(
        url, auth_scheme, auth_credential, auth_header_name
    )

    try:
        async with httpx.AsyncClient() as client:
            github_headers = await _github_auth.get_auth_headers(str(url))
            merged_headers = {**fetch_headers, **github_headers}
            response = await client.get(
                fetch_url,
                headers=merged_headers,
                follow_redirects=True,
                timeout=URL_VALIDATION_TIMEOUT,
            )

            final_url = str(response.url)
            if final_url != fetch_url and not _is_safe_url(final_url):
                logger.warning(
                    f"SSRF protection: Blocked redirect from {url} to unsafe URL {final_url}"
                )
                raise SkillUrlValidationError(url, f"Redirect to unsafe URL blocked: {final_url}")

            if response.status_code >= 400:
                raise SkillUrlValidationError(url, f"HTTP {response.status_code}")

            content_hash = hashlib.sha256(response.content).hexdigest()[:16]

            return {
                "valid": True,
                "content_version": content_hash,
                "content_updated_at": datetime.now(UTC),
            }

    except httpx.RequestError as e:
        raise SkillUrlValidationError(url, str(e)) from e


async def _parse_skill_md_content(
    url: str,
    auth_scheme: str = "none",
    auth_credential: str | None = None,
    auth_header_name: str | None = None,
) -> dict[str, Any]:
    """Parse SKILL.md content and extract metadata.

    Parses the SKILL.md markdown file to extract:
    - name: From H1 heading or YAML frontmatter
    - description: From first paragraph or YAML frontmatter
    - version: From YAML frontmatter if present
    - tags: From YAML frontmatter if present

    Also translates GitHub URLs to raw content URLs.

    Args:
        url: URL to SKILL.md file (user-provided)

    Returns:
        Dict with parsed metadata including:
        - skill_md_url: Original user-provided URL
        - skill_md_raw_url: Translated raw URL for content fetching

    Raises:
        SkillUrlValidationError: If URL is not accessible
    """
    import re

    # Translate URL to get both user-provided and raw URL
    user_url, raw_url = translate_skill_url(url)

    # Extract the repository URL from the user-provided URL
    repository_url = extract_repository_url(url)

    # Normalize to string for further validation
    raw_url_str = str(raw_url)

    # Basic scheme/hostname validation before SSRF/IP checks
    parsed_raw = urlparse(raw_url_str)
    if parsed_raw.scheme not in {"http", "https"} or not parsed_raw.hostname:
        raise SkillUrlValidationError(url, "URL must use http/https scheme and include a hostname")

    # SSRF protection - check the raw URL we'll actually fetch
    if not _is_safe_url(raw_url_str):
        raise SkillUrlValidationError(
            url, "URL failed SSRF validation - private/internal addresses are not allowed"
        )

    try:
        async with httpx.AsyncClient() as client:
            fetch_url, fetch_headers = _build_fetch_headers(
                raw_url_str,
                auth_scheme,
                auth_credential,
                auth_header_name,
            )
            if auth_scheme == "none":
                headers = fetch_headers
            elif auth_scheme == "global_credentials":
                headers = await _github_auth.get_auth_headers(fetch_url)
            else:
                github_headers = await _github_auth.get_auth_headers(fetch_url)
                headers = {**github_headers, **fetch_headers}
            response = await client.get(
                fetch_url, headers=headers, follow_redirects=True, timeout=URL_VALIDATION_TIMEOUT
            )

            # SSRF protection: validate final URL after redirects
            final_url = str(response.url)
            if final_url != str(raw_url) and not _is_safe_url(final_url):
                logger.warning(
                    f"SSRF protection: Blocked redirect from {raw_url} to unsafe URL {final_url}"
                )
                raise SkillUrlValidationError(url, f"Redirect to unsafe URL blocked: {final_url}")

            if response.status_code >= 400:
                raise SkillUrlValidationError(url, f"HTTP {response.status_code}")

            content = response.text
            result: dict[str, Any] = {
                "name": None,
                "description": None,
                "version": None,
                "tags": [],
                "content_version": hashlib.sha256(response.content).hexdigest()[:16],
                "skill_md_url": user_url,
                "skill_md_raw_url": raw_url,
                "repository_url": repository_url,
            }

            # Try to parse YAML frontmatter from multiple formats:
            # 1. Standard: --- at start of file
            # 2. Code block with ---: ```yaml\n---\n...\n---\n```
            # 3. Code block without ---: ```yaml\n...\n```
            frontmatter = None
            frontmatter_end_pos = 0

            # Format 1: Standard frontmatter at start of file
            frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            if frontmatter_match:
                frontmatter = frontmatter_match.group(1)
                frontmatter_end_pos = frontmatter_match.end()
            else:
                # Format 2: YAML code block with --- markers inside
                # Matches: ```yaml\n---\nkey: value\n---\n```
                codeblock_with_markers = re.search(
                    r"```ya?ml\s*\n---\s*\n(.*?)\n---\s*\n```",
                    content,
                    re.DOTALL | re.IGNORECASE,
                )
                if codeblock_with_markers:
                    frontmatter = codeblock_with_markers.group(1)
                    frontmatter_end_pos = codeblock_with_markers.end()
                else:
                    # Format 3: YAML code block without --- markers
                    # Matches: ```yaml\nkey: value\n```
                    codeblock_no_markers = re.search(
                        r"```ya?ml\s*\n(.*?)\n```",
                        content,
                        re.DOTALL | re.IGNORECASE,
                    )
                    if codeblock_no_markers:
                        frontmatter = codeblock_no_markers.group(1)
                        frontmatter_end_pos = codeblock_no_markers.end()

            if frontmatter:
                # Parse simple YAML key: value pairs
                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip().strip('"').strip("'")
                        if key == "name":
                            result["name"] = value
                        elif key == "description":
                            result["description"] = value
                        elif key == "version":
                            result["version"] = value
                        elif key == "tags":
                            # Handle comma-separated or YAML list
                            if value.startswith("["):
                                value = value.strip("[]")
                            result["tags"] = [
                                t.strip().strip('"').strip("'")
                                for t in value.split(",")
                                if t.strip()
                            ]

                # Remove frontmatter from content for further parsing
                content = content[frontmatter_end_pos:]

            # Extract name from first H1 heading if not in frontmatter
            if not result["name"]:
                h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                if h1_match:
                    result["name"] = h1_match.group(1).strip()

            # Extract description from first paragraph if not in frontmatter
            if not result["description"]:
                # Skip headings and find first non-empty paragraph
                lines = content.split("\n")
                paragraph_lines = []
                in_paragraph = False

                for line in lines:
                    stripped = line.strip()
                    # Skip headings and empty lines at start
                    if stripped.startswith("#"):
                        if in_paragraph:
                            break
                        continue
                    if not stripped:
                        if in_paragraph:
                            break
                        continue
                    # Skip code blocks
                    if stripped.startswith("```"):
                        if in_paragraph:
                            break
                        continue

                    in_paragraph = True
                    paragraph_lines.append(stripped)

                if paragraph_lines:
                    result["description"] = " ".join(paragraph_lines)[:500]

            # Convert name to slug format if found
            if result["name"]:
                # Convert "My Skill Name" to "my-skill-name"
                name_slug = result["name"].lower()
                name_slug = re.sub(r"[^a-z0-9]+", "-", name_slug)
                name_slug = re.sub(r"-+", "-", name_slug)
                name_slug = name_slug.strip("-")
                result["name_slug"] = name_slug

            logger.info(
                f"Parsed SKILL.md from {user_url} (raw: {raw_url}): "
                f"name={result.get('name')}, has_description={bool(result.get('description'))}"
            )
            return result

    except httpx.RequestError as e:
        raise SkillUrlValidationError(url, str(e)) from e


async def _check_skill_health(
    url: str,
    auth_scheme: str = "none",
    auth_credential_encrypted: str | None = None,
    auth_header_name: str | None = None,
) -> dict[str, Any]:
    """Check skill health by performing HEAD request to SKILL.md URL.

    Args:
        url: URL to SKILL.md file
        auth_scheme: Auth scheme for private repos
        auth_credential_encrypted: Encrypted credential
        auth_header_name: Custom header name

    Returns:
        Dict with health status
    """
    import time

    start_time = time.perf_counter()

    # SSRF protection
    if not _is_safe_url(url):
        return {
            "healthy": False,
            "status_code": None,
            "error": "URL failed SSRF validation",
            "response_time_ms": 0,
        }

    # Build auth headers for private repos
    credential = None
    if auth_scheme not in ("none", "global_credentials") and auth_credential_encrypted:
        from ..utils.credential_encryption import decrypt_credential

        credential = decrypt_credential(auth_credential_encrypted)

    fetch_url, fetch_headers = _build_fetch_headers(url, auth_scheme, credential, auth_header_name)

    try:
        async with httpx.AsyncClient() as client:
            github_headers = await _github_auth.get_auth_headers(str(url))
            merged_headers = {**github_headers, **fetch_headers}
            response = await client.head(
                fetch_url,
                headers=merged_headers,
                follow_redirects=True,
                timeout=URL_VALIDATION_TIMEOUT,
            )

            # SSRF protection: validate final URL after redirects
            final_url = str(response.url)
            if final_url != str(url) and not _is_safe_url(final_url):
                logger.warning(
                    f"SSRF protection: Blocked redirect from {url} to unsafe URL {final_url}"
                )
                response_time_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "healthy": False,
                    "status_code": None,
                    "error": f"Redirect to unsafe URL blocked: {final_url}",
                    "response_time_ms": round(response_time_ms, 2),
                }

            response_time_ms = (time.perf_counter() - start_time) * 1000

            return {
                "healthy": response.status_code < 400,
                "status_code": response.status_code,
                "error": None if response.status_code < 400 else f"HTTP {response.status_code}",
                "response_time_ms": round(response_time_ms, 2),
            }

    except httpx.RequestError as e:
        # Log detailed exception on the server, but return a generic message to the client
        logger.error("Error while checking skill health for URL %s: %s", url, e)
        response_time_ms = (time.perf_counter() - start_time) * 1000
        return {
            "healthy": False,
            "status_code": None,
            "error": "Unexpected error during health check",
            "response_time_ms": round(response_time_ms, 2),
        }


async def _compute_content_integrity(
    skill_md_url: str,
    resource_manifest: "SkillResourceManifest | None",
    auth_scheme: str = "none",
    auth_credential: str | None = None,
    auth_header_name: str | None = None,
) -> ContentIntegrity | None:
    """Compute SHA-256 hashes for SKILL.md and all companion resources.

    Returns a ContentIntegrity record with per-file hashes and a composite
    hash, or None if the SKILL.md cannot be fetched.
    """
    file_hashes: list[FileHash] = []

    async def _hash_url(
        client: httpx.AsyncClient,
        url: str,
        rel_path: str,
    ) -> FileHash | None:
        fetch_url, fetch_headers = _build_fetch_headers(
            url,
            auth_scheme,
            auth_credential,
            auth_header_name,
        )
        try:
            resp = await client.get(
                fetch_url,
                headers=fetch_headers,
                follow_redirects=True,
                timeout=URL_VALIDATION_TIMEOUT,
            )
            if resp.status_code >= 400:
                logger.warning("Integrity fetch failed for %s: HTTP %s", rel_path, resp.status_code)
                return None
            digest = hashlib.sha256(resp.content).hexdigest()
            return FileHash(path=rel_path, sha256=digest, size_bytes=len(resp.content))
        except httpx.RequestError as e:
            logger.warning("Integrity fetch error for %s: %s", rel_path, e)
            return None

    async with httpx.AsyncClient() as client:
        skill_md_hash = await _hash_url(client, skill_md_url, "SKILL.md")
        if not skill_md_hash:
            return None
        file_hashes.append(skill_md_hash)

        if resource_manifest:
            from ..utils.url_utils import derive_resource_url

            all_resources = (
                resource_manifest.references
                + resource_manifest.scripts
                + resource_manifest.agents
                + resource_manifest.assets
            )
            for res in all_resources:
                res_url = derive_resource_url(skill_md_url, res.path)
                fh = await _hash_url(client, res_url, res.path)
                if fh:
                    file_hashes.append(fh)

    sorted_entries = sorted(file_hashes, key=lambda h: h.path)
    composite_input = "".join(f"{h.path}:{h.sha256}" for h in sorted_entries)
    composite_hash = hashlib.sha256(composite_input.encode()).hexdigest()

    return ContentIntegrity(
        composite_hash=composite_hash,
        file_hashes=file_hashes,
        computed_at=datetime.now(UTC),
    )


def _decrypt_skill_auth(
    skill: SkillCard,
) -> tuple[str, str | None, str | None]:
    """Extract and decrypt authentication details from a skill.

    Returns (auth_scheme, plaintext_credential_or_none, auth_header_name).
    """
    auth_scheme = getattr(skill, "auth_scheme", "none")
    encrypted_cred = getattr(skill, "auth_credential_encrypted", None)
    credential = None
    if auth_scheme not in ("none", "global_credentials") and encrypted_cred:
        from ..utils.credential_encryption import decrypt_credential

        credential = decrypt_credential(encrypted_cred)
    return auth_scheme, credential, getattr(skill, "auth_header_name", None)


async def _fetch_authenticated_content(
    url: str,
    skill: SkillCard,
    *,
    max_size: int | None = None,
    timeout: float = 30.0,
) -> "httpx.Response":
    """Fetch content from a URL using the skill's encrypted credentials.

    Handles SSRF validation, credential decryption, header building,
    and redirect safety checks.

    Raises:
        SkillContentSSRFError: URL or redirect target fails SSRF check.
        SkillContentFetchError: Upstream returned an HTTP error or was unreachable.
        SkillContentTooLargeError: Response body exceeds max_size.
    """
    from ..exceptions import (
        SkillContentFetchError,
        SkillContentSSRFError,
        SkillContentTooLargeError,
    )

    if not _is_safe_url(url):
        raise SkillContentSSRFError(url)

    auth_scheme, credential, auth_header_name = _decrypt_skill_auth(skill)
    fetch_url, fetch_headers = _build_fetch_headers(
        url,
        auth_scheme,
        credential,
        auth_header_name,
    )

    if auth_scheme == "none":
        merged_headers = fetch_headers
    elif auth_scheme == "global_credentials":
        merged_headers = await _github_auth.get_auth_headers(fetch_url)
    else:
        github_headers = await _github_auth.get_auth_headers(fetch_url)
        merged_headers = {**github_headers, **fetch_headers}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                fetch_url,
                headers=merged_headers,
                follow_redirects=True,
                timeout=timeout,
            )

            final_url = str(response.url)
            if final_url != fetch_url and not _is_safe_url(final_url):
                raise SkillContentSSRFError(final_url)

            if response.status_code >= 400:
                raise SkillContentFetchError(
                    url,
                    f"HTTP {response.status_code}",
                )

            if max_size and len(response.content) > max_size:
                raise SkillContentTooLargeError(max_size)

            return response
    except httpx.RequestError as e:
        logger.error("Failed to fetch from %s: %s", url, e)
        raise SkillContentFetchError(url, str(e))


async def _check_drift_inline(
    service: "SkillService",
    skill_path: str,
    skill: SkillCard,
    file_path: str,
    content_bytes: bytes,
) -> None:
    """Compare fetched content against the stored integrity baseline.

    Runs as a fire-and-forget background task so it never blocks the
    content response.  Updates the DB only when a change is detected
    (or when a previously-drifted file returns to its baseline).
    """
    integrity = skill.content_integrity
    if not integrity or not integrity.file_hashes:
        return

    baseline = {fh.path: fh.sha256 for fh in integrity.file_hashes}
    expected = baseline.get(file_path)
    if expected is None:
        return

    actual = hashlib.sha256(content_bytes).hexdigest()
    file_drifted = actual != expected

    previously_drifted = file_path in (integrity.drifted_files or [])
    now = datetime.now(UTC).isoformat()

    try:
        if file_drifted == previously_drifted:
            await service.update_skill(
                skill_path,
                {"content_integrity.last_drift_check": now},
            )
            return

        current_drifted = list(integrity.drifted_files or [])
        if file_drifted and file_path not in current_drifted:
            current_drifted.append(file_path)
        elif not file_drifted and file_path in current_drifted:
            current_drifted.remove(file_path)

        current_tags = list(skill.tags or [])
        drift_tag = "content-drifted"
        if current_drifted and drift_tag not in current_tags:
            current_tags.append(drift_tag)
        elif not current_drifted and drift_tag in current_tags:
            current_tags.remove(drift_tag)

        combined_updates: dict[str, Any] = {
            "content_integrity.drift_detected": bool(current_drifted),
            "content_integrity.last_drift_check": now,
            "content_integrity.drifted_files": current_drifted,
            "tags": current_tags,
        }
        await service.update_skill(skill_path, combined_updates)

        if current_drifted:
            await service.toggle_skill(skill_path, enabled=False)
            logger.warning(
                "Drift detected for %s in skill %s, skill disabled",
                file_path,
                skill_path,
            )
        else:
            await service.toggle_skill(skill_path, enabled=True)
            logger.info("Drift cleared for skill %s, skill re-enabled", skill_path)
    except Exception:
        logger.debug("Failed to persist drift state for %s", skill_path, exc_info=True)


def _build_skill_card(
    request: SkillRegistrationRequest,
    path: str,
    owner: str | None,
    content_version: str | None,
    content_updated_at: datetime | None,
    skill_md_raw_url: str | None = None,
    resource_manifest: "SkillResourceManifest | None" = None,
    content_integrity: ContentIntegrity | None = None,
) -> SkillCard:
    """Build SkillCard from registration request.

    Args:
        request: Registration request
        path: Skill path
        owner: Owner username/email
        content_version: Content hash
        content_updated_at: Content update timestamp
        skill_md_raw_url: Raw URL for fetching SKILL.md content
        resource_manifest: Discovered companion resource files

    Returns:
        SkillCard instance
    """
    # Convert metadata dict to SkillMetadata if provided
    # Use explicit version field if provided, otherwise fall back to metadata.version
    version = request.version
    if not version and request.metadata:
        version = request.metadata.get("version")

    metadata = None
    if request.metadata or version:
        metadata = SkillMetadata(
            author=request.metadata.get("author") if request.metadata else None,
            version=version,
            extra={k: v for k, v in request.metadata.items() if k not in ("author", "version")}
            if request.metadata
            else {},
        )

    # Encrypt credential if provided
    auth_credential_encrypted = None
    credential_updated_at = None
    if getattr(request, "auth_credential", None) and getattr(
        request, "auth_scheme", "none"
    ) not in ("none", "global_credentials"):
        from ..utils.credential_encryption import encrypt_credential

        auth_credential_encrypted = encrypt_credential(request.auth_credential)
        credential_updated_at = datetime.now(UTC)

    return SkillCard(
        path=path,
        name=request.name,
        description=request.description,
        skill_md_url=request.skill_md_url,
        skill_md_raw_url=skill_md_raw_url,
        repository_url=request.repository_url,
        license=request.license,
        compatibility=request.compatibility,
        requirements=request.requirements,
        target_agents=request.target_agents,
        metadata=metadata,
        allowed_tools=request.allowed_tools,
        tags=request.tags,
        visibility=request.visibility,
        allowed_groups=request.allowed_groups,
        owner=owner,
        is_enabled=True,
        status=request.status,
        auth_scheme=getattr(request, "auth_scheme", "none"),
        auth_credential_encrypted=auth_credential_encrypted,
        auth_header_name=getattr(request, "auth_header_name", None),
        credential_updated_at=credential_updated_at,
        resource_manifest=resource_manifest,
        content_version=content_version,
        content_updated_at=content_updated_at,
        content_integrity=content_integrity,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


class SkillService:
    """Service for skill CRUD operations.

    Simplified design with no in-memory state duplication.
    Database is the source of truth.
    """

    def __init__(self):
        self._repo: SkillRepositoryBase | None = None
        self._search_repo: SearchRepositoryBase | None = None

    def _get_repo(self) -> SkillRepositoryBase:
        """Lazy initialization of repository."""
        if self._repo is None:
            self._repo = get_skill_repository()
        return self._repo

    def _get_search_repo(self) -> SearchRepositoryBase:
        """Lazy initialization of search repository."""
        if self._search_repo is None:
            self._search_repo = get_search_repository()
        return self._search_repo

    async def register_skill(
        self,
        request: SkillRegistrationRequest,
        owner: str | None = None,
        validate_url: bool = True,
    ) -> SkillCard:
        """Register a new skill.

        Args:
            request: Skill registration request
            owner: Owner username/email for access control
            validate_url: Whether to validate SKILL.md URL

        Returns:
            Created SkillCard

        Raises:
            SkillUrlValidationError: If URL validation fails
            SkillAlreadyExistsError: If skill name exists
        """
        # Generate path
        path = normalize_skill_path(request.name)

        # Translate URL to get the raw URL for content fetching
        _, raw_url = translate_skill_url(str(request.skill_md_url))

        # Validate URL and get content hash (validate the raw URL)
        content_version = None
        content_updated_at = None

        if validate_url:
            validation = await _validate_skill_md_url(
                raw_url,
                auth_scheme=getattr(request, "auth_scheme", "none"),
                auth_credential=getattr(request, "auth_credential", None),
                auth_header_name=getattr(request, "auth_header_name", None),
            )
            content_version = validation["content_version"]
            content_updated_at = validation["content_updated_at"]

        # Discover companion resource files (non-fatal)
        resource_manifest = None
        try:
            resource_manifest = await _discover_skill_resources(
                raw_url,
                auth_scheme=getattr(request, "auth_scheme", "none"),
                auth_credential=getattr(request, "auth_credential", None),
                auth_header_name=getattr(request, "auth_header_name", None),
            )
        except Exception as e:
            logger.warning("Resource discovery failed for %s: %s", request.name, e)

        # Compute content integrity (SKILL.md + all resources)
        content_integrity = None
        try:
            content_integrity = await _compute_content_integrity(
                raw_url,
                resource_manifest,
                auth_scheme=getattr(request, "auth_scheme", "none"),
                auth_credential=getattr(request, "auth_credential", None),
                auth_header_name=getattr(request, "auth_header_name", None),
            )
        except Exception as e:
            logger.warning("Content integrity computation failed for %s: %s", request.name, e)

        # Build SkillCard
        skill = _build_skill_card(
            request=request,
            path=path,
            owner=owner,
            content_version=content_version,
            content_updated_at=content_updated_at,
            skill_md_raw_url=raw_url,
            resource_manifest=resource_manifest,
            content_integrity=content_integrity,
        )

        # Save to repository
        repo = self._get_repo()
        created_skill = await repo.create(skill)

        # Index for search
        try:
            search_repo = self._get_search_repo()
            await search_repo.index_skill(
                path=path,
                skill=created_skill,
                is_enabled=True,
            )
        except Exception as e:
            logger.warning(f"Failed to index skill for search: {e}")

        logger.info(f"Registered skill: {path}")
        return created_skill

    async def get_skill(
        self,
        path: str,
    ) -> SkillCard | None:
        """Get a skill by path."""
        normalized = normalize_skill_path(path)
        repo = self._get_repo()
        return await repo.get(normalized)

    async def list_skills(
        self,
        include_disabled: bool = False,
        tag: str | None = None,
        visibility: str | None = None,
        registry_name: str | None = None,
    ) -> list[SkillInfo]:
        """List skills with optional filtering.

        Uses database-level filtering for performance.

        Args:
            include_disabled: Whether to include disabled skills
            tag: Filter by tag
            visibility: Filter by visibility
            registry_name: Filter by registry

        Returns:
            List of SkillInfo summaries
        """
        repo = self._get_repo()
        skills = await repo.list_filtered(
            include_disabled=include_disabled,
            tag=tag,
            visibility=visibility,
            registry_name=registry_name,
        )

        return [
            SkillInfo(
                id=s.id,
                path=s.path,
                name=s.name,
                description=s.description,
                skill_md_url=str(s.skill_md_url),
                skill_md_raw_url=str(s.skill_md_raw_url) if s.skill_md_raw_url else None,
                repository_url=s.repository_url,
                tags=s.tags,
                author=s.metadata.author if s.metadata else None,
                version=s.metadata.version if s.metadata else None,
                metadata=s.metadata,
                compatibility=s.compatibility,
                target_agents=s.target_agents,
                is_enabled=s.is_enabled,
                visibility=s.visibility,
                allowed_groups=s.allowed_groups,
                registry_name=s.registry_name,
                owner=s.owner,
                auth_scheme=s.auth_scheme,
                auth_header_name=s.auth_header_name,
                num_stars=s.num_stars,
                health_status=s.health_status,
                last_checked_time=s.last_checked_time,
                status=s.status,
            )
            for s in skills
        ]

    async def list_skills_for_user(
        self,
        user_context: dict[str, Any] | None,
        include_disabled: bool = False,
        tag: str | None = None,
    ) -> list[SkillInfo]:
        """List skills filtered by user's visibility access.

        Args:
            user_context: User context with groups and username
            include_disabled: Whether to include disabled skills
            tag: Filter by tag

        Returns:
            List of SkillInfo visible to user
        """
        all_skills = await self.list_skills(
            include_disabled=include_disabled,
            tag=tag,
        )

        if not user_context:
            # Anonymous - only public
            return [s for s in all_skills if s.visibility == VisibilityEnum.PUBLIC]

        if user_context.get("is_admin"):
            return all_skills

        user_groups = set(user_context.get("groups", []))
        username = user_context.get("username", "")

        filtered = []
        for skill in all_skills:
            if skill.visibility == VisibilityEnum.PUBLIC:
                filtered.append(skill)
            elif skill.visibility == VisibilityEnum.PRIVATE:
                # Check owner directly from SkillInfo (no N+1 query)
                if skill.owner == username:
                    filtered.append(skill)
            elif skill.visibility == VisibilityEnum.GROUP:
                if user_groups & set(skill.allowed_groups):
                    filtered.append(skill)

        return filtered

    async def get_skills_paginated(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[SkillCard], int]:
        """Get a page of skills with total count.

        Used for unrestricted users (admins) where DB-level pagination
        is correct because no skills are filtered out by access control.

        Note: list_paginated and count are separate DB calls, so total_count
        may be slightly inconsistent if skills are added/removed between calls.
        This is standard for offset-based pagination.

        Args:
            skip: Number of skills to skip.
            limit: Maximum number of skills to return.

        Returns:
            Tuple of (page of skills, total count of all skills).
        """
        repo = self._get_repo()
        skills = await repo.list_paginated(skip=skip, limit=limit)
        total = await repo.count()
        return skills, total

    async def update_skill(
        self,
        path: str,
        updates: dict[str, Any],
    ) -> SkillCard | None:
        """Update a skill."""
        normalized = normalize_skill_path(path)
        repo = self._get_repo()
        updated = await repo.update(normalized, updates)

        if updated:
            # Update search index
            try:
                search_repo = self._get_search_repo()
                await search_repo.index_skill(
                    path=normalized,
                    skill=updated,
                    is_enabled=updated.is_enabled,
                )
            except Exception as e:
                logger.warning(f"Failed to update skill in search index: {e}")
            logger.info(f"Updated skill: {normalized}")

        return updated

    async def delete_skill(
        self,
        path: str,
    ) -> bool:
        """Delete a skill."""
        normalized = normalize_skill_path(path)
        repo = self._get_repo()
        success = await repo.delete(normalized)

        if success:
            # Remove from search index
            try:
                search_repo = self._get_search_repo()
                await search_repo.remove_entity(normalized)
            except Exception as e:
                logger.warning(f"Failed to remove skill from search index: {e}")
            logger.info(f"Deleted skill: {normalized}")

        return success

    async def toggle_skill(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Toggle skill enabled state."""
        normalized = normalize_skill_path(path)
        repo = self._get_repo()
        success = await repo.set_state(normalized, enabled)

        if success:
            # Update search index
            skill = await repo.get(normalized)
            if skill:
                try:
                    search_repo = self._get_search_repo()
                    await search_repo.index_skill(
                        path=normalized,
                        skill=skill,
                        is_enabled=enabled,
                    )
                except Exception as e:
                    logger.warning(f"Failed to update skill in search index: {e}")
            logger.info(f"Toggled skill {normalized} to enabled={enabled}")

        return success

    async def parse_skill_md(
        self,
        url: str,
        auth_scheme: str = "none",
        auth_credential: str | None = None,
        auth_header_name: str | None = None,
    ) -> dict[str, Any]:
        """Parse SKILL.md content and extract metadata.

        Args:
            url: URL to SKILL.md file
            auth_scheme: Auth scheme (none, global_credentials, bearer, api_key)
            auth_credential: Plaintext credential for bearer/api_key
            auth_header_name: Custom header name for api_key scheme

        Returns:
            Dict with parsed metadata (name, description, version, tags)
        """
        return await _parse_skill_md_content(
            url,
            auth_scheme=auth_scheme,
            auth_credential=auth_credential,
            auth_header_name=auth_header_name,
        )

    async def check_skill_health(
        self,
        path: str,
    ) -> dict[str, Any]:
        """Check skill health by performing HEAD request to SKILL.md URL.

        Args:
            path: Skill path

        Returns:
            Dict with health status
        """
        from datetime import UTC, datetime

        normalized = normalize_skill_path(path)
        repo = self._get_repo()
        skill = await repo.get(normalized)

        if not skill:
            return {
                "healthy": False,
                "status_code": None,
                "error": "Skill not found",
                "response_time_ms": 0,
            }

        # Use raw URL for health check (more reliable, returns actual content)
        url = skill.skill_md_raw_url or skill.skill_md_url
        result = await _check_skill_health(
            str(url),
            auth_scheme=getattr(skill, "auth_scheme", "none"),
            auth_credential_encrypted=getattr(skill, "auth_credential_encrypted", None),
            auth_header_name=getattr(skill, "auth_header_name", None),
        )

        # Persist health status to database
        health_status = "healthy" if result.get("healthy") else "unhealthy"
        checked_time = datetime.now(UTC)

        await repo.update(
            normalized,
            {
                "health_status": health_status,
                "last_checked_time": checked_time.isoformat(),
            },
        )

        logger.info(f"Updated health status for skill {normalized}: {health_status}")

        return result

    async def update_rating(
        self,
        path: str,
        username: str,
        rating: int,
    ) -> float:
        """Update rating for a skill.

        Args:
            path: Skill path
            username: The user who submitted rating
            rating: integer between 1-5

        Returns:
            Updated average rating

        Raises:
            ValueError: If skill not found or invalid rating
        """
        from . import rating_service

        normalized = normalize_skill_path(path)
        repo = self._get_repo()

        # Get existing skill
        existing_skill = await repo.get(normalized)
        if not existing_skill:
            logger.error(f"Cannot update skill at path '{normalized}': not found")
            raise ValueError(f"Skill not found at path: {normalized}")

        # Validate rating using shared service
        rating_service.validate_rating(rating)

        # Convert to dict for modification - use mode="json" to serialize HttpUrl to strings
        skill_dict = existing_skill.model_dump(mode="json")

        # Ensure rating_details is a list
        if "rating_details" not in skill_dict or skill_dict["rating_details"] is None:
            skill_dict["rating_details"] = []

        # Update rating details using shared service
        updated_details, is_new_rating = rating_service.update_rating_details(
            skill_dict["rating_details"], username, rating
        )
        skill_dict["rating_details"] = updated_details

        # Calculate average rating using shared service
        skill_dict["num_stars"] = rating_service.calculate_average_rating(
            skill_dict["rating_details"]
        )

        # Save to repository
        await repo.update(normalized, skill_dict)

        logger.info(
            f"Updated rating for skill {normalized}: user {username} rated {rating}, "
            f"new average: {skill_dict['num_stars']:.2f}"
        )

        return skill_dict["num_stars"]

    async def get_rating(
        self,
        path: str,
    ) -> dict[str, Any]:
        """Get rating information for a skill.

        Args:
            path: Skill path

        Returns:
            Dict with num_stars and rating_details

        Raises:
            ValueError: If skill not found
        """
        normalized = normalize_skill_path(path)
        repo = self._get_repo()

        skill = await repo.get(normalized)
        if not skill:
            raise ValueError(f"Skill not found at path: {normalized}")

        return {
            "num_stars": skill.num_stars,
            "rating_details": skill.rating_details,
        }


# Singleton instance
_skill_service: SkillService | None = None


def get_skill_service() -> SkillService:
    """Get or create skill service singleton."""
    global _skill_service
    if _skill_service is None:
        _skill_service = SkillService()
    return _skill_service
