"""
GitLab-specific URL utilities for API v4 endpoint translation.

Provides functions to translate GitLab web URLs (/-/raw/) to API v4
authenticated endpoints, derive resource URLs within GitLab repos,
and build tree API URLs for resource discovery.

These helpers are not needed for GitHub-hosted skills and can be
omitted from upstream distributions.
"""

import re
from urllib.parse import quote as url_quote
from urllib.parse import unquote


class _GitLabParts:
    """Parsed components of a GitLab /-/raw/ or API v4 file URL."""

    __slots__ = ("base", "project", "ref", "filepath")

    def __init__(self, base: str, project: str, ref: str, filepath: str):
        self.base = base
        self.project = project
        self.ref = ref
        self.filepath = filepath

    @property
    def encoded_project(self) -> str:
        return url_quote(self.project, safe="")

    @property
    def file_dir(self) -> str:
        """Directory containing the file (everything before the last /)."""
        return self.filepath.rsplit("/", 1)[0] if "/" in self.filepath else ""


def parse_gitlab_url(url: str) -> _GitLabParts | None:
    """Parse a GitLab raw-web or API v4 file URL into its components.

    Handles two forms:
      /-/raw/  web URLs:  https://host/group/repo/-/raw/branch/path/to/file
      API v4 file URLs:   https://host/api/v4/projects/group%2Frepo/repository/files/path%2Fto%2Ffile/raw?ref=branch

    Returns None if the URL matches neither pattern.
    """
    m = re.match(r"(https?://[^/]+)/(.+?)/-/raw/([^/]+)/(.+)$", url)
    if m:
        return _GitLabParts(*m.groups())

    api_m = re.match(
        r"(https?://[^/]+)/api/v4/projects/([^/]+)/repository/files/(.+?)/raw\?ref=(.+)$",
        url,
    )
    if api_m:
        base, encoded_project, encoded_path, ref = api_m.groups()
        return _GitLabParts(base, unquote(encoded_project), ref, unquote(encoded_path))

    return None


def translate_gitlab_to_api_url(url: str) -> str | None:
    """Translate a GitLab web raw URL to a GitLab API v4 raw file endpoint.

    GitLab's /-/raw/ web URLs require session cookies for private repos.
    The API v4 endpoint accepts PRIVATE-TOKEN header authentication.

    Returns None if the URL doesn't match the expected GitLab pattern.
    """
    parts = parse_gitlab_url(url)
    if not parts:
        return None
    encoded_path = url_quote(parts.filepath, safe="")
    return (
        f"{parts.base}/api/v4/projects/{parts.encoded_project}"
        f"/repository/files/{encoded_path}/raw?ref={parts.ref}"
    )


def derive_gitlab_resource_url(skill_md_url: str, resource_path: str) -> str | None:
    """Derive a resource URL from a GitLab SKILL.md URL.

    Returns an API v4 file URL for the resource, or None if the
    skill_md_url is not a recognised GitLab URL.
    """
    parts = parse_gitlab_url(skill_md_url)
    if not parts:
        return None

    file_dir = parts.file_dir
    new_path = f"{file_dir}/{resource_path}" if file_dir else resource_path
    encoded_path = url_quote(new_path, safe="")
    return (
        f"{parts.base}/api/v4/projects/{parts.encoded_project}"
        f"/repository/files/{encoded_path}/raw?ref={parts.ref}"
    )


def translate_gitlab_tree_api_url(skill_md_url: str) -> tuple[str, str, str, str] | None:
    """Derive a GitLab API v4 tree endpoint from a skill's URL.

    Returns (tree_api_url, project_encoded, ref, skill_dir) or None if not a
    GitLab URL.  *skill_dir* is the directory prefix that the tree API will
    prepend to every returned path (e.g. ``skills/jira-to-pr``).
    """
    parts = parse_gitlab_url(skill_md_url)
    if not parts:
        return None

    skill_dir = parts.file_dir
    if not skill_dir:
        return None

    tree_url = (
        f"{parts.base}/api/v4/projects/{parts.encoded_project}/repository/tree"
        f"?path={url_quote(skill_dir, safe='')}&ref={parts.ref}&recursive=true&per_page=100"
    )
    return tree_url, parts.encoded_project, parts.ref, skill_dir
