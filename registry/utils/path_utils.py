"""
Utility functions for path handling.

Extracted to avoid code duplication across routes.
"""

import re


def normalize_skill_path(
    path: str,
) -> str:
    """Normalize skill path, ensuring /skills/ prefix.

    Args:
        path: Raw path string

    Returns:
        Normalized path with /skills/ prefix
    """
    # Remove leading/trailing whitespace
    path = path.strip()

    # Remove duplicate slashes
    path = re.sub(r"/+", "/", path)

    # Ensure /skills/ prefix
    if not path.startswith("/skills/"):
        # Remove leading slash if present
        path = path.lstrip("/")
        path = f"/skills/{path}"

    return path


def extract_skill_name(
    path: str,
) -> str:
    """Extract skill name from path.

    Args:
        path: Skill path (e.g., /skills/pdf-processing)

    Returns:
        Skill name (e.g., pdf-processing)
    """
    normalized = normalize_skill_path(path)
    return normalized.replace("/skills/", "").strip("/")


def validate_skill_name(
    name: str,
) -> bool:
    """Validate skill name follows Agent Skills spec.

    Args:
        name: Skill name to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r"^[a-z0-9]+(-[a-z0-9]+)*$"
    return bool(re.match(pattern, name))


# Separator between a server's registered path and a version label in the
# document id of an inactive version (e.g. "/github:v2"). Declared here rather
# than inline so the format has one definition: it was built with an f-string in
# seven places across three modules, and PR #1869 fixed four separate defects
# that were all two components disagreeing about an identifier's canonical form.
VERSION_ID_SEPARATOR: str = ":"


def make_version_id(
    path: str,
    version: str,
) -> str:
    """Build the document id of a server's inactive version.

    Idempotent: a path that already carries this version's suffix is returned
    unchanged, so a caller that cannot tell whether it holds an active path or a
    version id does not produce "/github:v2:v2".

    Args:
        path: The server's registered path (e.g. ``/github``).
        version: The version label (e.g. ``v2``).

    Returns:
        The version document id (e.g. ``/github:v2``), or ``path`` unchanged
        when it already ends with that suffix.

    Example:
        >>> make_version_id("/github", "v2")
        '/github:v2'
        >>> make_version_id("/github:v2", "v2")
        '/github:v2'
    """
    suffix = f"{VERSION_ID_SEPARATOR}{version}"
    if path.endswith(suffix):
        return path
    return f"{path}{suffix}"
