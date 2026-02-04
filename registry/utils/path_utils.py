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
