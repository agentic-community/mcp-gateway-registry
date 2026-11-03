"""Shared helper that decides whether a reported version is a clean release tag.

A clean release tag is an optional leading v, then MAJOR.MINOR.PATCH, then an
optional -pN patch suffix (for example 1.24.4, v1.0.20, v1.0.22-p1). Anything
else is a build: git-describe strings, bare commit hashes, dev, and fork or
branch-suffixed versions such as 1.32.1+isg.0.15.1.2.gff7840030.

The Version Adoption table, the Community vs Internal split, and the Cloud
Detection narrative all use this one rule, so a version is labeled the same
way in every section.
"""

import re

RELEASE_VERSION_PATTERN = re.compile(r"^v?\d+\.\d+\.\d+(-p\d+)?$")


def is_release_version(
    version: str,
) -> bool:
    """Return True when the version string is a clean release tag."""
    return bool(RELEASE_VERSION_PATTERN.match((version or "").strip()))
