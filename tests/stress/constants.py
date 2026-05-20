"""Shared constants for the stress test harness."""

from typing import Literal

EntityType = Literal["servers", "agents", "skills"]

ENTITY_TYPES: tuple[EntityType, ...] = ("servers", "agents", "skills")
TARGET_SIZES: tuple[int, ...] = (100, 500, 1000)
BACKENDS: tuple[str, ...] = ("mongodb-ce", "documentdb", "mongodb", "mongodb-atlas", "file")

STRESS_TAG: str = "stress-test"
STRESS_SUFFIX_TEMPLATE: str = "-stress-{index:05d}"

ANTHROPIC_REGISTRY_BASE: str = "https://registry.modelcontextprotocol.io/v0/servers"
ANTHROPIC_PAGE_LIMIT: int = 100

ANS_DEFAULT_ENDPOINT: str = "https://api.godaddy.com"
ANS_AGENTS_PATH: str = "/v1/agents"
ANS_PAGE_LIMIT: int = 100

GITHUB_TREE_API: str = "https://api.github.com/repos/anthropics/skills/git/trees/main?recursive=1"
GITHUB_RAW_BASE: str = "https://raw.githubusercontent.com/anthropics/skills/main"

DEFAULT_BASE_URL: str = "http://localhost"
DEFAULT_TOKEN_FILE: str = ".token"
DEFAULT_CONCURRENCY: int = 3
HTTP_TIMEOUT_SECONDS: float = 120.0
