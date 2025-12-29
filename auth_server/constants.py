from __future__ import annotations

import os

JWT_ISSUER: str = "mcp-auth-server"
JWT_AUDIENCE: str = "mcp-registry"

MAX_TOKEN_LIFETIME_HOURS: int = 24
DEFAULT_TOKEN_LIFETIME_HOURS: int = 8

MAX_TOKENS_PER_USER_PER_HOUR: int = int(os.environ.get("MAX_TOKENS_PER_USER_PER_HOUR", "100"))

