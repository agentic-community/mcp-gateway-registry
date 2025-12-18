from __future__ import annotations

import os

from ..config import (
    UpstreamOAuthProviderConfig,
)


def load_upstream_oauth_client_secret(
    *,
    provider: UpstreamOAuthProviderConfig,
) -> str:
    ref = provider.client_secret_ref
    if ref.kind == "env":
        if ref.env_var is None:
            raise ValueError("client_secret_ref.env_var is required for env secrets")
        value = os.environ.get(ref.env_var)
        if value is None or not value.strip():
            raise ValueError(f"Missing upstream OAuth client secret env var: {ref.env_var}")
        return value.strip()

    if ref.kind == "file":
        if ref.path is None:
            raise ValueError("client_secret_ref.path is required for file secrets")
        try:
            raw = ref.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError("Failed to read upstream OAuth client secret file") from exc
        if not raw.strip():
            raise ValueError("Upstream OAuth client secret file is empty")
        return raw.strip()

    raise ValueError("Unsupported upstream OAuth client secret ref kind")

