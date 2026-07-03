"""Pluggable secret store for per-user egress tokens.

This is the SINGLE SOURCE OF TRUTH for egress token state -- there is no
companion app-DB metadata table. Tokens are addressed by deterministic
namespacing keyed on ``(auth_method, user_id, provider, server_path)`` so the
``secret_ref`` is always recomputable and "list my connections" is a vault
prefix scan.

Backends (selected by ``settings.secret_store_backend`` via ``factory.py``):
- ``secrets-manager``: one AWS Secrets Manager secret per principal (JSON map).
- ``openbao``: per-entry OpenBao KV v2 paths.

Mirrors the ``registry.repositories`` factory + ABC pattern.
"""
