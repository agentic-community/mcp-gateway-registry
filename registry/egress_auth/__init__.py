"""Per-user egress credential vault (third-party OBO support).

This package implements the generic OAuth authorization-code engine, the
per-provider config table, and the orchestration service that vends per-user
third-party access tokens on the egress (``mcp_proxy``) hop. Token material
lives only in the pluggable SecretStore (``registry.secrets``), never in the
app DB.

"""
