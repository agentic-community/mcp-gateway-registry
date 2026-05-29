// MCP Gateway Registry — Cedar Authorization Policies
//
// These policies are evaluated against every request that passes through
// the auth-server /validate endpoint when AUTHORIZER_MODE=custom or AUTHORIZER_MODE=both.
//
// Principal attributes available:
//   principal.username  — authenticated username
//   principal.groups    — list of group names (e.g. ["mcp-registry-admins"])
//   principal.scopes    — list of OAuth2 scopes
//   principal.auth_method — authentication method (keycloak, cognito, …)
//
// Resource attributes available:
//   resource.path       — request path (e.g. "/api/servers")
//   resource.method     — HTTP method (GET, POST, PUT, DELETE, PATCH)
//
// Action represents the HTTP method as a string (Action::"GET", Action::"POST", etc.)
//
// Note: This file uses a practical Cedar subset. Supported constructs:
//   permit/forbid with principal/action/resource conditions
//   when { } clauses with: .like, ==, !=, &&, ||, !, in, principal.groups.contains()

// ─────────────────────────────────────────────────────────────────────────────
// PERMIT rules
// ─────────────────────────────────────────────────────────────────────────────

// Admins can perform any action on any resource.
permit(
    principal,
    action,
    resource
)
when {
    principal.groups.contains("mcp-registry-admins")
};

// LOB1 users can read any resource and invoke MCP tool endpoints.
permit(
    principal,
    action == Action::"GET",
    resource
)
when {
    principal.groups.contains("registry-users-lob1")
};

permit(
    principal,
    action == Action::"POST",
    resource
)
when {
    principal.groups.contains("registry-users-lob1") &&
    resource.path.like("*/mcp/*")
};

// LOB2 users have read-only access.
permit(
    principal,
    action == Action::"GET",
    resource
)
when {
    principal.groups.contains("registry-users-lob2")
};

// M2M service accounts (any client with the m2m-servers-unrestricted scope) get full access.
permit(
    principal,
    action,
    resource
)
when {
    principal.scopes.contains("mcp-servers-unrestricted/read") &&
    principal.scopes.contains("mcp-servers-unrestricted/execute")
};

// ─────────────────────────────────────────────────────────────────────────────
// FORBID rules — evaluated after all permits; a forbid overrides any permit.
// ─────────────────────────────────────────────────────────────────────────────

// Block write access to the admin API for non-admins.
forbid(
    principal,
    action,
    resource
)
when {
    resource.path.like("/api/admin/*") &&
    !principal.groups.contains("mcp-registry-admins")
};

// Block DELETE on servers/agents for non-admins.
forbid(
    principal,
    action == Action::"DELETE",
    resource
)
when {
    !principal.groups.contains("mcp-registry-admins")
};
