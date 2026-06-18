// MCP Gateway Registry — Cedar Authorization Policies
//
// Evaluated by the cedarpy engine (Rust Cedar Policy bindings).
// These policies are evaluated against every request that passes through
// the auth-server /validate endpoint when AUTHORIZER_MODE=custom or AUTHORIZER_MODE=both.
//
// Entity model:
//   Principal  — attrs: username (String), groups (Set<String>), scopes (Set<String>)
//   Action     — entity ID is the HTTP method: "GET", "POST", "PUT", "DELETE", "PATCH"
//   Resource   — attrs: path (String)
//
// In BOTH mode, principal attributes come from the native JWT (real user identity).
// In CUSTOM mode, principal is anonymous with empty groups/scopes.
//
// Decision logic: default-deny; permit fires → allow; forbid overrides permit.

// ─────────────────────────────────────────────────────────────────────────────
// PERMIT rules
// ─────────────────────────────────────────────────────────────────────────────

// Rule 1: Admins can perform any action on any resource.
permit(
    principal,
    action,
    resource
)
when {
    principal.groups.contains("mcp-registry-admins") ||
    principal.groups.contains("registry-admins")
};

// Rule 2: LOB1 users can read any resource.
permit(
    principal,
    action == Action::"GET",
    resource
)
when {
    principal.groups.contains("registry-users-lob1")
};

// Rule 3: LOB1 users can POST to any resource.
permit(
    principal,
    action == Action::"POST",
    resource
)
when {
    principal.groups.contains("registry-users-lob1")
};

// Rule 4: LOB2 users have read-only access.
permit(
    principal,
    action == Action::"GET",
    resource
)
when {
    principal.groups.contains("registry-users-lob2")
};

// Rule 5: M2M service accounts with unrestricted scopes get full access.
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

// Rule 6: Block all access to the admin API for non-admins.
forbid(
    principal,
    action,
    resource
)
when {
    resource.path like "/api/admin*" &&
    !principal.groups.contains("mcp-registry-admins") &&
    !principal.groups.contains("registry-admins")
};

// Rule 7: Block DELETE on any resource for non-admins.
forbid(
    principal,
    action == Action::"DELETE",
    resource
)
when {
    !principal.groups.contains("mcp-registry-admins") &&
    !principal.groups.contains("registry-admins")
};
