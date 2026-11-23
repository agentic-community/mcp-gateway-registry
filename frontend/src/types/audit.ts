/**
 * One audit record as returned by `/api/audit/events`. The three streams share
 * this shape loosely: `registry_api` fills `request`/`response`/`action`,
 * `mcp_access` fills `mcp_server`/`mcp_request`/`mcp_response`, and
 * `token_mint` carries flat identity and token fields with no `identity` block.
 */
export interface AuditEvent {
  _id?: string;
  timestamp: string;
  request_id: string;
  log_type: string;
  version?: string;
  correlation_id?: string;
  // Absent on the token_mint stream, which carries no `identity` block (its
  // identity fields are flat, below).
  identity?: {
    username: string;
    auth_method: string;
    provider?: string;
    groups?: string[];
    scopes?: string[];
    is_admin: boolean;
    credential_type: string;
    credential_hint?: string;
    // Durable / IdP identity claims (present when the token carries them,
    // notably Entra delegated/OBO access tokens). `username` stays the
    // human-facing display value; these are for correlation / contact. The API
    // serializes a claim the token did not carry as an explicit `null`, and
    // records written before the claims existed omit the key entirely.
    subject?: string | null;
    canonical_id?: string | null;
    principal_name?: string | null;
    object_id?: string | null;
    tenant_id?: string | null;
    app_id?: string | null;
  };
  request?: {
    method: string;
    path: string;
    query_params?: Record<string, unknown>;
    client_ip: string;
    forwarded_for?: string;
    user_agent?: string;
    content_length?: number;
  };
  response?: {
    status_code: number;
    duration_ms: number;
    content_length?: number;
  };
  action?: {
    operation: string;
    resource_type: string;
    resource_id?: string;
    description?: string;
  };
  authorization?: {
    decision: string;
    required_permission?: string;
    evaluated_scopes?: string[];
  };
  // MCP-specific fields
  mcp_server?: {
    name: string;
    path: string;
    version?: string;
    proxy_target: string;
  };
  mcp_request?: {
    method: string;
    tool_name?: string;
    resource_uri?: string;
    mcp_session_id?: string;
    transport: string;
    jsonrpc_id?: string;
  };
  mcp_response?: {
    status: string;
    duration_ms: number;
    error_code?: number;
    error_message?: string;
  };
  // Token-mint-specific fields (token_mint stream has no `identity` block).
  // `username` is the raw human-readable identity (email -> preferred_username
  // -> upn -> sub); `username_hash` is DEPRECATED, kept only for old records.
  username?: string;
  username_hash?: string;
  token_kind?: string;
  token_path?: string;
  resource_type?: string;
  resource_id?: string;
  outcome?: string;
  failure_reason?: string;
  expires_in_seconds?: number;
  // The same durable / IdP identity claims declared under `identity` above:
  // token_mint records have no `identity` block, so they carry them flat.
  subject?: string | null;
  canonical_id?: string | null;
  principal_name?: string | null;
  object_id?: string | null;
  tenant_id?: string | null;
  app_id?: string | null;
}

export type AuditStream = 'registry_api' | 'mcp_access' | 'token_mint';
