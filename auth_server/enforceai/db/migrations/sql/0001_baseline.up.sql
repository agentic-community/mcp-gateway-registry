CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    scopes_json TEXT NOT NULL,
    allowed_tools_json TEXT NULL,
    alias TEXT NULL,
    metadata_json TEXT NULL,
    revoked_at TEXT NULL,
    tokens_valid_after TEXT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agents_user_id ON agents(user_id);
CREATE INDEX IF NOT EXISTS idx_agents_user_id_revoked_at ON agents(user_id, revoked_at);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    secret_hash TEXT NOT NULL,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    scopes_json TEXT NULL,
    expires_at TEXT NULL,
    revoked_at TEXT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_agent_id ON api_keys(agent_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at);

CREATE TABLE IF NOT EXISTS token_revocations (
    jti TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    revoked_at TEXT NOT NULL,
    expires_at TEXT NULL,
    reason TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_revocations_agent_id ON token_revocations(agent_id);
CREATE INDEX IF NOT EXISTS idx_token_revocations_expires_at ON token_revocations(expires_at);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    action TEXT NOT NULL,
    outcome TEXT NOT NULL,
    request_id TEXT NULL,
    details_json TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_events_occurred_at ON audit_events(occurred_at);
CREATE INDEX IF NOT EXISTS idx_audit_events_user_id_occurred_at ON audit_events(user_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_audit_events_agent_id_occurred_at ON audit_events(agent_id, occurred_at);

