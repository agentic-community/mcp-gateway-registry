CREATE TABLE IF NOT EXISTS upstream_credentials (
    credential_id TEXT PRIMARY KEY,
    server_path TEXT NOT NULL,
    credential_type TEXT NOT NULL,
    credential_binding TEXT NOT NULL,
    user_id TEXT NULL,
    agent_id TEXT NULL,
    provider TEXT NULL,
    scopes_json TEXT NULL,
    token_type TEXT NULL,
    secret_version INTEGER NULL,
    secret_nonce BLOB NULL,
    secret_ciphertext BLOB NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT NULL,
    revoked_at TEXT NULL,
    last_used_at TEXT NULL,
    CHECK (credential_type IN ('api-key','oauth2','oidc','provider-oauth','jwt','mtls','header-trust')),
    CHECK (credential_binding IN ('service','user','agent','user+agent')),
    CHECK (
        (credential_binding = 'service' AND user_id IS NULL AND agent_id IS NULL) OR
        (credential_binding = 'user' AND user_id IS NOT NULL AND agent_id IS NULL) OR
        (credential_binding = 'agent' AND user_id IS NULL AND agent_id IS NOT NULL) OR
        (credential_binding = 'user+agent' AND user_id IS NOT NULL AND agent_id IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_upstream_credentials_server_path
    ON upstream_credentials(server_path);
CREATE INDEX IF NOT EXISTS idx_upstream_credentials_user_id
    ON upstream_credentials(user_id);
CREATE INDEX IF NOT EXISTS idx_upstream_credentials_agent_id
    ON upstream_credentials(agent_id);
CREATE INDEX IF NOT EXISTS idx_upstream_credentials_revoked_at
    ON upstream_credentials(revoked_at);
CREATE INDEX IF NOT EXISTS idx_upstream_credentials_expires_at
    ON upstream_credentials(expires_at);

