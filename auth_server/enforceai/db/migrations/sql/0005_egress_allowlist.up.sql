CREATE TABLE IF NOT EXISTS egress_allowlist_entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    comment TEXT NULL,
    expires_at TEXT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (kind IN ('hostname','domain-suffix','ip-cidr'))
);

CREATE INDEX IF NOT EXISTS idx_egress_allowlist_entries_kind
    ON egress_allowlist_entries(kind);
CREATE INDEX IF NOT EXISTS idx_egress_allowlist_entries_expires_at
    ON egress_allowlist_entries(expires_at);

