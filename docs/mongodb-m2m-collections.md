# MongoDB Collections for M2M Accounts

## Overview

M2M accounts are stored in **THREE** MongoDB collections with different purposes:

```
┌─────────────────────────────────────────────────────┐
│         M2M Account Storage Architecture            │
└─────────────────────────────────────────────────────┘

1. idp_m2m_clients          ← PRIMARY (used by auth-server)
   ├─ All providers: Keycloak, Okta, Entra, Auth0
   ├─ Purpose: Groups enrichment during authentication
   └─ Used by: auth_server/mongodb_groups_enrichment.py

2. okta_m2m_clients         ← Okta-specific metadata
   ├─ Only Okta M2M clients
   ├─ Purpose: Okta sync tracking
   └─ Used by: registry/services/okta_m2m_sync.py

3. auth0_m2m_clients        ← Auth0-specific metadata
   ├─ Only Auth0 M2M clients
   ├─ Purpose: Auth0 sync tracking
   └─ Used by: registry/services/auth0_m2m_sync.py
```

---

## Collection Details

### 1. `idp_m2m_clients` (PRIMARY - Generic)

**Purpose:** Provider-agnostic collection for ALL M2M clients
**Used by:** Auth-server for groups enrichment
**Scope:** All IdP providers (Keycloak, Okta, Entra, Auth0)

**Schema:**
```javascript
{
  "_id": ObjectId("..."),
  "client_id": "KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd",
  "name": "MCP Gateway M2M",
  "description": "M2M client for registry access",
  "groups": ["registry-admins", "developers"],
  "enabled": true,
  "provider": "auth0",  // or "okta", "keycloak", "entra"
  "idp_app_id": "KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd",
  "created_at": ISODate("2026-03-29T00:00:00Z"),
  "updated_at": ISODate("2026-03-29T00:00:00Z")
}
```

**How it's used:**
1. M2M token arrives with empty `groups: []`
2. Auth-server validates JWT
3. Queries: `db.idp_m2m_clients.find_one({client_id: "..."})`
4. Returns groups: `["registry-admins"]`
5. Token is enriched with groups for authorization

**Created by:**
- Manual: `POST /api/iam/users/m2m` (management API)
- Auto-sync: `POST /api/iam/okta/m2m/sync` (Okta)
- Auto-sync: `POST /api/iam/auth0/m2m/sync` (Auth0)

**Updated by:**
- `PATCH /api/iam/users/{username}/groups`
- `PATCH /api/iam/okta/m2m/clients/{id}/groups`
- `PATCH /api/iam/auth0/m2m/clients/{id}/groups`

---

### 2. `okta_m2m_clients` (Okta-specific)

**Purpose:** Okta-specific M2M client metadata
**Used by:** Okta sync service
**Scope:** Only Okta M2M clients

**Schema:**
```javascript
{
  "_id": ObjectId("..."),
  "client_id": "0oa1100req1AzfKaY698",
  "name": "ai-agent",
  "description": "AI agent with admin access",
  "groups": ["registry-admins"],
  "enabled": true,
  "okta_app_id": "0oa1100req1AzfKaY698",
  "last_synced": ISODate("2026-03-29T00:00:00Z"),
  "created_at": ISODate("2026-03-29T00:00:00Z"),
  "updated_at": ISODate("2026-03-29T00:00:00Z")
}
```

**How it's used:**
- Sync service fetches Okta apps with `grant_type: client_credentials`
- Stores in `okta_m2m_clients` for tracking
- **ALSO** writes to `idp_m2m_clients` for auth enrichment

**Operations:**
- `GET /api/iam/okta/m2m/clients` - Lists from this collection
- `POST /api/iam/okta/m2m/sync` - Syncs to this collection

---

### 3. `auth0_m2m_clients` (Auth0-specific)

**Purpose:** Auth0-specific M2M client metadata
**Used by:** Auth0 sync service
**Scope:** Only Auth0 M2M clients

**Schema:**
```javascript
{
  "_id": ObjectId("..."),
  "client_id": "KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd",
  "name": "MCP Gateway M2M",
  "description": "M2M client for registry access",
  "groups": ["registry-admins"],
  "enabled": true,
  "auth0_client_id": "KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd",
  "app_type": "non_interactive",
  "last_synced": ISODate("2026-03-29T00:00:00Z"),
  "created_at": ISODate("2026-03-29T00:00:00Z"),
  "updated_at": ISODate("2026-03-29T00:00:00Z")
}
```

**How it's used:**
- Sync service fetches Auth0 apps with `app_type: non_interactive`
- Stores in `auth0_m2m_clients` for tracking
- **ALSO** writes to `idp_m2m_clients` for auth enrichment

**Operations:**
- `GET /api/iam/auth0/m2m/clients` - Lists from this collection
- `POST /api/iam/auth0/m2m/sync` - Syncs to this collection

---

## Data Flow

### Creating an M2M Account

#### Option 1: Manual Creation (All Providers)
```
POST /api/iam/users/m2m
  ↓
Creates in IdP (Keycloak/Okta/Entra/Auth0)
  ↓
Writes to: idp_m2m_clients ✓
```

#### Option 2: Okta Auto-Sync
```
POST /api/iam/okta/m2m/sync
  ↓
Fetches from Okta API
  ↓
Writes to: okta_m2m_clients ✓
  ↓
Writes to: idp_m2m_clients ✓
```

#### Option 3: Auth0 Auto-Sync
```
POST /api/iam/auth0/m2m/sync
  ↓
Fetches from Auth0 API
  ↓
Writes to: auth0_m2m_clients ✓
  ↓
Writes to: idp_m2m_clients ✓
```

---

## Authentication Flow

```
1. M2M Token arrives (groups: [])
   ├─ provider: okta/auth0/keycloak/entra
   ├─ client_id: "abc123..."
   └─ groups: [] (empty)

2. Auth-server validates JWT
   └─ auth_server/providers/{provider}.py

3. Groups enrichment triggered
   └─ mongodb_groups_enrichment.py
      └─ Queries: db.idp_m2m_clients.find_one({client_id})

4. Groups found
   └─ Returns: ["registry-admins"]

5. Authorization succeeds
   └─ Token enriched with groups
```

---

## Query Examples

### List ALL M2M accounts (all providers)
```javascript
db.idp_m2m_clients.find().pretty()
```

### List by provider
```javascript
// Auth0 M2M clients
db.idp_m2m_clients.find({ provider: "auth0" }).pretty()

// Okta M2M clients
db.idp_m2m_clients.find({ provider: "okta" }).pretty()
```

### Find specific M2M client
```javascript
db.idp_m2m_clients.findOne({ client_id: "KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd" })
```

### Check groups for client
```javascript
db.idp_m2m_clients.findOne(
  { client_id: "abc123..." },
  { groups: 1, name: 1, provider: 1 }
)
```

### Update groups manually
```javascript
db.idp_m2m_clients.updateOne(
  { client_id: "abc123..." },
  {
    $set: {
      groups: ["registry-admins", "developers"],
      updated_at: new Date()
    }
  }
)
```

---

## Key Points

### ✅ Every M2M account MUST be in `idp_m2m_clients`
This is the **ONLY** collection that auth-server queries for groups enrichment.

### ✅ Provider-specific collections are optional
`okta_m2m_clients` and `auth0_m2m_clients` are for tracking sync metadata.

### ✅ Dual-write pattern
When syncing, both collections are updated:
- Provider-specific collection (okta/auth0)
- Generic `idp_m2m_clients` collection

### ✅ Groups enrichment is automatic
Auth-server automatically queries `idp_m2m_clients` when token has empty groups.

---

## Summary Table

| Collection | Providers | Used By | Purpose |
|------------|-----------|---------|---------|
| `idp_m2m_clients` | All (Keycloak, Okta, Entra, Auth0) | Auth-server | Groups enrichment |
| `okta_m2m_clients` | Okta only | Okta sync service | Sync tracking |
| `auth0_m2m_clients` | Auth0 only | Auth0 sync service | Sync tracking |

**Bottom line:** All M2M accounts are listed in `idp_m2m_clients` regardless of provider.
