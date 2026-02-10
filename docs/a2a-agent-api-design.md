# A2A Agent Management API

API design for A2A Agent management based on MCP Server management API patterns.

## API Route Prefix

```
/api/v1/agents
```

---

## API Endpoints

### 1. List Agents

**Endpoint**: `GET /api/v1/agents`

**Query Parameters**:
```typescript
{
  query?: string;           // Search keywords (name, description, tags, skills)
  status?: string;          // Status filter: active | inactive | error
  page?: number;            // Page number (default: 1)
  per_page?: number;        // Items per page (default: 20, max: 100)
}
```

**Response**: `200 OK`
```json
{
  "agents": [
    {
      "id": "507f1f77bcf86cd799439011",
      "path": "/code-reviewer",
      "name": "Code Review Agent",
      "description": "AI-powered code review assistant",
      "url": "https://example.com/agents/code-reviewer",
      "version": "1.0.0",
      "protocolVersion": "1.0",
      "tags": ["code", "review"],
      "numSkills": 5,
      "isEnabled": true,
      "status": "active",
      "aclPermission": 15,
      "registeredBy": "user123",
      "registeredAt": "2024-01-15T10:30:00Z",
      "updatedAt": "2024-01-20T15:45:00Z"
    }
  ],
  "pagination": {
    "total": 150,
    "page": 1,
    "per_page": 20,
    "total_pages": 8
  }
}
```

---

### 2. Get Agent Statistics

**Endpoint**: `GET /api/v1/agents/stats`

**Response**: `200 OK`
```json
{
  "totalAgents": 150,
  "enabledAgents": 120,
  "disabledAgents": 30,
  "byStatus": {
    "active": 130,
    "inactive": 15,
    "error": 5
  },
  "byTransport": {
    "HTTP+JSON": 100,
    "JSONRPC": 30,
    "GRPC": 20
  },
  "totalSkills": 450,
  "averageSkillsPerAgent": 3.0
}
```

**Permission**: Admin only

---

### 3. Get Agent Detail

**Endpoint**: `GET /api/v1/agents/{agent_id}`

**Response**: `200 OK`
```json
{
  "id": "507f1f77bcf86cd799439011",
  "path": "/code-reviewer",
  "name": "Code Review Agent",
  "description": "AI-powered code review assistant",
  "url": "https://example.com/agents/code-reviewer",
  "version": "1.0.0",
  "protocolVersion": "1.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "code-analysis",
      "name": "Code Analysis",
      "description": "Analyze code quality",
      "tags": ["analysis"],
      "inputModes": ["text/plain"],
      "outputModes": ["application/json"]
    }
  ],
  "securitySchemes": {
    "bearer": {
      "type": "http",
      "scheme": "bearer"
    }
  },
  "preferredTransport": "HTTP+JSON",
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["application/json"],
  "provider": {
    "organization": "AI Labs",
    "url": "https://ailabs.com"
  },
  "tags": ["code", "review"],
  "status": "active",
  "isEnabled": true,
  "aclPermission": 15,
  "author": "507f1f77bcf86cd799439012",
  "wellKnown": {
    "enabled": true,
    "url": "https://example.com/.well-known/agent-card.json",
    "lastSyncAt": "2024-01-20T12:00:00Z",
    "lastSyncStatus": "success",
    "lastSyncVersion": "1.0.0"
  },
  "registeredBy": "user123",
  "registeredAt": "2024-01-15T10:30:00Z",
  "createdAt": "2024-01-15T10:30:00Z",
  "updatedAt": "2024-01-20T15:45:00Z"
}
```

**Error**: `404` Agent not found, `403` Access denied

---

### 4. Create Agent

**Endpoint**: `POST /api/v1/agents`

**Request Body**:
```json
{
  "path": "/code-reviewer",
  "name": "Code Review Agent",
  "description": "AI-powered code review assistant",
  "url": "https://example.com/agents/code-reviewer",
  "version": "1.0.0",
  "protocolVersion": "1.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "code-analysis",
      "name": "Code Analysis",
      "description": "Analyze code quality",
      "tags": ["analysis"],
      "inputModes": ["text/plain"],
      "outputModes": ["application/json"]
    }
  ],
  "securitySchemes": {},
  "preferredTransport": "HTTP+JSON",
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["application/json"],
  "provider": {
    "organization": "AI Labs",
    "url": "https://ailabs.com"
  },
  "tags": ["code", "review"],
  "isEnabled": false
}
```

**Response**: `201 Created`
```json
{
  "message": "Agent registered successfully",
  "agent": {
    "id": "507f1f77bcf86cd799439011",
    "path": "/code-reviewer",
    "name": "Code Review Agent",
    "url": "https://example.com/agents/code-reviewer",
    "registeredAt": "2024-01-15T10:30:00Z"
  }
}
```

**Error**: `400` Validation error, `409` Path already exists

**Note**: Automatically grants OWNER permission to creator when creating Agent, ACL resource type is `ResourceType.A2AAGENT`

---

### 5. Update Agent

**Endpoint**: `PATCH /api/v1/agents/{agent_id}`

**Request Body** (all fields optional):
```json
{
  "name": "Updated Agent Name",
  "description": "Updated description",
  "version": "1.1.0",
  "skills": [...],
  "tags": ["new", "tags"],
  "isEnabled": true
}
```

**Response**: `200 OK`
```json
{
  "message": "Agent updated successfully",
  "agent": {
    "id": "507f1f77bcf86cd799439011",
    "path": "/code-reviewer",
    "name": "Updated Agent Name",
    "updatedAt": "2024-01-20T15:45:00Z"
  }
}
```

**Error**: `404` Agent not found, `403` Access denied

**Permission**: Requires EDIT permission

---

### 6. Delete Agent

**Endpoint**: `DELETE /api/v1/agents/{agent_id}`

**Response**: `204 No Content`

**Error**: `404` Agent not found, `403` Access denied

**Permission**: Requires DELETE permission

**Note**: Deletes all associated ACL permission records when deleting Agent

---

### 7. Toggle Agent

**Endpoint**: `POST /api/v1/agents/{agent_id}/toggle`

**Request Body**:
```json
{
  "enabled": true
}
```

**Response**: `200 OK`
```json
{
  "message": "Agent enabled successfully",
  "agent": {
    "id": "507f1f77bcf86cd799439011",
    "path": "/code-reviewer",
    "isEnabled": true
  }
}
```

**Error**: `404` Agent not found, `403` Access denied

**Permission**: Requires EDIT permission

---

### 8. Get Agent Skills

**Endpoint**: `GET /api/v1/agents/{agent_id}/skills`

**Response**: `200 OK`
```json
{
  "agentId": "507f1f77bcf86cd799439011",
  "agentName": "Code Review Agent",
  "skills": [
    {
      "id": "code-analysis",
      "name": "Code Analysis",
      "description": "Analyze code quality",
      "tags": ["analysis", "quality"],
      "inputModes": ["text/plain"],
      "outputModes": ["application/json"]
    }
  ],
  "totalSkills": 1
}
```

**Error**: `404` Agent not found, `403` Access denied

**Permission**: Requires VIEW permission

---

### 9. Sync Well-Known

**Endpoint**: `POST /api/v1/agents/{agent_id}/sync-wellknown`

**Response**: `200 OK`
```json
{
  "message": "Well-known configuration synced successfully",
  "syncStatus": "success",
  "syncedAt": "2024-01-20T15:45:00Z",
  "version": "1.0.0",
  "changes": [
    "Updated version to 1.0.0",
    "Added 2 new skills"
  ]
}
```

**Error**: `400` Well-known sync not enabled, `404` Agent not found or URL unreachable

**Permission**: Requires EDIT permission

---