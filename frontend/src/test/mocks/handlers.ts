import { http, HttpResponse } from 'msw';
import type { Server, A2AAgent, EnforceAIAgent, ServerDetails, Tool, ApiKeySummary, ScopeCatalog, EgressAllowlistEntry, UpstreamOAuthCredential } from '@/api/types';

// Mock tools data
export const mockTools: Tool[] = [
  {
    name: 'read_query',
    description: 'Execute a SELECT query on the database',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'The SQL SELECT query' },
      },
      required: ['query'],
    },
  },
  {
    name: 'list_tables',
    description: 'List all tables in the database',
    input_schema: {},
  },
  {
    name: 'describe_table',
    description: 'Get the schema of a table',
    input_schema: {
      type: 'object',
      properties: {
        table_name: { type: 'string', description: 'Name of the table' },
      },
      required: ['table_name'],
    },
  },
];

// Mock data for testing
export const mockServers: Server[] = [
  {
    display_name: 'SQLite Server',
    path: 'sqlite',
    proxy_pass_url: 'http://localhost:3031/mcp',
    description: 'SQLite MCP Server',
    tags: ['database', 'sql'],
    is_enabled: true,
    health_status: 'healthy',
    num_tools: 6,
    upstream_auth: {
      mode: 'gateway-managed',
      type: 'api-key',
      credential_binding: 'user',
    },
    upstream_credential_status: 'configured',
  },
  {
    display_name: 'Filesystem Server',
    path: 'filesystem',
    proxy_pass_url: 'http://localhost:3032/mcp',
    description: 'Filesystem MCP Server',
    tags: ['files', 'io'],
    is_enabled: true,
    health_status: 'healthy',
    num_tools: 4,
    upstream_auth: {
      mode: 'gateway-managed',
      type: 'oauth2',
      provider: 'github',
      credential_binding: 'user',
    },
    upstream_credential_status: 'missing',
  },
  {
    display_name: 'Disabled Server',
    path: 'disabled',
    proxy_pass_url: 'http://localhost:3033/mcp',
    description: 'A disabled server',
    is_enabled: false,
    health_status: 'unknown',
    num_tools: 0,
    upstream_auth: {
      mode: 'gateway-managed',
      type: 'jwt',
      credential_binding: 'service',
    },
    upstream_credential_status: 'expired',
  },
  {
    display_name: 'GitHub MCP',
    path: 'github-mcp',
    proxy_pass_url: 'http://localhost:3034/mcp',
    description: 'GitHub API MCP Server',
    tags: ['github', 'vcs'],
    is_enabled: true,
    health_status: 'healthy',
    num_tools: 10,
    upstream_auth: {
      mode: 'gateway-managed',
      type: 'provider-oauth',
      provider: 'github',
      credential_binding: 'user',
    },
    upstream_credential_status: 'configured',
  },
  {
    display_name: 'Slack MCP',
    path: 'slack-mcp',
    proxy_pass_url: 'http://localhost:3035/mcp',
    description: 'Slack API MCP Server',
    tags: ['slack', 'messaging'],
    is_enabled: true,
    health_status: 'healthy',
    num_tools: 8,
    upstream_auth: {
      mode: 'gateway-managed',
      type: 'oauth2',
      provider: 'slack',
      credential_binding: 'user',
    },
    upstream_credential_status: 'missing',
  },
];

// Mock server details (with tools)
export const mockServerDetails: ServerDetails[] = mockServers.map((server) => ({
  ...server,
  tools: server.path === 'sqlite' ? mockTools : [],
  supported_transports: ['http', 'sse'],
}));

export const mockA2AAgents: A2AAgent[] = [
  {
    name: 'Code Assistant',
    path: 'code-assistant',
    url: 'http://localhost:5001',
    description: 'AI-powered code assistant',
    skills: ['code-review', 'refactoring'],
    tags: ['ai', 'code'],
    num_skills: 2,
    num_stars: 5,
    visibility: 'public',
    is_enabled: true,
    health_status: 'healthy',
  },
  {
    name: 'Data Analyst',
    path: 'data-analyst',
    url: 'http://localhost:5002',
    description: 'Data analysis agent',
    skills: ['data-analysis', 'visualization'],
    tags: ['ai', 'data'],
    num_skills: 2,
    num_stars: 3,
    visibility: 'private',
    is_enabled: false,
    health_status: 'unknown',
  },
];

export const mockEnforceAIAgents: EnforceAIAgent[] = [
  {
    user_id: 'test|user123',
    agent_id: '9d2724e9-1753-4493-8993-0d6986754414',
    scopes: ['sqlite.manage', 'filesystem.read'],
    allowed_tools: ['read_query', 'list_tables'],
    alias: 'sqlite-agent',
    metadata: null,
    revoked_at: null,
    tokens_valid_after: null,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-15T00:00:00Z',
  },
  {
    user_id: 'test|user123',
    agent_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
    scopes: ['filesystem.read'],
    allowed_tools: null,
    alias: 'fs-reader',
    metadata: null,
    revoked_at: '2024-01-10T00:00:00Z', // Revoked agent
    tokens_valid_after: null,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-10T00:00:00Z',
  },
];

export const mockApiKeys: ApiKeySummary[] = [
  {
    key_id: 'eak_abc123',
    user_id: 'test|user123',
    agent_id: '9d2724e9-1753-4493-8993-0d6986754414',
    scopes: ['sqlite.manage'],
    expires_at: '2026-01-01T00:00:00Z',
    revoked_at: null,
    created_at: '2024-01-05T00:00:00Z',
    last_used_at: '2024-01-15T00:00:00Z',
  },
  {
    key_id: 'eak_def456',
    user_id: 'test|user123',
    agent_id: '9d2724e9-1753-4493-8993-0d6986754414',
    scopes: null,
    expires_at: null,
    revoked_at: '2024-01-12T00:00:00Z',
    created_at: '2024-01-02T00:00:00Z',
    last_used_at: null,
  },
];

export const mockEgressAllowlistEntries: EgressAllowlistEntry[] = [
  {
    entry_id: 'entry-1',
    pattern: 'localhost:*',
    description: 'Allow all localhost connections for development',
    expires_at: null,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  {
    entry_id: 'entry-2',
    pattern: '*.example.com',
    description: 'Allow all example.com subdomains',
    expires_at: '2025-12-31T23:59:59Z',
    created_at: '2024-01-05T00:00:00Z',
    updated_at: '2024-01-05T00:00:00Z',
  },
  {
    entry_id: 'entry-3',
    pattern: 'api.trusted-service.io',
    description: 'Production API endpoint',
    expires_at: null,
    created_at: '2024-01-10T00:00:00Z',
    updated_at: '2024-01-15T00:00:00Z',
  },
];

export const mockScopeCatalog: ScopeCatalog = {
  version: '1.0',
  generated_at: new Date().toISOString(),
  group_mappings: {
    'registry-admins': ['registry-admins', 'mcp-servers-unrestricted/read'],
    'registry-users-lob1': ['registry-users-lob1'],
  },
  scopes: {
    'mcp-servers-unrestricted/read': {
      name: 'mcp-servers-unrestricted/read',
      server_permissions: [
        {
          server: '*',
          methods: {
            all_methods: false,
            methods: ['initialize', 'tools/list', 'tools/call', 'GET'],
          },
          tools: {
            all_tools: true,
            tools: [],
          },
        },
      ],
      agent_permissions: [],
    },
    'registry-admins': {
      name: 'registry-admins',
      server_permissions: [
        {
          server: '*',
          methods: {
            all_methods: true,
            methods: [],
          },
          tools: {
            all_tools: true,
            tools: [],
          },
        },
      ],
      agent_permissions: [
        { action: 'list_agents', resources: ['all'] },
        { action: 'get_agent', resources: ['all'] },
        { action: 'publish_agent', resources: ['all'] },
        { action: 'modify_agent', resources: ['all'] },
        { action: 'delete_agent', resources: ['all'] },
      ],
    },
    'registry-users-lob1': {
      name: 'registry-users-lob1',
      server_permissions: [
        {
          server: 'api',
          methods: {
            all_methods: false,
            methods: ['initialize', 'GET'],
          },
          tools: null,
        },
        {
          server: 'sqlite',
          methods: {
            all_methods: false,
            methods: ['initialize', 'tools/list', 'tools/call'],
          },
          tools: {
            all_tools: false,
            tools: ['read_query', 'list_tables'],
          },
        },
      ],
      agent_permissions: [
        { action: 'list_agents', resources: ['/code-reviewer', '/test-automation'] },
        { action: 'get_agent', resources: ['/code-reviewer', '/test-automation'] },
      ],
    },
    'sqlite.manage': {
      name: 'sqlite.manage',
      server_permissions: [
        {
          server: 'sqlite',
          methods: {
            all_methods: false,
            methods: ['initialize', 'tools/list', 'tools/call'],
          },
          tools: {
            all_tools: true,
            tools: [],
          },
        },
      ],
      agent_permissions: [],
    },
  },
};

// Default mock handlers for API endpoints
export const handlers = [
  // Auth endpoints
  http.get('/api/auth/me', () => {
    return HttpResponse.json({
      user_id: 'test|user123',
      email: 'test@example.com',
      username: 'testuser',
      auth_method: 'password',
      is_admin: false,
      roles: [],
      groups: [],
    });
  }),

  http.get('/api/auth/csrf', () => {
    return HttpResponse.json({
      csrf_token: 'test-csrf-token-12345',
    });
  }),

  http.post('/api/auth/login', async ({ request }) => {
    const body = (await request.json()) as {
      username?: string;
      password?: string;
    };
    if (body.username === 'testuser' && body.password === 'testpass') {
      return HttpResponse.json({ success: true });
    }
    return HttpResponse.json({ detail: 'Invalid credentials' }, { status: 401 });
  }),

  http.post('/api/auth/logout', () => {
    return HttpResponse.json({ success: true });
  }),

  http.get('/api/auth/providers', () => {
    return HttpResponse.json({
      providers: [
        { name: 'google', display_name: 'Google' },
        { name: 'github', display_name: 'GitHub' },
      ],
    });
  }),

  http.post('/api/auth/refresh', () => {
    return HttpResponse.json({ success: true });
  }),

  // Registry: Servers endpoints
  http.get('/api/servers', () => {
    return HttpResponse.json({ servers: mockServers });
  }),

  http.get('/api/servers/:path', ({ params }) => {
    const server = mockServerDetails.find((s) => s.path === params.path);
    if (server) {
      return HttpResponse.json(server);
    }
    return HttpResponse.json({ detail: 'Server not found' }, { status: 404 });
  }),

  http.post('/api/servers', async ({ request }) => {
    const body = (await request.json()) as {
      name: string;
      path: string;
      proxy_pass_url: string;
      description?: string;
      tags?: string[];
      upstream_auth?: Server['upstream_auth'];
    };
    const newServer: Server = {
      display_name: body.name,
      path: body.path,
      proxy_pass_url: body.proxy_pass_url,
      description: body.description,
      tags: body.tags,
      is_enabled: true,
      health_status: 'unknown',
      num_tools: 0,
      upstream_auth: body.upstream_auth,
    };
    return HttpResponse.json(newServer, { status: 201 });
  }),

  http.put('/api/servers/:path', async ({ params, request }) => {
    const server = mockServers.find((s) => s.path === params.path);
    if (!server) {
      return HttpResponse.json({ detail: 'Server not found' }, { status: 404 });
    }
    const body = (await request.json()) as {
      name?: string;
      proxy_pass_url?: string;
      description?: string;
      tags?: string[];
      enabled?: boolean;
      upstream_auth?: Server['upstream_auth'];
    };
    const updatedServer: Server = {
      ...server,
      display_name: body.name ?? server.display_name,
      proxy_pass_url: body.proxy_pass_url ?? server.proxy_pass_url,
      description: body.description ?? server.description,
      tags: body.tags ?? server.tags,
      is_enabled: body.enabled !== undefined ? body.enabled : server.is_enabled,
      upstream_auth: body.upstream_auth ?? server.upstream_auth,
    };
    return HttpResponse.json(updatedServer);
  }),

  http.delete('/api/servers/:path', ({ params }) => {
    const server = mockServers.find((s) => s.path === params.path);
    if (!server) {
      return HttpResponse.json({ detail: 'Server not found' }, { status: 404 });
    }
    return new HttpResponse(null, { status: 204 });
  }),

  http.post('/api/servers/:path/refresh', ({ params }) => {
    const server = mockServers.find((s) => s.path === params.path);
    if (server) {
      return HttpResponse.json({ ...server, health_status: 'healthy' });
    }
    return HttpResponse.json({ detail: 'Server not found' }, { status: 404 });
  }),

  // Registry: A2A Agents endpoints
  http.get('/api/agents', () => {
    return HttpResponse.json({
      agents: mockA2AAgents,
      total_count: mockA2AAgents.length,
    });
  }),

  http.get('/api/agents/:path', ({ params }) => {
    const agent = mockA2AAgents.find((a) => a.path === params.path);
    if (agent) {
      return HttpResponse.json(agent);
    }
    return HttpResponse.json({ detail: 'Agent not found' }, { status: 404 });
  }),

  http.post('/api/agents', async ({ request }) => {
    const body = (await request.json()) as {
      name: string;
      path: string;
      description?: string;
      skills?: string[];
      tags?: string[];
      visibility?: 'public' | 'private';
    };
    const newAgent: A2AAgent = {
      name: body.name,
      path: body.path,
      url: `http://localhost:${5000 + Math.floor(Math.random() * 1000)}`,
      description: body.description,
      skills: body.skills ?? [],
      tags: body.tags ?? [],
      num_skills: body.skills?.length ?? 0,
      num_stars: 0,
      visibility: body.visibility ?? 'public',
      is_enabled: true,
      health_status: 'unknown',
    };
    return HttpResponse.json(newAgent, { status: 201 });
  }),

  http.put('/api/agents/:path', async ({ params, request }) => {
    const agent = mockA2AAgents.find((a) => a.path === params.path);
    if (!agent) {
      return HttpResponse.json({ detail: 'Agent not found' }, { status: 404 });
    }
    const body = (await request.json()) as {
      name?: string;
      description?: string;
      skills?: string[];
      tags?: string[];
      visibility?: 'public' | 'private';
      enabled?: boolean;
    };
    const updatedAgent: A2AAgent = {
      ...agent,
      name: body.name ?? agent.name,
      description: body.description ?? agent.description,
      skills: body.skills ?? agent.skills,
      tags: body.tags ?? agent.tags,
      visibility: body.visibility ?? agent.visibility,
      is_enabled: body.enabled !== undefined ? body.enabled : agent.is_enabled,
      num_skills: body.skills?.length ?? agent.num_skills,
    };
    return HttpResponse.json(updatedAgent);
  }),

  http.delete('/api/agents/:path', ({ params }) => {
    const agent = mockA2AAgents.find((a) => a.path === params.path);
    if (!agent) {
      return HttpResponse.json({ detail: 'Agent not found' }, { status: 404 });
    }
    return new HttpResponse(null, { status: 204 });
  }),

  // EnforceAI: Agents endpoints
  http.get('/enforceai/agents', () => {
    return HttpResponse.json(mockEnforceAIAgents);
  }),

  http.get('/enforceai/agents/:agentId', ({ params }) => {
    const agent = mockEnforceAIAgents.find((a) => a.agent_id === params.agentId);
    if (agent) {
      return HttpResponse.json(agent);
    }
    return HttpResponse.json({ detail: 'Agent not found' }, { status: 404 });
  }),

  http.post('/enforceai/agents', async ({ request }) => {
    const body = (await request.json()) as {
      scopes: string[];
      alias?: string;
      allowed_tools?: string[] | null;
      metadata?: Record<string, unknown> | null;
    };
    const newAgent: EnforceAIAgent = {
      user_id: 'test|user123',
      agent_id: 'new-agent-' + Date.now(),
      scopes: body.scopes,
      allowed_tools: body.allowed_tools ?? null,
      alias: body.alias ?? null,
      metadata: body.metadata ?? null,
      revoked_at: null,
      tokens_valid_after: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    return HttpResponse.json(newAgent, { status: 201 });
  }),

  http.put('/enforceai/agents/:agentId', async ({ params, request }) => {
    const agent = mockEnforceAIAgents.find((a) => a.agent_id === params.agentId);
    if (!agent) {
      return HttpResponse.json({ detail: 'Agent not found' }, { status: 404 });
    }
    const body = (await request.json()) as {
      scopes?: string[] | null;
      alias?: string | null;
      allowed_tools?: string[] | null;
      metadata?: Record<string, unknown> | null;
    };
    const updatedAgent: EnforceAIAgent = {
      ...agent,
      scopes: body.scopes ?? agent.scopes,
      alias: body.alias !== undefined ? body.alias : agent.alias,
      allowed_tools: body.allowed_tools !== undefined ? body.allowed_tools : agent.allowed_tools,
      metadata: body.metadata !== undefined ? body.metadata : agent.metadata,
      updated_at: new Date().toISOString(),
    };
    return HttpResponse.json(updatedAgent);
  }),

  http.post('/enforceai/agents/:agentId/revoke', ({ params }) => {
    const agent = mockEnforceAIAgents.find((a) => a.agent_id === params.agentId);
    if (!agent) {
      return HttpResponse.json({ detail: 'Agent not found' }, { status: 404 });
    }
    const revokedAgent: EnforceAIAgent = {
      ...agent,
      revoked_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    return HttpResponse.json(revokedAgent);
  }),

  http.post('/enforceai/agents/:agentId/revoke-all-tokens', ({ params }) => {
    const agent = mockEnforceAIAgents.find((a) => a.agent_id === params.agentId);
    if (!agent) {
      return HttpResponse.json({ detail: 'Agent not found' }, { status: 404 });
    }
    const updatedAgent: EnforceAIAgent = {
      ...agent,
      tokens_valid_after: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    return HttpResponse.json(updatedAgent);
  }),

  // EnforceAI: API Keys endpoints
  http.get('/enforceai/agents/:agentId/api-keys', ({ params }) => {
    const keys = mockApiKeys.filter((k) => k.agent_id === params.agentId);
    return HttpResponse.json(keys);
  }),

  http.post('/enforceai/agents/:agentId/api-keys', async ({ params, request }) => {
    const body = (await request.json()) as {
      scopes?: string[] | null;
      expires_at?: string | null;
    };
    const keyId = 'eak_' + Date.now();
    return HttpResponse.json({
      key_id: keyId,
      secret: 'test-secret-value-' + Date.now(),
      api_key_value: `${keyId}.testsecretvalue`,
    });
  }),

  http.post('/enforceai/api-keys/:keyId/revoke', ({ params }) => {
    const key = mockApiKeys.find((k) => k.key_id === params.keyId);
    if (!key) {
      return HttpResponse.json({ detail: 'API key not found' }, { status: 404 });
    }
    return new HttpResponse(null, { status: 204 });
  }),

  // EnforceAI: Tokens endpoints
  http.post('/enforceai/agents/:agentId/tokens/mint', async ({ params, request }) => {
    const body = (await request.json()) as {
      scopes: string[];
      ttl_seconds?: number | null;
      expires_at?: string | null;
    };
    // Generate a mock JWT token with realistic structure
    const header = btoa(JSON.stringify({ alg: 'RS256', typ: 'JWT', kid: 'kid-local-1' }));
    const now = Math.floor(Date.now() / 1000);
    const exp = body.ttl_seconds
      ? now + body.ttl_seconds
      : body.expires_at
        ? Math.floor(new Date(body.expires_at).getTime() / 1000)
        : now + 86400;
    const payload = btoa(JSON.stringify({
      iss: 'enforceai-gateway',
      sub: params.agentId,
      aud: ['mcp-gateway'],
      exp,
      iat: now,
      jti: 'jti-' + Date.now(),
      type: 'agent',
      name: 'test-agent',
      scopes: body.scopes,
    }));
    const signature = 'mocksignature' + Date.now();
    return HttpResponse.json({
      token: `${header}.${payload}.${signature}`,
    });
  }),

  http.post('/enforceai/tokens/revoke', async ({ request }) => {
    const body = (await request.json()) as {
      gateway_token?: string;
      agent_id?: string;
      jti?: string;
      reason?: string;
    };
    // Extract JTI from token if provided
    let jti = body.jti;
    let agentId = body.agent_id;
    if (body.gateway_token && !jti) {
      try {
        const parts = body.gateway_token.split('.');
        if (parts.length === 3) {
          const payload = JSON.parse(atob(parts[1]));
          jti = payload.jti;
          agentId = payload.sub;
        }
      } catch {
        // Ignore decode errors
      }
    }
    return HttpResponse.json({
      jti: jti ?? 'revoked-jti-' + Date.now(),
      user_id: 'test|user123',
      agent_id: agentId ?? '9d2724e9-1753-4493-8993-0d6986754414',
      revoked_at: new Date().toISOString(),
      expires_at: null,
      reason: body.reason ?? null,
    });
  }),

  // Scopes endpoints
  http.get('/enforceai/scopes/catalog', () => {
    return HttpResponse.json(mockScopeCatalog);
  }),

  // Admin endpoints
  http.get('/enforceai/admin/users', ({ request }) => {
    const url = new URL(request.url);
    const query = url.searchParams.get('query') || '';
    const mockUsers = [
      {
        user_id: 'test|user123',
        email: 'test@example.com',
        username: 'testuser',
        auth_method: 'password',
        role: 'admin',
        last_seen_at: '2024-01-15T00:00:00Z',
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        user_id: 'local|admin',
        email: 'admin@example.com',
        username: 'admin',
        auth_method: 'password',
        role: 'admin',
        last_seen_at: '2024-01-16T00:00:00Z',
        created_at: '2024-01-01T00:00:00Z',
      },
    ];
    if (query) {
      return HttpResponse.json(
        mockUsers.filter(
          (u) =>
            u.email?.toLowerCase().includes(query.toLowerCase()) ||
            u.username?.toLowerCase().includes(query.toLowerCase())
        )
      );
    }
    return HttpResponse.json(mockUsers);
  }),

  http.get('/enforceai/admin/users/:userId', ({ params }) => {
    const mockUser = {
      user_id: params.userId,
      email: 'test@example.com',
      username: 'testuser',
      auth_method: 'password',
      role: 'admin',
      last_seen_at: '2024-01-15T00:00:00Z',
      created_at: '2024-01-01T00:00:00Z',
    };
    return HttpResponse.json(mockUser);
  }),

  http.get('/enforceai/admin/users/:userId/agents', () => {
    return HttpResponse.json(mockEnforceAIAgents);
  }),

  http.post('/enforceai/admin/users/:userId/agents', async ({ params, request }) => {
    const body = (await request.json()) as {
      scopes: string[];
      alias?: string;
      allowed_tools?: string[] | null;
      metadata?: Record<string, unknown> | null;
    };
    const newAgent: EnforceAIAgent = {
      user_id: String(params.userId),
      agent_id: 'admin-created-agent-' + Date.now(),
      scopes: body.scopes,
      allowed_tools: body.allowed_tools ?? null,
      alias: body.alias ?? null,
      metadata: body.metadata ?? null,
      revoked_at: null,
      tokens_valid_after: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    return HttpResponse.json(newAgent, { status: 201 });
  }),

  http.post('/enforceai/admin/users/:userId/agents/:agentId/revoke', ({ params }) => {
    const agent = mockEnforceAIAgents.find((a) => a.agent_id === params.agentId);
    if (!agent) {
      return HttpResponse.json({ detail: 'Agent not found' }, { status: 404 });
    }
    const revokedAgent: EnforceAIAgent = {
      ...agent,
      user_id: String(params.userId),
      revoked_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    return HttpResponse.json(revokedAgent);
  }),

  http.post('/enforceai/admin/users/:userId/agents/:agentId/tokens/revoke-all', ({ params }) => {
    const agent = mockEnforceAIAgents.find((a) => a.agent_id === params.agentId);
    if (!agent) {
      return HttpResponse.json({ detail: 'Agent not found' }, { status: 404 });
    }
    const updatedAgent: EnforceAIAgent = {
      ...agent,
      user_id: String(params.userId),
      tokens_valid_after: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    return HttpResponse.json(updatedAgent);
  }),

  http.get('/enforceai/admin/users/:userId/agents/:agentId/api-keys', ({ params }) => {
    const keys = mockApiKeys.filter((k) => k.agent_id === params.agentId);
    return HttpResponse.json(keys);
  }),

  http.post('/enforceai/admin/users/:userId/agents/:agentId/api-keys', async () => {
    const keyId = 'eak_admin_' + Date.now();
    return HttpResponse.json({
      key_id: keyId,
      secret: 'admin-test-secret-value-' + Date.now(),
      api_key_value: `${keyId}.admintestsecret`,
    });
  }),

  http.post('/enforceai/admin/users/:userId/api-keys/:keyId/revoke', ({ params }) => {
    const key = mockApiKeys.find((k) => k.key_id === params.keyId);
    if (!key) {
      return HttpResponse.json({ detail: 'API key not found' }, { status: 404 });
    }
    const revokedKey = {
      ...key,
      revoked_at: new Date().toISOString(),
    };
    return HttpResponse.json(revokedKey);
  }),

  http.post('/enforceai/admin/users/:userId/tokens/revoke', async ({ request, params }) => {
    const body = (await request.json()) as {
      agent_id: string;
      jti: string;
      reason?: string;
    };
    return HttpResponse.json({
      jti: body.jti,
      user_id: String(params.userId),
      agent_id: body.agent_id,
      revoked_at: new Date().toISOString(),
      expires_at: null,
      reason: body.reason ?? null,
    });
  }),

  // Egress Allowlist endpoints
  http.get('/enforceai/admin/egress-allowlist', () => {
    return HttpResponse.json(mockEgressAllowlistEntries);
  }),

  http.post('/enforceai/admin/egress-allowlist', async ({ request }) => {
    const body = (await request.json()) as {
      pattern: string;
      description?: string;
      expires_at?: string | null;
    };
    const newEntry: EgressAllowlistEntry = {
      entry_id: 'entry-' + Date.now(),
      pattern: body.pattern,
      description: body.description,
      expires_at: body.expires_at ?? null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    return HttpResponse.json(newEntry, { status: 201 });
  }),

  http.put('/enforceai/admin/egress-allowlist/:entryId', async ({ params, request }) => {
    const entry = mockEgressAllowlistEntries.find((e) => e.entry_id === params.entryId);
    if (!entry) {
      return HttpResponse.json({ detail: 'Entry not found' }, { status: 404 });
    }
    const body = (await request.json()) as {
      pattern?: string;
      description?: string;
      expires_at?: string | null;
    };
    const updatedEntry: EgressAllowlistEntry = {
      ...entry,
      pattern: body.pattern ?? entry.pattern,
      description: body.description !== undefined ? body.description : entry.description,
      expires_at: body.expires_at !== undefined ? body.expires_at : entry.expires_at,
      updated_at: new Date().toISOString(),
    };
    return HttpResponse.json(updatedEntry);
  }),

  http.delete('/enforceai/admin/egress-allowlist/:entryId', ({ params }) => {
    const entry = mockEgressAllowlistEntries.find((e) => e.entry_id === params.entryId);
    if (!entry) {
      return HttpResponse.json({ detail: 'Entry not found' }, { status: 404 });
    }
    return new HttpResponse(null, { status: 204 });
  }),

  http.post('/enforceai/admin/egress-allowlist/check', async ({ request }) => {
    const body = (await request.json()) as { pattern: string };
    // Simple mock check - always return allowed for now
    return HttpResponse.json({
      allowed: true,
      reason: 'Pattern matches allowlist',
    });
  }),

  // ============================================================================
  // Upstream Credentials API
  // ============================================================================

  // Mock upstream credentials storage (keyed by server path)
  // In real tests, this would be managed by the test setup

  http.get('/enforceai/upstream/servers/:serverPath/credentials', ({ params }) => {
    const serverPath = params.serverPath as string;

    // Mock OAuth credentials for OAuth-type servers
    if (serverPath === 'github-mcp') {
      const oauthCredential: UpstreamOAuthCredential = {
        credential_id: `cred-${serverPath}`,
        server_path: serverPath,
        credential_type: 'provider-oauth',
        binding: 'user',
        user_id: 'local|testuser',
        status: 'configured',
        provider: 'github',
        oauth_scopes: ['repo', 'read:user', 'read:org'],
        token_expires_at: new Date(Date.now() + 3600000).toISOString(), // 1 hour from now
        has_refresh_token: true,
        expires_at: null,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        last_used_at: '2024-01-15T10:30:00Z',
      };
      return HttpResponse.json(oauthCredential);
    }

    // Mock API key credentials for api-key type servers
    if (serverPath === 'sqlite') {
      return HttpResponse.json({
        credential_id: `cred-${serverPath}`,
        server_path: serverPath,
        credential_type: 'api-key',
        binding: 'user',
        user_id: 'local|testuser',
        status: 'configured',
        expires_at: null,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        last_used_at: '2024-01-15T10:30:00Z',
      });
    }

    // No credential configured for other servers
    return HttpResponse.json(
      { detail: 'No credential configured for this server' },
      { status: 404 }
    );
  }),

  http.post('/enforceai/upstream/servers/:serverPath/credentials', async ({ params, request }) => {
    const serverPath = params.serverPath as string;
    const body = (await request.json()) as {
      credential_type: 'api-key' | 'jwt';
      secret: string;
      expires_at?: string | null;
    };

    // Return the created credential with the secret (only returned once)
    return HttpResponse.json({
      credential_id: `cred-${serverPath}-${Date.now()}`,
      server_path: serverPath,
      credential_type: body.credential_type,
      secret: body.secret, // Echoed back once
      created_at: new Date().toISOString(),
    });
  }),

  http.post('/enforceai/upstream/credentials/:credentialId/revoke', ({ params }) => {
    const credentialId = params.credentialId as string;

    // Simple mock - just return success
    return new HttpResponse(null, { status: 204 });
  }),

  // ============================================================================
  // Upstream OAuth Flow API
  // ============================================================================

  http.post('/enforceai/upstream/servers/:serverPath/oauth/start', async ({ params, request }) => {
    const serverPath = params.serverPath as string;
    const body = (await request.json()) as { return_url?: string };

    // Build state with server info
    const stateData = {
      server_path: serverPath,
      return_url: body.return_url,
    };
    const state = btoa(JSON.stringify(stateData));

    // Return mock authorization URL (in tests, we don't actually redirect)
    return HttpResponse.json({
      authorization_url: `https://mock-oauth-provider.test/authorize?state=${state}&client_id=mock-client&redirect_uri=${encodeURIComponent(body.return_url || '')}`,
      state,
    });
  }),

  http.post('/enforceai/upstream/servers/:serverPath/oauth/disconnect', async ({ params }) => {
    const serverPath = params.serverPath as string;

    // Simple mock - just return success
    return new HttpResponse(null, { status: 204 });
  }),
];
