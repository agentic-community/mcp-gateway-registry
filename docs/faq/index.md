# Frequently Asked Questions

Common questions and answers about the MCP Gateway Registry.

## Getting Started

- [What is MCP and why do I need a gateway?](what-is-mcp-and-gateway.md)
- [How do I deploy and register MCP servers and agents?](deploying-and-registering-servers-agents.md)

## Tool and Agent Discovery

- [How do I discover available MCP tools for my AI agent?](discovering-mcp-tools.md)
- [How do I handle tool discovery when I don't know what tools are available?](agent-autonomous-tool-discovery.md)
- [What filtering options are available for agents in the registry?](filtering-agents-by-tags-and-fields.md)

## Connecting and Integration

- [How do I get my AI coding assistant to work with this registry?](connect-ai-coding-assistant.md)
- [How do I connect my agent to multiple MCP servers through the gateway?](connecting-multiple-mcp-servers.md)
- [How do I test my agent's integration with the MCP Gateway locally?](local-testing-agent-integration.md)

## Operations and Monitoring

- [How do I monitor the health of MCP servers?](monitoring-server-health.md)
- [How do I configure MongoDB Atlas instead of MongoDB CE?](configuring-mongodb-atlas-backend.md)
- [Why are some of my assets not showing up in semantic search?](fix-missing-search-embeddings.md)
- [Why do I sometimes see search results for assets that no longer exist?](fix-stale-search-embeddings.md)

## Amazon Bedrock AgentCore

- [How do I bulk-register all AgentCore Gateways and Runtimes from my AWS account?](agentcore-bulk-registration.md)

## Deployment Customization

- [How do I add custom environment variables to the registry, auth-server, or mcpgw services?](adding-custom-env-vars.md)

## Access Control and Visibility

- [How do I restrict which agents a user can see based on their group?](group-restricted-agent-visibility.md)
- [How do I restrict which MCP servers a user can see based on their Entra ID group?](restrict-server-visibility-by-entra-group.md)
- [How do I create a non-admin group that can register servers and run health checks but cannot toggle, edit, or delete them?](read-write-non-admin-group.md)
- [How do I set up a self-service workflow for AI assets (draft to review to active)?](self-service-asset-lifecycle-workflow.md)

## Authentication and API Access

- [How do I register and manage MCP servers that require authentication?](registering-auth-protected-servers.md)
- [Can I use an Entra ID token to call the registry API instead of the UI-generated token?](use-entra-token-for-registry-api.md)
- [How do I register an M2M client and assign it groups without an IdP Admin API token?](registering-m2m-client-without-idp-admin-token.md)
- [Registry API Authentication FAQ (static token, IdP JWT, coexistence)](registry-api-auth-faq.md)
- [How do I pass an M2M token from Entra to the registration gate?](oauth2-token-for-registration-gate.md)
