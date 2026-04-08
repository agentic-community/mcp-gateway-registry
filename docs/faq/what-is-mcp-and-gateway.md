# What is the Model Context Protocol (MCP) and why do I need a gateway?

**Model Context Protocol (MCP)** is an open standard that allows AI models to connect with external systems, tools, and data sources.

## Why You Need a Gateway

- **Service Discovery**: Find approved MCP servers in your organization
- **Centralized Access Control**: Secure, governed access to tools
- **Dynamic Tool Discovery**: Agents can find new tools autonomously
- **Simplified Client Configuration**: Single endpoint for multiple servers
- **Enterprise Security**: Authentication, authorization, and audit logging

**Without Gateway**: Each agent connects directly to individual MCP servers
**With Gateway**: All agents connect through a single, secure, managed endpoint

## What's the difference between the Registry and the Gateway?

They are complementary components:

**Registry**:
- **Purpose**: Service discovery and management
- **Features**: Web UI, server registration, health monitoring, tool catalog
- **Users**: Platform administrators, developers
- **Access**: Web browser at port 80 (HTTP) or 443 (HTTPS) via nginx reverse proxy

**Gateway**:
- **Purpose**: Secure proxy for MCP protocol traffic
- **Features**: Authentication, authorization, request routing
- **Users**: AI agents, MCP clients
- **Access**: MCP protocol at `/server-name/sse`

**Together**: Registry manages what's available, Gateway controls access to it.

## Related Documentation

- [Quick Start Guide](../quickstart.md) -- getting started
- [Installation Guide](../installation.md) -- deployment options
- [Architecture](../design/architectural-decision-reverse-proxy-vs-application-layer-gateway.md) -- architectural design decisions
